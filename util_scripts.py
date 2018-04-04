# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import time
import re
import bisect
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import scipy.ndimage
import scipy.misc

import config
import misc
import tfutil
import train
import dataset

#----------------------------------------------------------------------------
# Generate random images or image grids using a previously trained network.
# To run, uncomment the appropriate line in config.py and launch train.py.

def generate_fake_images(run_id, snapshot=None, grid_size=[1,1], num_pngs=1, image_shrink=1, png_prefix=None, random_seed=1000, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    if png_prefix is None:
        png_prefix = misc.get_id_string_for_network_pkl(network_pkl) + '-'
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    for png_idx in range(num_pngs):
        print('Generating png %d / %d...' % (png_idx, num_pngs))
        latents = misc.random_latents(np.prod(grid_size), Gs, random_state=random_state)
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        misc.save_image_grid(images, os.path.join(result_subdir, '%s%06d.png' % (png_prefix, png_idx)), [0,255], grid_size)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Generate MP4 video of random interpolations using a previously trained network.
# To run, uncomment the appropriate line in config.py and launch train.py.

def generate_interpolation_video(run_id, snapshot=None, grid_size=[1,1], image_shrink=1, image_zoom=1, duration_sec=60.0, smoothing_sec=1.0, mp4=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    if mp4 is None:
        mp4 = misc.get_id_string_for_network_pkl(network_pkl) + '-lerp.mp4'
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    print('Generating latent vectors...')
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        grid = misc.create_image_grid(images, grid_size).transpose(1, 2, 0) # HWC
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2) # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor # pip install moviepy
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, mp4), fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Generate MP4 video of training progress for a previous training run.
# To run, uncomment the appropriate line in config.py and launch train.py.

def generate_training_video(run_id, duration_sec=20.0, time_warp=1.5, mp4=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M'):
    src_result_subdir = misc.locate_result_subdir(run_id)
    if mp4 is None:
        mp4 = os.path.basename(src_result_subdir) + '-train.mp4'

    # Parse log.
    times = []
    snaps = [] # [(png, kimg, lod), ...]
    with open(os.path.join(src_result_subdir, 'log.txt'), 'rt') as log:
        for line in log:
            k = re.search(r'kimg ([\d\.]+) ', line)
            l = re.search(r'lod ([\d\.]+) ', line)
            t = re.search(r'time (\d+d)? *(\d+h)? *(\d+m)? *(\d+s)? ', line)
            if k and l and t:
                k = float(k.group(1))
                l = float(l.group(1))
                t = [int(t.group(i)[:-1]) if t.group(i) else 0 for i in range(1, 5)]
                t = t[0] * 24*60*60 + t[1] * 60*60 + t[2] * 60 + t[3]
                png = os.path.join(src_result_subdir, 'fakes%06d.png' % int(np.floor(k)))
                if os.path.isfile(png):
                    times.append(t)
                    snaps.append((png, k, l))
    assert len(times)

    # Frame generation func for moviepy.
    png_cache = [None, None] # [png, img]
    def make_frame(t):
        wallclock = ((t / duration_sec) ** time_warp) * times[-1]
        png, kimg, lod = snaps[max(bisect.bisect(times, wallclock) - 1, 0)]
        if png_cache[0] == png:
            img = png_cache[1]
        else:
            img = scipy.misc.imread(png)
            while img.shape[1] > 1920 or img.shape[0] > 1080:
                img = img.astype(np.float32).reshape(img.shape[0]//2, 2, img.shape[1]//2, 2, -1).mean(axis=(1,3))
            png_cache[:] = [png, img]
        img = misc.draw_text_label(img, 'lod %.2f' % lod, 16, img.shape[0]-4, alignx=0.0, aligny=1.0)
        img = misc.draw_text_label(img, misc.format_time(int(np.rint(wallclock))), img.shape[1]//2, img.shape[0]-4, alignx=0.5, aligny=1.0)
        img = misc.draw_text_label(img, '%.0f kimg' % kimg, img.shape[1]-16, img.shape[0]-4, alignx=1.0, aligny=1.0)
        return img

    # Generate video.
    import moviepy.editor # pip install moviepy
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, mp4), fps=mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Evaluate one or more metrics for a previous training run.
# To run, uncomment one of the appropriate lines in config.py and launch train.py.

def evaluate_metrics(run_id, log, metrics, num_images, real_passes, minibatch_size=None):
    metric_class_names = {
        'swd':      'metrics.sliced_wasserstein.API',
        'fid':      'metrics.frechet_inception_distance.API',
        'is':       'metrics.inception_score.API',
        'msssim':   'metrics.ms_ssim.API',
    }

    # Locate training run and initialize logging.
    result_subdir = misc.locate_result_subdir(run_id)
    snapshot_pkls = misc.list_network_pkls(result_subdir, include_final=False)
    assert len(snapshot_pkls) >= 1
    log_file = os.path.join(result_subdir, log)
    print('Logging output to', log_file)
    misc.set_output_log_file(log_file)

    # Initialize dataset and select minibatch size.
    dataset_obj, mirror_augment = misc.load_dataset_for_previous_run(result_subdir, verbose=True, shuffle_mb=0)
    if minibatch_size is None:
        minibatch_size = np.clip(8192 // dataset_obj.shape[1], 4, 256)

    # Initialize metrics.
    metric_objs = []
    for name in metrics:
        class_name = metric_class_names.get(name, name)
        print('Initializing %s...' % class_name)
        class_def = tfutil.import_obj(class_name)
        image_shape = [3] + dataset_obj.shape[1:]
        obj = class_def(num_images=num_images, image_shape=image_shape, image_dtype=np.uint8, minibatch_size=minibatch_size)
        tfutil.init_uninited_vars()
        mode = 'warmup'
        obj.begin(mode)
        for idx in range(10):
            obj.feed(mode, np.random.randint(0, 256, size=[minibatch_size]+image_shape, dtype=np.uint8))
        obj.end(mode)
        metric_objs.append(obj)

    # Print table header.
    print()
    print('%-10s%-12s' % ('Snapshot', 'Time_eval'), end='')
    for obj in metric_objs:
        for name, fmt in zip(obj.get_metric_names(), obj.get_metric_formatting()):
            print('%-*s' % (len(fmt % 0), name), end='')
    print()
    print('%-10s%-12s' % ('---', '---'), end='')
    for obj in metric_objs:
        for fmt in obj.get_metric_formatting():
            print('%-*s' % (len(fmt % 0), '---'), end='')
    print()

    # Feed in reals.
    for title, mode in [('Reals', 'reals'), ('Reals2', 'fakes')][:real_passes]:
        print('%-10s' % title, end='')
        time_begin = time.time()
        labels = np.zeros([num_images, dataset_obj.label_size], dtype=np.float32)
        [obj.begin(mode) for obj in metric_objs]
        for begin in range(0, num_images, minibatch_size):
            end = min(begin + minibatch_size, num_images)
            images, labels[begin:end] = dataset_obj.get_minibatch_np(end - begin)
            if mirror_augment:
                images = misc.apply_mirror_augment(images)
            if images.shape[1] == 1:
                images = np.tile(images, [1, 3, 1, 1]) # grayscale => RGB
            [obj.feed(mode, images) for obj in metric_objs]
        results = [obj.end(mode) for obj in metric_objs]
        print('%-12s' % misc.format_time(time.time() - time_begin), end='')
        for obj, vals in zip(metric_objs, results):
            for val, fmt in zip(vals, obj.get_metric_formatting()):
                print(fmt % val, end='')
        print()

    # Evaluate each network snapshot.
    for snapshot_idx, snapshot_pkl in enumerate(reversed(snapshot_pkls)):
        prefix = 'network-snapshot-'; postfix = '.pkl'
        snapshot_name = os.path.basename(snapshot_pkl)
        assert snapshot_name.startswith(prefix) and snapshot_name.endswith(postfix)
        snapshot_kimg = int(snapshot_name[len(prefix) : -len(postfix)])

        print('%-10d' % snapshot_kimg, end='')
        mode ='fakes'
        [obj.begin(mode) for obj in metric_objs]
        time_begin = time.time()
        with tf.Graph().as_default(), tfutil.create_session(config.tf_config).as_default():
            G, D, Gs = misc.load_pkl(snapshot_pkl)
            for begin in range(0, num_images, minibatch_size):
                end = min(begin + minibatch_size, num_images)
                latents = misc.random_latents(end - begin, Gs)
                images = Gs.run(latents, labels[begin:end], num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_dtype=np.uint8)
                if images.shape[1] == 1:
                    images = np.tile(images, [1, 3, 1, 1]) # grayscale => RGB
                [obj.feed(mode, images) for obj in metric_objs]
        results = [obj.end(mode) for obj in metric_objs]
        print('%-12s' % misc.format_time(time.time() - time_begin), end='')
        for obj, vals in zip(metric_objs, results):
            for val, fmt in zip(vals, obj.get_metric_formatting()):
                print(fmt % val, end='')
        print()
    print()

#----------------------------------------------------------------------------
