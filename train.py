# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
from __future__ import print_function

import os
import sys
import time
import glob
import shutil
import operator
import six
import numpy as np
import scipy.ndimage

import misc
misc.init_output_logging()

if __name__ == "__main__":
    print('Importing Theano...')

import config
os.environ['THEANO_FLAGS'] = ','.join([key + '=' + value for key, value in six.iteritems(config.theano_flags)])
sys.setrecursionlimit(10000)
import theano
from theano import tensor as T
import lasagne

import network
import dataset

#----------------------------------------------------------------------------
# Convenience.

def Tsum (*args, **kwargs): return T.sum (*args, dtype=theano.config.floatX, acc_dtype=theano.config.floatX, **kwargs)
def Tmean(*args, **kwargs): return T.mean(*args, dtype=theano.config.floatX, acc_dtype=theano.config.floatX, **kwargs)

def adam(loss, params, **kwargs):
    connected_params = []
    connected_grads = []
    for p in params:
        try:
            g = theano.grad(loss, p)
            connected_params.append(p)
            connected_grads.append(g)
        except theano.gradient.DisconnectedInputError:
            pass
    return lasagne.updates.adam(connected_grads, connected_params, **kwargs)

def random_latents(num_latents, G_input_shape):
    return np.random.randn(num_latents, *G_input_shape[1:]).astype(np.float32)

def random_labels(num_labels, training_set):
    return training_set.labels[np.random.randint(training_set.labels.shape[0], size=num_labels)]

def load_dataset(dataset_spec=None, verbose=False, **spec_overrides):
    if verbose: print('Loading dataset...')
    if dataset_spec is None: dataset_spec = config.dataset
    dataset_spec = dict(dataset_spec) # take a copy of the dict before modifying it
    dataset_spec.update(spec_overrides)
    dataset_spec['h5_path'] = os.path.join(config.data_dir, dataset_spec['h5_path'])
    if 'label_path' in dataset_spec: dataset_spec['label_path'] = os.path.join(config.data_dir, dataset_spec['label_path'])
    training_set = dataset.Dataset(**dataset_spec)
    if verbose: print('Dataset shape =', np.int32(training_set.shape).tolist())
    drange_orig = training_set.get_dynamic_range()
    if verbose: print('Dynamic range =', drange_orig)
    return training_set, drange_orig

def load_dataset_for_previous_run(result_subdir, **kwargs):
    dataset = None
    with open(os.path.join(result_subdir, 'config.txt'), 'rt') as f:
        for line in f:
            if line.startswith('dataset = '):
                exec(line)
    return load_dataset(dataset, **kwargs)

#----------------------------------------------------------------------------

def train_gan(
    separate_funcs          = False,
    D_training_repeats      = 1,
    G_learning_rate_max     = 0.0010,
    D_learning_rate_max     = 0.0010,
    G_smoothing             = 0.999,
    adam_beta1              = 0.0,
    adam_beta2              = 0.99,
    adam_epsilon            = 1e-8,
    minibatch_default       = 16,
    minibatch_overrides     = {},
    rampup_kimg             = 40,
    rampdown_kimg           = 0,
    lod_initial_resolution  = 4,
    lod_training_kimg       = 400,
    lod_transition_kimg     = 400,
    total_kimg              = 10000,
    dequantize_reals        = False,
    gdrop_beta              = 0.9,
    gdrop_lim               = 0.5,
    gdrop_coef              = 0.2,
    gdrop_exp               = 2.0,
    drange_net              = [-1,1],
    drange_viz              = [-1,1],
    image_grid_size         = None,
    tick_kimg_default       = 50,
    tick_kimg_overrides     = {32:20, 64:10, 128:10, 256:5, 512:2, 1024:1},
    image_snapshot_ticks    = 4,
    network_snapshot_ticks  = 40,
    image_grid_type         = 'default',
    resume_network_pkl      = None,
    resume_kimg             = 0.0,
    resume_time             = 0.0):

    # Load dataset and build networks.
    training_set, drange_orig = load_dataset()
    if resume_network_pkl:
        print('Resuming', resume_network_pkl)
        G, D, _ = misc.load_pkl(os.path.join(config.result_dir, resume_network_pkl))
    else:
        G = network.Network(num_channels=training_set.shape[1], resolution=training_set.shape[2], label_size=training_set.labels.shape[1], **config.G)
        D = network.Network(num_channels=training_set.shape[1], resolution=training_set.shape[2], label_size=training_set.labels.shape[1], **config.D)
    Gs = G.create_temporally_smoothed_version(beta=G_smoothing, explicit_updates=True)
    misc.print_network_topology_info(G.output_layers)
    misc.print_network_topology_info(D.output_layers)

    # Setup snapshot image grid.
    if image_grid_type == 'default':
        if image_grid_size is None:
            w, h = G.output_shape[3], G.output_shape[2]
            image_grid_size = np.clip(1920 / w, 3, 16), np.clip(1080 / h, 2, 16)
        example_real_images, snapshot_fake_labels = training_set.get_random_minibatch(np.prod(image_grid_size), labels=True)
        snapshot_fake_latents = random_latents(np.prod(image_grid_size), G.input_shape)
    elif image_grid_type == 'category':
        W = training_set.labels.shape[1]
        H = W if image_grid_size is None else image_grid_size[1]
        image_grid_size = W, H
        snapshot_fake_latents = random_latents(W*H, G.input_shape)
        snapshot_fake_labels = np.zeros((W*H, W), dtype=training_set.labels.dtype)
        example_real_images = np.zeros((W*H,) + training_set.shape[1:], dtype=training_set.dtype)
        for x in xrange(W):
            snapshot_fake_labels[x::W, x] = 1.0
            indices = np.arange(training_set.shape[0])[training_set.labels[:,x] != 0]
            for y in xrange(H):
                example_real_images[x + y * W] = training_set.h5_lods[0][np.random.choice(indices)]
    else:
        raise ValueError('Invalid image_grid_type', image_grid_type)

    # Theano input variables and compile generation func.
    print('Setting up Theano...')
    real_images_var  = T.TensorType('float32', [False] * len(D.input_shape))            ('real_images_var')
    real_labels_var  = T.TensorType('float32', [False] * len(training_set.labels.shape))('real_labels_var')
    fake_latents_var = T.TensorType('float32', [False] * len(G.input_shape))            ('fake_latents_var')
    fake_labels_var  = T.TensorType('float32', [False] * len(training_set.labels.shape))('fake_labels_var')
    G_lrate = theano.shared(np.float32(0.0))
    D_lrate = theano.shared(np.float32(0.0))
    gen_fn = theano.function([fake_latents_var, fake_labels_var], Gs.eval_nd(fake_latents_var, fake_labels_var, ignore_unused_inputs=True), on_unused_input='ignore')

    # Misc init.
    resolution_log2 = int(np.round(np.log2(G.output_shape[2])))
    initial_lod = max(resolution_log2 - int(np.round(np.log2(lod_initial_resolution))), 0)
    cur_lod = 0.0
    min_lod, max_lod = -1.0, -2.0
    fake_score_avg = 0.0

    if config.D.get('mbdisc_kernels', None):
        print('Initializing minibatch discrimination...')
        if hasattr(D, 'cur_lod'): D.cur_lod.set_value(np.float32(initial_lod))
        D.eval(real_images_var, deterministic=False, init=True)
        init_layers = lasagne.layers.get_all_layers(D.output_layers)
        init_updates = [update for layer in init_layers for update in getattr(layer, 'init_updates', [])]
        init_fn = theano.function(inputs=[real_images_var], outputs=None, updates=init_updates)
        init_reals = training_set.get_random_minibatch(500, lod=initial_lod)
        init_reals = misc.adjust_dynamic_range(init_reals, drange_orig, drange_net)
        init_fn(init_reals)
        del init_reals

    # Save example images.
    snapshot_fake_images = gen_fn(snapshot_fake_latents, snapshot_fake_labels)
    result_subdir = misc.create_result_subdir(config.result_dir, config.run_desc)
    misc.save_image_grid(example_real_images, os.path.join(result_subdir, 'reals.png'), drange=drange_orig, grid_size=image_grid_size)
    misc.save_image_grid(snapshot_fake_images, os.path.join(result_subdir, 'fakes%06d.png' % 0), drange=drange_viz, grid_size=image_grid_size)

    # Training loop.
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    tick_train_out = []
    train_start_time = tick_start_time - resume_time
    while cur_nimg < total_kimg * 1000:

        # Calculate current LOD.
        cur_lod = initial_lod
        if lod_training_kimg or lod_transition_kimg:
            tlod = (cur_nimg / 1000.0) / (lod_training_kimg + lod_transition_kimg)
            cur_lod -= np.floor(tlod)
            if lod_transition_kimg:
                cur_lod -= max(1.0 + (np.fmod(tlod, 1.0) - 1.0) * (lod_training_kimg + lod_transition_kimg) / lod_transition_kimg, 0.0)
            cur_lod = max(cur_lod, 0.0)

        # Look up resolution-dependent parameters.
        cur_res = 2 ** (resolution_log2 - int(np.floor(cur_lod)))
        minibatch_size = minibatch_overrides.get(cur_res, minibatch_default)
        tick_duration_kimg = tick_kimg_overrides.get(cur_res, tick_kimg_default)

        # Update network config.
        lrate_coef = misc.rampup(cur_nimg / 1000.0, rampup_kimg)
        lrate_coef *= misc.rampdown_linear(cur_nimg / 1000.0, total_kimg, rampdown_kimg)
        G_lrate.set_value(np.float32(lrate_coef * G_learning_rate_max))
        D_lrate.set_value(np.float32(lrate_coef * D_learning_rate_max))
        if hasattr(G, 'cur_lod'): G.cur_lod.set_value(np.float32(cur_lod))
        if hasattr(D, 'cur_lod'): D.cur_lod.set_value(np.float32(cur_lod))

        # Setup training func for current LOD.
        new_min_lod, new_max_lod = int(np.floor(cur_lod)), int(np.ceil(cur_lod))
        if min_lod != new_min_lod or max_lod != new_max_lod:
            print('Compiling training funcs...')
            min_lod, max_lod = new_min_lod, new_max_lod

            # Pre-process reals.
            real_images_expr = real_images_var
            if dequantize_reals:
                rnd = theano.sandbox.rng_mrg.MRG_RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
                epsilon_noise = rnd.uniform(size=real_images_expr.shape, low=-0.5, high=0.5, dtype='float32')
                real_images_expr = T.cast(real_images_expr, 'float32') + epsilon_noise # match original implementation of Improved Wasserstein
            real_images_expr = misc.adjust_dynamic_range(real_images_expr, drange_orig, drange_net)
            if min_lod > 0: # compensate for shrink_based_on_lod
                real_images_expr = T.extra_ops.repeat(real_images_expr, 2**min_lod, axis=2)
                real_images_expr = T.extra_ops.repeat(real_images_expr, 2**min_lod, axis=3)

            # Optimize loss.
            G_loss, D_loss, real_scores_out, fake_scores_out = evaluate_loss(G, D, min_lod, max_lod, real_images_expr, real_labels_var, fake_latents_var, fake_labels_var, **config.loss)
            G_updates = adam(G_loss, G.trainable_params(), learning_rate=G_lrate, beta1=adam_beta1, beta2=adam_beta2, epsilon=adam_epsilon).items()
            D_updates = adam(D_loss, D.trainable_params(), learning_rate=D_lrate, beta1=adam_beta1, beta2=adam_beta2, epsilon=adam_epsilon).items()

            # Compile training funcs.
            if not separate_funcs:
                GD_train_fn = theano.function(
                    [real_images_var, real_labels_var, fake_latents_var, fake_labels_var],
                    [G_loss, D_loss, real_scores_out, fake_scores_out],
                    updates=G_updates+D_updates+Gs.updates,
                    on_unused_input='ignore')
            else:
                D_train_fn = theano.function(
                    [real_images_var, real_labels_var, fake_latents_var, fake_labels_var],
                    [G_loss, D_loss, real_scores_out, fake_scores_out],
                    updates=D_updates, on_unused_input='ignore')
                G_train_fn = theano.function(
                    [fake_latents_var, fake_labels_var],
                    [],
                    updates=G_updates+Gs.updates, on_unused_input='ignore')

        # Invoke training funcs.
        if not separate_funcs:
            assert D_training_repeats == 1
            mb_reals, mb_labels = training_set.get_random_minibatch(minibatch_size, lod=cur_lod, shrink_based_on_lod=True, labels=True)
            mb_train_out = GD_train_fn(mb_reals, mb_labels, random_latents(minibatch_size, G.input_shape), random_labels(minibatch_size, training_set))
            cur_nimg += minibatch_size
            tick_train_out.append(mb_train_out)
        else:
            for idx in xrange(D_training_repeats):
                mb_reals, mb_labels = training_set.get_random_minibatch(minibatch_size, lod=cur_lod, shrink_based_on_lod=True, labels=True)
                mb_train_out = D_train_fn(mb_reals, mb_labels, random_latents(minibatch_size, G.input_shape), random_labels(minibatch_size, training_set))
                cur_nimg += minibatch_size
                tick_train_out.append(mb_train_out)
            G_train_fn(random_latents(minibatch_size, G.input_shape), random_labels(minibatch_size, training_set))

        # Fade in D noise if we're close to becoming unstable
        fake_score_cur = np.clip(np.mean(mb_train_out[1]), 0.0, 1.0)
        fake_score_avg = fake_score_avg * gdrop_beta + fake_score_cur * (1.0 - gdrop_beta)
        gdrop_strength = gdrop_coef * (max(fake_score_avg - gdrop_lim, 0.0) ** gdrop_exp)
        if hasattr(D, 'gdrop_strength'): D.gdrop_strength.set_value(np.float32(gdrop_strength))

        # Perform maintenance operations once per tick.
        if cur_nimg >= tick_start_nimg + tick_duration_kimg * 1000 or cur_nimg >= total_kimg * 1000:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            tick_start_time = cur_time
            tick_train_avg = tuple(np.mean(np.concatenate([np.asarray(v).flatten() for v in vals])) for vals in zip(*tick_train_out))
            tick_train_out = []

            # Print progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-9.1f sec/kimg %-6.1f Dgdrop %-8.4f Gloss %-8.4f Dloss %-8.4f Dreal %-8.4f Dfake %-8.4f' % (
                (cur_tick, cur_nimg / 1000.0, cur_lod, minibatch_size, misc.format_time(cur_time - train_start_time), tick_time, tick_time / tick_kimg, gdrop_strength) + tick_train_avg))

            # Visualize generated images.
            if cur_tick % image_snapshot_ticks == 0 or cur_nimg >= total_kimg * 1000:
                snapshot_fake_images = gen_fn(snapshot_fake_latents, snapshot_fake_labels)
                misc.save_image_grid(snapshot_fake_images, os.path.join(result_subdir, 'fakes%06d.png' % (cur_nimg / 1000)), drange=drange_viz, grid_size=image_grid_size)

            # Save network snapshot every N ticks.
            if cur_tick % network_snapshot_ticks == 0 or cur_nimg >= total_kimg * 1000:
                misc.save_pkl((G, D, Gs), os.path.join(result_subdir, 'network-snapshot-%06d.pkl' % (cur_nimg / 1000)))

    # Write final results.
    misc.save_pkl((G, D, Gs), os.path.join(result_subdir, 'network-final.pkl'))
    training_set.close()
    print('Done.')
    with open(os.path.join(result_subdir, '_training-done.txt'), 'wt'):
        pass

#----------------------------------------------------------------------------

def evaluate_loss(
    G, D, min_lod, max_lod, real_images_in,
    real_labels_in, fake_latents_in, fake_labels_in,
    type            = 'iwass',
    L2_fake_weight  = 0.1,
    iwass_lambda    = 10.0,
    iwass_epsilon   = 0.001,
    iwass_target    = 1.0,
    cond_type       = 'acgan',
    cond_weight     = 1.0,
    cond_tweak_G    = 1.0): # set cond_tweak_G=0.1 to match original improved Wasserstein implementation

    # Helpers.
    def L2(a, b): return 0 if a is None or b is None else Tmean(T.square(a - b))
    def crossent(a, b): return 0 if a is None or b is None else Tmean(lasagne.objectives.categorical_crossentropy(lasagne.nonlinearities.softmax(a), b))
    rnd = theano.sandbox.rng_mrg.MRG_RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

    # Evaluate generator.
    fake_images_out = G.eval_nd(fake_latents_in, fake_labels_in, min_lod=min_lod, max_lod=max_lod, ignore_unused_inputs=True)

    # Mix reals and fakes through linear crossfade.
    mixing_factors = rnd.uniform((real_images_in.shape[0], 1, 1, 1), dtype='float32')
    mixed_images_out = real_images_in * (1 - mixing_factors) + fake_images_out * mixing_factors

    # Evaluate discriminator.
    real_scores_out,  real_labels_out  = D.eval_nd(real_images_in,   min_lod=min_lod, max_lod=max_lod, expect_num_outputs=2)
    fake_scores_out,  fake_labels_out  = D.eval_nd(fake_images_out,  min_lod=min_lod, max_lod=max_lod, expect_num_outputs=2)
    mixed_scores_out, mixed_labels_out = D.eval_nd(mixed_images_out, min_lod=min_lod, max_lod=max_lod, expect_num_outputs=2)

    if type == 'iwass': # Improved Wasserstein
        mixed_grads = theano.grad(Tsum(mixed_scores_out), mixed_images_out)
        mixed_norms = T.sqrt(Tsum(T.square(mixed_grads), axis=(1,2,3)))
        G_loss = -Tmean(fake_scores_out)
        D_loss = (Tmean(fake_scores_out) - Tmean(real_scores_out)) + Tmean(T.square(mixed_norms - iwass_target)) * iwass_lambda / (iwass_target**2)
        D_loss += L2(real_scores_out, 0) * iwass_epsilon # additional penalty term to keep the scores from drifting too far from zero
        fake_scores_out = fake_scores_out - real_scores_out # reporting tweak
        real_scores_out = T.constant(0) # reporting tweak

    if type == 'lsgan': # LSGAN
        G_loss = L2(fake_scores_out, 0)
        D_loss = L2(real_scores_out, 0) + L2(fake_scores_out, 1) * L2_fake_weight

    if cond_type == 'acgan': # AC-GAN
        G_loss += crossent(fake_labels_out, fake_labels_in) * cond_weight
        D_loss += (crossent(real_labels_out, real_labels_in) + crossent(fake_labels_out, fake_labels_in)) * cond_weight

    return G_loss, D_loss, real_scores_out, fake_scores_out

#----------------------------------------------------------------------------
# Image generation API.

def imgapi_load_net(run_id, snapshot=None, random_seed=1000, num_example_latents=1000, compile_gen_fn=True):
    class Net: pass
    net = Net()
    net.result_subdir = misc.locate_result_subdir(run_id)
    net.network_pkl = misc.locate_network_pkl(net.result_subdir, snapshot)
    _, _, net.G = misc.load_pkl(net.network_pkl)

    # Generate example latents and labels.
    np.random.seed(random_seed)
    net.example_latents = random_latents(num_example_latents, net.G.input_shape)
    net.training_set, net.dynamic_range = load_dataset_for_previous_run(net.result_subdir, verbose=False)
    net.example_labels = net.training_set.labels

    # Compile Theano func.
    net.latents_var = T.TensorType('float32', [False] * len(net.example_latents.shape))('latents_var')
    net.labels_var  = T.TensorType('float32', [False] * len(net.example_labels.shape)) ('labels_var')

    if hasattr(net.G, 'cur_lod'):
        net.lod = net.G.cur_lod.get_value()
        net.images_expr = net.G.eval(net.latents_var, net.labels_var, min_lod=net.lod, max_lod=net.lod, ignore_unused_inputs=True)
    else:
        net.lod = 0.0
        net.images_expr = net.G.eval(net.latents_var, net.labels_var, ignore_unused_inputs=True)

    net.images_expr = misc.adjust_dynamic_range(net.images_expr, [-1,1], net.dynamic_range)
    if compile_gen_fn:
        imgapi_compile_gen_fn(net)
    return net

def imgapi_compile_gen_fn(net):
    net.gen_fn = theano.function([net.latents_var, net.labels_var], net.images_expr, on_unused_input='ignore')

def imgapi_generate_batch(net, latents, labels, minibatch_size=16, convert_to_uint8=False):
    assert latents.shape[0] == labels.shape[0]
    dtype = np.uint8 if convert_to_uint8 else np.float32
    images = np.zeros((latents.shape[0],) + net.G.output_shape[1:], dtype=dtype)
    for begin in xrange(0, latents.shape[0], minibatch_size):
        end = min(begin + minibatch_size, latents.shape[0])
        tmp = net.gen_fn(latents[begin:end], labels[begin:end])
        if convert_to_uint8:
            tmp = np.round(tmp).clip(0, 255).astype(np.uint8)
        images[begin:end] = tmp
    return images

def imgapi_example(run_id, snapshot):
    net = imgapi_load_net(run_id, snapshot)
    images = net.gen_fn(net.example_latents[:1], net.example_labels[:1])
    # latents: [minibatch, component], normalized automatically by the network, value represents a point on the unit hypersphere
    # labels:  [minibatch, component], value depends on the dataset and training config
    # images:  [minibatch, channel, height, width], dynamic range 0--255
    misc.save_image(images[0], os.path.join(config.result_dir, 'debug.png'), drange=[0,255])

#----------------------------------------------------------------------------

def interpolate_latents(
    run_id,
    snapshot,
    video_fps       = 30,
    filter_frames   = 30,
    num_frames      = 60*30,
    drange_net      = [-1,1],
    image_grid_size = None,
    zoom            = None,
    video_bitrate   = '16M'):

    import moviepy.editor # pip install moviepy

    # Choose parameters.
    net = imgapi_load_net(run_id=run_id, snapshot=snapshot)
    w, h = net.G.output_shape[3], net.G.output_shape[2]
    if image_grid_size is None and zoom is None: image_grid_size = (1, 1)
    if zoom is None: zoom = max(min(1920 / w, 1080 / h), 1)
    if image_grid_size is None: image_grid_size = np.clip(int(np.floor(1920 / (w * zoom))), 1, 16), np.clip(int(np.floor(1080 / (h * zoom))), 1, 16)

    # Generate latent vectors (frame, image, channel, component).
    print('Generating latent vectors...')
    latents = np.random.randn(num_frames, np.prod(image_grid_size), *net.G.input_shape[1:]).astype(np.float32)
    latents = scipy.ndimage.gaussian_filter(latents, [filter_frames] + [0] * len(net.G.input_shape), mode='wrap')
    latents /= np.sqrt(np.mean(latents ** 2))

    # Create video.
    print('Generating video...')
    result_subdir = misc.create_result_subdir(config.result_dir, config.run_desc)
    def make_frame(t):
        frame_idx = np.clip(int(np.round(t * video_fps)), 0, num_frames - 1)
        images = net.gen_fn(latents[frame_idx], net.example_labels[:latents.shape[1]])
        grid = misc.create_image_grid(images, grid_size=image_grid_size)
        if zoom != 1: grid = scipy.ndimage.zoom(grid, [1, zoom, zoom], order=0)
        grid = grid.clip(0, 255).transpose(1, 2, 0) # CHW => HWC
        if grid.shape[2] == 1: grid = grid.repeat(3, 2) # grayscale => RGB
        return grid
    video = moviepy.editor.VideoClip(make_frame, duration=float(num_frames)/video_fps)
    video.write_videofile(os.path.join(result_subdir, os.path.basename(result_subdir) + '.mp4'), fps=video_fps, codec='libx264', bitrate=video_bitrate)

    # Done.
    print('Done.')
    with open(os.path.join(result_subdir, '_video-done.txt'), 'wt'):
        pass

#----------------------------------------------------------------------------

def calc_inception_scores(run_id, log='inception.txt', num_images=50000, minibatch_size=16, eval_reals=True, reverse_order=False):
    result_subdir = misc.locate_result_subdir(run_id)
    network_pkls = misc.list_network_pkls(result_subdir)
    misc.set_output_log_file(os.path.join(result_subdir, log))

    print('Importing inception score module...')
    import inception_score
    def calc_inception_score(images):
        if images.shape[1] == 1:
            images = images.repeat(3, axis=1)
        images = list(images.transpose(0, 2, 3, 1))
        return inception_score.get_inception_score(images)

    # Load dataset.
    training_set, drange_orig = load_dataset_for_previous_run(result_subdir, shuffle=False)
    reals, labels = training_set.get_random_minibatch(num_images, labels=True)

    # Evaluate reals.
    if eval_reals:
        print('Evaluating inception score for reals...')
        time_begin = time.time()
        mean, std = calc_inception_score(reals)
        print('Done in %s' % misc.format_time(time.time() - time_begin))
        print('%-32s mean %-8.4f std %-8.4f' % ('reals', mean, std))

    # Evaluate each network snapshot.
    network_pkls = list(enumerate(network_pkls))
    if reverse_order:
        network_pkls = network_pkls[::-1]
    for network_idx, network_pkl in network_pkls:
        print('%-32s' % os.path.basename(network_pkl), end=' ')
        net = imgapi_load_net(run_id=result_subdir, snapshot=network_pkl, num_example_latents=num_images, random_seed=network_idx)
        fakes = imgapi_generate_batch(net, net.example_latents, np.random.permutation(labels), minibatch_size=minibatch_size, convert_to_uint8=True)
        mean, std = calc_inception_score(fakes)
        print('mean %-8.4f std %-8.4f' % (mean, std))
    print()
    print('Done.')

#----------------------------------------------------------------------------

def calc_sliced_wasserstein_scores(
    run_id,
    log                 = 'sliced-wasserstein.txt',
    resolution_min      = 16,
    resolution_max      = 1024,
    num_images          = 8192,
    nhoods_per_image    = 64,
    nhood_size          = 7,
    dir_repeats         = 1,
    dirs_per_repeat     = 147,
    minibatch_size      = 16):

    import sliced_wasserstein
    result_subdir = misc.locate_result_subdir(run_id)
    network_pkls = misc.list_network_pkls(result_subdir)
    misc.set_output_log_file(os.path.join(result_subdir, log))

    # Load dataset.
    print('Loading dataset...')
    training_set, drange_orig = load_dataset_for_previous_run(result_subdir)
    assert training_set.shape[1] == 3 # RGB
    assert num_images % minibatch_size == 0

    # Select resolutions.
    resolution_full = training_set.shape[3]
    resolution_min = min(resolution_min, resolution_full)
    resolution_max = min(resolution_max, resolution_full)
    base_lod = int(np.log2(resolution_full)) - int(np.log2(resolution_max))
    resolutions = [2**i for i in xrange(int(np.log2(resolution_max)), int(np.log2(resolution_min)) - 1, -1)]

    # Collect descriptors for reals.
    print('Extracting descriptors for reals...', end=' ')
    time_begin = time.time()
    desc_real = [[] for res in resolutions]
    desc_test = [[] for res in resolutions]
    for minibatch_begin in xrange(0, num_images, minibatch_size):
        minibatch = training_set.get_random_minibatch(minibatch_size, lod=base_lod)
        for lod, level in enumerate(sliced_wasserstein.generate_laplacian_pyramid(minibatch, len(resolutions))):
            desc_real[lod].append(sliced_wasserstein.get_descriptors_for_minibatch(level, nhood_size, nhoods_per_image))
            desc_test[lod].append(sliced_wasserstein.get_descriptors_for_minibatch(level, nhood_size, nhoods_per_image))
    print('done in %s' % misc.format_time(time.time() - time_begin))

    # Evaluate scores for reals.
    print('Evaluating scores for reals...', end=' ')
    time_begin = time.time()
    scores = []
    for lod, res in enumerate(resolutions):
        desc_real[lod] = sliced_wasserstein.finalize_descriptors(desc_real[lod])
        desc_test[lod] = sliced_wasserstein.finalize_descriptors(desc_test[lod])
        scores.append(sliced_wasserstein.sliced_wasserstein(desc_real[lod], desc_test[lod], dir_repeats, dirs_per_repeat))
    del desc_test
    print('done in %s' % misc.format_time(time.time() - time_begin))

    # Print table header.
    print()
    print('%-32s' % 'Case', end=' ')
    for lod, res in enumerate(resolutions):
        print('%-12s' % ('%dx%d' % (res, res)), end=' ')
    print('Average')
    print('%-32s' % '---', end=' ')
    for lod, res in enumerate(resolutions):
        print('%-12s' % '---', end=' ')
    print('---')
    print('%-32s' % 'reals', end=' ')
    for lod, res in enumerate(resolutions):
        print('%-12.6f' % scores[lod], end=' ')
    print('%.6f' % np.mean(scores))

    # Process each network snapshot.
    for network_idx, network_pkl in enumerate(network_pkls):
        print('%-32s' % os.path.basename(network_pkl), end=' ')
        net = imgapi_load_net(run_id=result_subdir, snapshot=network_pkl, num_example_latents=num_images, random_seed=network_idx)

        # Extract descriptors for generated images.
        desc_fake = [[] for res in resolutions]
        for minibatch_begin in xrange(0, num_images, minibatch_size):
            latents = net.example_latents[minibatch_begin : minibatch_begin + minibatch_size]
            labels = net.example_labels[minibatch_begin : minibatch_begin + minibatch_size]
            minibatch = imgapi_generate_batch(net, latents, labels, minibatch_size=minibatch_size, convert_to_uint8=True)
            minibatch = sliced_wasserstein.downscale_minibatch(minibatch, base_lod)
            for lod, level in enumerate(sliced_wasserstein.generate_laplacian_pyramid(minibatch, len(resolutions))):
                desc_fake[lod].append(sliced_wasserstein.get_descriptors_for_minibatch(level, nhood_size, nhoods_per_image))

        # Evaluate scores.
        scores = []
        for lod, res in enumerate(resolutions):
            desc_fake[lod] = sliced_wasserstein.finalize_descriptors(desc_fake[lod])
            scores.append(sliced_wasserstein.sliced_wasserstein(desc_real[lod], desc_fake[lod], dir_repeats, dirs_per_repeat))
        del desc_fake

        # Report results.
        for lod, res in enumerate(resolutions):
            print('%-12.6f' % scores[lod], end=' ')
        print('%.6f' % np.mean(scores))
    print()
    print('Done.')

#----------------------------------------------------------------------------

def calc_mnistrgb_histogram(run_id, num_images=25600, log='histogram.txt', minibatch_size=256, num_evals=10, eval_reals=True, final_only=False):

    # Load the classification network.
    # NOTE: The PKL can be downloaded from https://drive.google.com/open?id=0B4qLcYyJmiz0NHFULTdYc05lX0U
    net = network.load_mnist_classifier(os.path.join(config.data_dir, '../networks/mnist_classifier_weights.pkl'))
    input_var = T.tensor4()
    output_expr = lasagne.layers.get_output(net, inputs=input_var, deterministic=True)
    classify_fn = theano.function([input_var], [output_expr])

    # Process folders
    print('Processing directory %s' % (run_id))
    result_subdir = misc.locate_result_subdir(run_id)

    network_pkls = misc.list_network_pkls(result_subdir)
    misc.set_output_log_file(os.path.join(result_subdir, log))

    if final_only:
        network_pkls = [network_pkls[-1]]

    # Histogram calculation.
    def calc_histogram(images_all):
        scores = []
        divergences = []
        for i in range(num_evals):
            images = images_all[i * num_images : (i + 1) * num_images]
            model = [0.]*1000
            for s in range(0, images.shape[0], minibatch_size):
                img = images[s:s+minibatch_size].reshape((-1, 1, 32, 32))
                res = np.asarray(classify_fn(img)[0])
                res = np.argmax(res, axis=1)
                res = res.reshape((-1, 3)) * np.asarray([[1, 10, 100]])
                res = np.sum(res, axis=1)
                for x in res:
                    model[int(x)] += 1.
            model = np.array([b/25600. for b in model if b>0])  # remove empty buckets, normalize
            data  = np.array([1./1000]*len(model))              # corresponding ideal counts
            scores.append(len(model))
            divergences.append(np.sum(model*np.log(model/data)))    # reverse KL? Metz et al. say KL(model || data)
        scores = np.asarray(scores, dtype=np.float32)
        return np.mean(scores), np.mean(divergences)

    # Load dataset.
    training_set, drange_orig = load_dataset_for_previous_run(result_subdir, shuffle=False)
    reals, labels = training_set.get_random_minibatch(num_images * num_evals, labels=True)

    # Evaluate reals.
    if eval_reals:
        print('Evaluating histogram for reals...')
        time_begin = time.time()
        mean, kld = calc_histogram(reals)
        print('Done in %s' % misc.format_time(time.time() - time_begin))
        print('mean %-8.4f kld %-8.4f' % (mean, kld))

    # Evaluate each network snapshot.
    latents = None
    for network_idx, network_pkl in enumerate(network_pkls):
        print('%-32s' % os.path.basename(network_pkl), end=' ')
        net = imgapi_load_net(run_id=result_subdir, snapshot=network_pkl, num_example_latents=num_images*num_evals)
        fakes = imgapi_generate_batch(net, net.example_latents, labels, minibatch_size=minibatch_size, convert_to_uint8=True)
        mean, kld = calc_histogram(fakes)
        print('mean %-8.4f kld %-8.4f' % (mean, kld))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(config.random_seed)
    func_params = config.train
    func_name = func_params['func']
    del func_params['func']
    globals()[func_name](**func_params)

#----------------------------------------------------------------------------
