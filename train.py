# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import time
import numpy as np
import tensorflow as tf

import config
import tfutil
import dataset
import misc

#----------------------------------------------------------------------------
# Choose the size and contents of the image snapshot grids that are exported
# periodically during training.

def setup_snapshot_image_grid(G, training_set,
    size    = '1080p',      # '1080p' = to be viewed on 1080p display, '4k' = to be viewed on 4k display.
    layout  = 'random'):    # 'random' = grid contents are selected randomly, 'row_per_class' = each row corresponds to one class label.

    # Select size.
    gw = 1; gh = 1
    if size == '1080p':
        gw = np.clip(1920 // G.output_shape[3], 3, 32)
        gh = np.clip(1080 // G.output_shape[2], 2, 32)
    if size == '4k':
        gw = np.clip(3840 // G.output_shape[3], 7, 32)
        gh = np.clip(2160 // G.output_shape[2], 4, 32)

    # Fill in reals and labels.
    reals = np.zeros([gw * gh] + training_set.shape, dtype=training_set.dtype)
    labels = np.zeros([gw * gh, training_set.label_size], dtype=training_set.label_dtype)
    for idx in range(gw * gh):
        x = idx % gw; y = idx // gw
        while True:
            real, label = training_set.get_minibatch_np(1)
            if layout == 'row_per_class' and training_set.label_size > 0:
                if label[0, y % training_set.label_size] == 0.0:
                    continue
            reals[idx] = real[0]
            labels[idx] = label[0]
            break

    # Generate latents.
    latents = misc.random_latents(gw * gh, G)
    return (gw, gh), reals, labels, latents

#----------------------------------------------------------------------------
# Adjust minibatch size in case the training happens to run out of memory.

def adjust_minibatch_when_out_of_memory(res, minibatch_size, minibatch_limits):
    limit = (minibatch_size - (minibatch_size - 1) // 16 - 1) // config.num_gpus
    limit = min(limit, minibatch_limits.get(res // 2, limit))
    group = config.D.get('mbstd_group_size', 4)
    if limit > group:
        limit = (limit // group) * group
    print('OUT OF MEMORY -- trying minibatch_limits[%d] = %d' % (res, limit))
    assert limit >= 1
    minibatch_limits = dict(minibatch_limits)
    minibatch_limits[res] = limit
    return minibatch_limits

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('DynamicRange'):
        x = tf.cast(x, tf.float32)
        x = misc.adjust_dynamic_range(x, drange_data, drange_net)
    if mirror_augment:
        with tf.name_scope('MirrorAugment'):
            s = tf.shape(x)
            mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
            mask = tf.tile(mask, [1, s[1], s[2], s[3]])
            x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
    with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
        s = tf.shape(x)
        y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
        y = tf.reduce_mean(y, axis=[3, 5], keep_dims=True)
        y = tf.tile(y, [1, 1, 1, 2, 1, 2])
        y = tf.reshape(y, [-1, s[1], s[2], s[3]])
        x = tfutil.lerp(x, y, lod - tf.floor(lod))
    with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
        s = tf.shape(x)
        factor = tf.cast(2 ** tf.floor(lod), tf.int32)
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x

#----------------------------------------------------------------------------
# Loss function.

def training_loss(
    G, D, D_opt,
    real_images_in, real_labels_in, fake_latents_in, fake_labels_in,
    type            = 'wgan-gp',    # Type of loss to use. Must be 'wgan-gp'.
    wgan_lambda     = 10.0,         # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,        # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,          # Target value for gradient magnitudes.
    cond_type       = 'acgan',      # Type of conditioning to use. Must be 'acgan'.
    cond_weight     = 1.0):         # Weight of the conditioning terms.

    G_loss = []; D_loss = [] # [(term, weight), ...]
    with tf.name_scope('G_fake'):
        fake_images_out = G.get_output_for(fake_latents_in, fake_labels_in)
    with tf.name_scope('D_real'):
        real_scores_out, real_labels_out = D.get_output_for(real_images_in)
        real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    with tf.name_scope('D_fake'):
        fake_scores_out, fake_labels_out = D.get_output_for(fake_images_out)
        fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)

    if type == 'wgan-gp':
        with tf.name_scope('Mix'):
            mixing_factors = tf.random_uniform([tf.shape(real_images_in)[0], 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
            mixed_images_out = tfutil.lerp(tf.cast(real_images_in, fake_images_out.dtype), fake_images_out, mixing_factors)
        with tf.name_scope('D_mixed'):
            mixed_scores_out, mixed_labels_out = D.get_output_for(mixed_images_out)
            mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        with tf.name_scope('GradientPenalty'):
            mixed_loss = tf.reduce_sum(tf.cast(mixed_scores_out, tf.float32))
            mixed_grads = tf.gradients(D_opt.apply_loss_scaling(mixed_loss), mixed_images_out)
            mixed_grads = D_opt.undo_loss_scaling(tf.cast(mixed_grads, tf.float32))
            mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
            mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
            gradient_penalty = tf.square(mixed_norms - wgan_target)
        with tf.name_scope('EpsilonPenalty'):
            epsilon_penalty = tf.square(tf.cast(real_scores_out, tf.float32))
            epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', epsilon_penalty)
        G_loss += [(fake_scores_out, -1.0)]
        D_loss += [(fake_scores_out, 1.0)]
        D_loss += [(real_scores_out, -1.0)]
        D_loss += [(gradient_penalty, wgan_lambda / (wgan_target**2))]
        D_loss += [(epsilon_penalty, wgan_epsilon)]

    if cond_type == 'acgan' and D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits(labels=real_labels_in, logits=real_labels_out)
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits(labels=fake_labels_in, logits=fake_labels_out)
            label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        G_loss += [(label_penalty_fakes, cond_weight)]
        D_loss += [(label_penalty_reals, cond_weight)]
        D_loss += [(label_penalty_fakes, cond_weight)]

    # Note: Cannot use tfutil.autosummary() for anything in this block, because the losses are never actually evaluated.
    with tf.name_scope('Loss'):
        G_loss = tf.add_n([tf.reduce_mean(tf.cast(term, tf.float32)) * weight for term, weight in G_loss])
        D_loss = tf.add_n([tf.reduce_mean(tf.cast(term, tf.float32)) * weight for term, weight in D_loss])
    return G_loss, D_loss

#----------------------------------------------------------------------------
# Select the level-of-detail and learning rates based on training progress.

def training_schedule(
    cur_nimg, resolution_log2,
    lod_initial_resolution  = 4,        # Image resolution used at the beginning.
    lod_training_kimg       = 600,      # Thousands of real images to show before doubling the resolution.
    lod_transition_kimg     = 600,      # Thousands of real images to show when fading in new layers.
    G_learning_rate_max     = 0.001,    # Target learning rate for the generator.
    D_learning_rate_max     = 0.001,    # Target learning rate for the discriminator.
    rampup_kimg             = 40):      # Duration of learning rate ramp-up after doubling the resolution.

    cur_kimg = cur_nimg / 1000.0
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(cur_kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = cur_kimg - phase_idx * phase_dur

    lod = resolution_log2
    lod -= np.floor(np.log2(lod_initial_resolution))
    lod -= phase_idx
    if lod_transition_kimg > 0:
        lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
    lod = max(lod, 0.0)

    rampup = 1.0
    if cur_kimg < rampup_kimg or (phase_kimg > lod_training_kimg and lod > 0.0):
        rampup = cur_kimg if cur_kimg < rampup_kimg else phase_kimg - lod_training_kimg
        rampup = np.clip(rampup / rampup_kimg, 0.0, 1.0)
        rampup = np.exp(-5.0 * np.square(1.0 - rampup))
    G_lrate = G_learning_rate_max * rampup
    D_lrate = D_learning_rate_max * rampup
    return lod, G_lrate, D_lrate

#----------------------------------------------------------------------------
# Main training script.
# To run, comment/uncomment appropriate lines in config.py and launch train.py.

def train_progressive_gan(
    G_smoothing             = 0.999**16,    # Exponential running average of generator weights.
    minibatch_default       = 64,           # Maximum minibatch size, divided evenly among GPUs.
    minibatch_limits        = {},           # Maximum minibatch size per GPU for each resolution.
    minibatch_repeats       = 4,            # Number of minibatches to run before adjusting training parameters.
    total_kimg              = 15000,        # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,        # Enable mirror augment?
    drange_net              = [-1,1],       # Dynamic range used when feeding image data to the networks.
    tick_kimg_default       = 160,          # Default interval of progress snapshots.
    tick_kimg_overrides     = {8:140, 16:120, 32:100, 64:80, 128:60, 256:40, 512:20, 1024:10}, # Interval of progress snapshots for each resolution.
    image_snapshot_ticks    = 1,            # How often to export image snapshots?
    network_snapshot_ticks  = 10,           # How often to export network snapshots?
    resume_run_id           = None,         # Run ID or network pkl to resume training from, None = start from scratch.
    resume_snapshot         = None,         # Snapshot index to resume training from, None = autodetect.
    resume_kimg             = 0.0,          # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0):         # Assumed wallclock time at the beginning. Affects reporting.

    maintenance_start_time = time.time()
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)

    # Construct networks.
    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            G, D, Gs = misc.load_pkl(network_pkl)
        else:
            print('Constructing networks...')
            G = tfutil.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **config.G)
            D = tfutil.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **config.D)
            Gs = G.clone('Gs')
        Gs_update_op = Gs.setup_as_moving_average_of(G, beta=G_smoothing)
    G.print_layers(); G.setup_weight_histograms()
    D.print_layers(); D.setup_weight_histograms()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // config.num_gpus
        reals, labels   = training_set.get_minibatch_tf()
        reals_split     = tf.split(reals, config.num_gpus)
        labels_split    = tf.split(labels, config.num_gpus)
    G_opt = tfutil.Optimizer(name='TrainG', learning_rate=lrate_in, **config.G_opt)
    D_opt = tfutil.Optimizer(name='TrainD', learning_rate=lrate_in, **config.D_opt)
    for gpu in range(config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            Gc = G if gpu == 0 else G.clone(G.name + '_shadow')
            Dc = D if gpu == 0 else D.clone(D.name + '_shadow')
            with tf.name_scope('Inputs'):
                real_images  = process_reals(reals_split[gpu], lod_in, mirror_augment, training_set.dynamic_range, drange_net)
                real_labels  = labels_split[gpu]
                fake_latents = tf.random_normal([minibatch_split] + Gc.input_shapes[0][1:])
                fake_labels  = training_set.get_random_labels_tf(minibatch_split)
                assign_ops   = [tf.assign(Gc.find_var('lod'), lod_in), tf.assign(Dc.find_var('lod'), lod_in)]
            with tf.control_dependencies(assign_ops):
                G_loss, D_loss = training_loss(Gc, Dc, D_opt, real_images, real_labels, fake_latents, fake_labels, **config.loss)
                G_opt.register_gradients(G_loss, Gc.trainables)
                D_opt.register_gradients(D_loss, Dc.trainables)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()
    tfutil.init_uninited_vars()

    print('Setting up result dir...')
    grid_size, grid_reals, grid_labels, grid_latents = setup_snapshot_image_grid(G, training_set, **config.grid)
    result_subdir = misc.create_result_subdir(config.result_dir, config.run_desc)
    misc.save_image_grid(grid_reals, os.path.join(result_subdir, 'reals.png'), drange=training_set.dynamic_range, grid_size=grid_size)
    misc.save_image_grid(Gs.run(grid_latents, grid_labels), os.path.join(result_subdir, 'fakes%06d.png' % 0), drange=drange_net, grid_size=grid_size)
    summary_log = tf.summary.FileWriter(result_subdir, tf.get_default_graph())

    print('Training...')
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    train_start_time = tick_start_time - resume_time
    while cur_nimg < total_kimg * 1000:

        # Choose training parameters.
        cur_lod, G_lrate, D_lrate = training_schedule(cur_nimg, training_set.resolution_log2, **config.schedule)
        cur_res = 2 ** (training_set.resolution_log2 - int(np.floor(cur_lod)))
        minibatch_size = minibatch_limits.get(cur_res, minibatch_default)
        minibatch_size = min(minibatch_size, minibatch_default // config.num_gpus) * config.num_gpus
        tick_duration_kimg = tick_kimg_overrides.get(cur_res, tick_kimg_default)

        # Run training ops.
        try:
            training_set.configure(minibatch_size, cur_lod)
            for repeat in range(minibatch_repeats):
                tfutil.run([D_train_op, Gs_update_op], {lod_in: cur_lod, lrate_in: D_lrate, minibatch_in: minibatch_size})
                tfutil.run([G_train_op], {lod_in: cur_lod, lrate_in: G_lrate, minibatch_in: minibatch_size})
                cur_nimg += minibatch_size
        except tf.errors.ResourceExhaustedError:
            minibatch_limits = adjust_minibatch_when_out_of_memory(cur_res, minibatch_size, minibatch_limits)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + tick_duration_kimg * 1000 or done:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            total_time = cur_time - train_start_time
            maintenance_time = tick_start_time - maintenance_start_time
            maintenance_start_time = cur_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %.1f' % (
                tfutil.autosummary('Progress/tick', cur_tick),
                tfutil.autosummary('Progress/kimg', cur_nimg / 1000.0),
                tfutil.autosummary('Progress/lod', cur_lod),
                tfutil.autosummary('Progress/minibatch', minibatch_size),
                misc.format_time(tfutil.autosummary('Timing/total_sec', total_time)),
                tfutil.autosummary('Timing/sec_per_tick', tick_time),
                tfutil.autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                tfutil.autosummary('Timing/maintenance_sec', maintenance_time)))
            tfutil.autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            tfutil.autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))
            tfutil.save_summaries(summary_log, cur_nimg)

            # Save snapshots.
            if cur_tick % image_snapshot_ticks == 0 or done:
                misc.save_image_grid(Gs.run(grid_latents, grid_labels), os.path.join(result_subdir, 'fakes%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
            if cur_tick % network_snapshot_ticks == 0 or done:
                misc.save_pkl((G, D, Gs), os.path.join(result_subdir, 'network-snapshot-%06d.pkl' % (cur_nimg // 1000)))

            # Record start time of the next tick.
            tick_start_time = time.time()

    # Write final results.
    misc.save_pkl((G, D, Gs), os.path.join(result_subdir, 'network-final.pkl'))
    summary_log.close()
    open(os.path.join(result_subdir, '_training-done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Main entry point.
# Calls the function indicated in config.py.

if __name__ == "__main__":
    misc.init_output_logging()
    np.random.seed(config.random_seed)
    print('Initializing TensorFlow...')
    tfutil.init_tf(config.tf_config)
    kwargs = dict(config.train)
    func = kwargs.pop('func')
    print('Running %s()...' % func)
    tfutil.import_obj(func)(**kwargs)
    print('Exiting...')

#----------------------------------------------------------------------------
