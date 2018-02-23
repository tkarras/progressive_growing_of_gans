# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

#----------------------------------------------------------------------------
# Paths.

data_dir = 'datasets'
result_dir = 'results'

#----------------------------------------------------------------------------
# Baseline training config.
# Comment/uncomment the lines as appropriate and launch train.py.

if 1:
    run_desc    = 'pgan'                                        # Description string included in result subdir name.
    random_seed = 1001                                          # Global random seed for NumPy.
    tf_config   = {'graph_options.place_pruned_graph': True}    # Options for tfutil.init_tf().
    dataset     = dict()                                        # Options for dataset.load_dataset().
    train       = dict(func='train.train_progressive_gan')      # Options for the main training func.
    G           = dict(func='networks.G_paper')                 # Options for the generator network.
    D           = dict(func='networks.D_paper')                 # Options for the discriminator network.
    G_opt       = dict(beta1=0.0, beta2=0.99, epsilon=1e-8)     # Options for tfutil.Optimizer('TrainG').
    D_opt       = dict(beta1=0.0, beta2=0.99, epsilon=1e-8)     # Options for tfutil.Optimizer('TrainD').
    loss        = dict(type='wgan-gp', cond_type='acgan')       # Options for train.training_loss().
    schedule    = dict()                                        # Options for train.training_schedule().
    grid        = dict(size='1080p', layout='random')           # Options for train.setup_snapshot_image_grid().

    # Dataset selection.

    run_desc += '-celebahq';   dataset = dict(tfrecord_dir='celebahq'); train.update(mirror_augment=True)
    #run_desc += '-celeba';     dataset = dict(tfrecord_dir='celeba'); train.update(mirror_augment=True)
    #run_desc += '-cifar10';    dataset = dict(tfrecord_dir='cifar10')
    #run_desc += '-cifar100';   dataset = dict(tfrecord_dir='cifar100')
    #run_desc += '-svhn';       dataset = dict(tfrecord_dir='svhn')
    #run_desc += '-mnist';      dataset = dict(tfrecord_dir='mnist')
    #run_desc += '-mnistrgb';   dataset = dict(tfrecord_dir='mnistrgb')
    #run_desc += '-syn1024rgb'; dataset = dict(class_name='dataset.SyntheticDataset', resolution=1024, num_channels=3)

    #run_desc += '-lsun-airplane';       dataset = dict(tfrecord_dir='lsun-airplane-100k');       train.update(mirror_augment=True)
    #run_desc += '-lsun-bedroom';        dataset = dict(tfrecord_dir='lsun-bedroom-100k');        train.update(mirror_augment=True)
    #run_desc += '-lsun-bicycle';        dataset = dict(tfrecord_dir='lsun-bicycle-100k');        train.update(mirror_augment=True)
    #run_desc += '-lsun-bird';           dataset = dict(tfrecord_dir='lsun-bird-100k');           train.update(mirror_augment=True)
    #run_desc += '-lsun-boat';           dataset = dict(tfrecord_dir='lsun-boat-100k');           train.update(mirror_augment=True)
    #run_desc += '-lsun-bottle';         dataset = dict(tfrecord_dir='lsun-bottle-100k');         train.update(mirror_augment=True)
    #run_desc += '-lsun-bridge';         dataset = dict(tfrecord_dir='lsun-bridge-100k');         train.update(mirror_augment=True)
    #run_desc += '-lsun-bus';            dataset = dict(tfrecord_dir='lsun-bus-100k');            train.update(mirror_augment=True)
    #run_desc += '-lsun-car';            dataset = dict(tfrecord_dir='lsun-car-100k');            train.update(mirror_augment=True)
    #run_desc += '-lsun-cat';            dataset = dict(tfrecord_dir='lsun-cat-100k');            train.update(mirror_augment=True)
    #run_desc += '-lsun-chair';          dataset = dict(tfrecord_dir='lsun-chair-100k');          train.update(mirror_augment=True)
    #run_desc += '-lsun-churchoutdoor';  dataset = dict(tfrecord_dir='lsun-churchoutdoor-100k');  train.update(mirror_augment=True)
    #run_desc += '-lsun-classroom';      dataset = dict(tfrecord_dir='lsun-classroom-100k');      train.update(mirror_augment=True)
    #run_desc += '-lsun-conferenceroom'; dataset = dict(tfrecord_dir='lsun-conferenceroom-100k'); train.update(mirror_augment=True)
    #run_desc += '-lsun-cow';            dataset = dict(tfrecord_dir='lsun-cow-100k');            train.update(mirror_augment=True)
    #run_desc += '-lsun-diningroom';     dataset = dict(tfrecord_dir='lsun-diningroom-100k');     train.update(mirror_augment=True)
    #run_desc += '-lsun-diningtable';    dataset = dict(tfrecord_dir='lsun-diningtable-100k');    train.update(mirror_augment=True)
    #run_desc += '-lsun-dog';            dataset = dict(tfrecord_dir='lsun-dog-100k');            train.update(mirror_augment=True)
    #run_desc += '-lsun-horse';          dataset = dict(tfrecord_dir='lsun-horse-100k');          train.update(mirror_augment=True)
    #run_desc += '-lsun-kitchen';        dataset = dict(tfrecord_dir='lsun-kitchen-100k');        train.update(mirror_augment=True)
    #run_desc += '-lsun-livingroom';     dataset = dict(tfrecord_dir='lsun-livingroom-100k');     train.update(mirror_augment=True)
    #run_desc += '-lsun-motorbike';      dataset = dict(tfrecord_dir='lsun-motorbike-100k');      train.update(mirror_augment=True)
    #run_desc += '-lsun-person';         dataset = dict(tfrecord_dir='lsun-person-100k');         train.update(mirror_augment=True)
    #run_desc += '-lsun-pottedplant';    dataset = dict(tfrecord_dir='lsun-pottedplant-100k');    train.update(mirror_augment=True)
    #run_desc += '-lsun-restaurant';     dataset = dict(tfrecord_dir='lsun-restaurant-100k');     train.update(mirror_augment=True)
    #run_desc += '-lsun-sheep';          dataset = dict(tfrecord_dir='lsun-sheep-100k');          train.update(mirror_augment=True)
    #run_desc += '-lsun-sofa';           dataset = dict(tfrecord_dir='lsun-sofa-100k');           train.update(mirror_augment=True)
    #run_desc += '-lsun-tower';          dataset = dict(tfrecord_dir='lsun-tower-100k');          train.update(mirror_augment=True)
    #run_desc += '-lsun-train';          dataset = dict(tfrecord_dir='lsun-train-100k');          train.update(mirror_augment=True)
    #run_desc += '-lsun-tvmonitor';      dataset = dict(tfrecord_dir='lsun-tvmonitor-100k');      train.update(mirror_augment=True)

    # Conditioning.

    #run_desc += '-cond'; dataset.update(max_label_size='full') # conditional on full label
    #run_desc += '-cond1'; dataset.update(max_label_size=1) # conditional on first component of the label
    #run_desc += '-g4k'; grid.update(size='4k')
    #run_desc += '-grpc'; grid.update(layout='row_per_class')

    # Numerical precision & memory usage.

    #run_desc += '-fp32'; train.update(minibatch_limits={8:208, 16:132, 32:88, 64:48, 128:32, 256:16, 512:8, 1024:4})
    run_desc += '-fp16'; G.update(dtype='float16'); D.update(dtype='float16'); G_opt.update(use_loss_scaling=True); D_opt.update(use_loss_scaling=True); train.update(minibatch_limits={8:240, 16:240, 32:156, 64:100, 128:56, 256:32, 512:16, 1024:8})

    # Individual parameters.

    #run_desc += '-mb16'; train.update(minibatch_default=16, G_smoothing=0.999**1)
    #run_desc += '-mb32'; train.update(minibatch_default=32, G_smoothing=0.999**2)
    run_desc += '-mb64'; train.update(minibatch_default=64, G_smoothing=0.999**4) # default
    #run_desc += '-mb128'; train.update(minibatch_default=128, G_smoothing=0.999**8)
    #run_desc += '-mb256'; train.update(minibatch_default=256, G_smoothing=0.999**16)

    run_desc += '-lod4-600'; schedule.update(lod_initial_resolution=4, lod_training_kimg=600, lod_transition_kimg=600); train.update(total_kimg=15000) # default
    #run_desc += '-lod4-800'; schedule.update(lod_initial_resolution=4, lod_training_kimg=800, lod_transition_kimg=800); train.update(total_kimg=20000)

    #run_desc += '-lr0.0001'; schedule.update(G_learning_rate_max=0.0001, D_learning_rate_max=0.0001)
    #run_desc += '-lr0.0002'; schedule.update(G_learning_rate_max=0.0002, D_learning_rate_max=0.0002)
    #run_desc += '-lr0.0005'; schedule.update(G_learning_rate_max=0.0005, D_learning_rate_max=0.0005)
    #run_desc += '-lr0.001'; schedule.update(G_learning_rate_max=0.001, D_learning_rate_max=0.001) # default
    #run_desc += '-lr0.002'; schedule.update(G_learning_rate_max=0.002, D_learning_rate_max=0.002)
    #run_desc += '-lr0.005'; schedule.update(G_learning_rate_max=0.005, D_learning_rate_max=0.005)
    #run_desc += '-lr0.01'; schedule.update(G_learning_rate_max=0.01, D_learning_rate_max=0.01)

    #run_desc += '-ramp10'; schedule.update(rampup_kimg=10)
    #run_desc += '-ramp40'; schedule.update(rampup_kimg=40) # default
    #run_desc += '-ramp100'; schedule.update(rampup_kimg=100)
    #run_desc += '-ramp200'; schedule.update(rampup_kimg=200)

    #run_desc += '-condw0.01'; loss.update(cond_weight=0.01)
    #run_desc += '-condw0.1'; loss.update(cond_weight=0.1)
    #run_desc += '-condw1'; loss.update(cond_weight=1.0) # default
    #run_desc += '-condw10'; loss.update(cond_weight=10.0)
    #run_desc += '-condw100'; loss.update(cond_weight=100.0)

    # Disable individual features.

    #run_desc += '-noprogression'; schedule.update(lod_initial_resolution=1024, lod_training_kimg=0, lod_transition_kimg=0); train.update(total_kimg=10000)
    #run_desc += '-nombstd'; D.update(mbstd_group_size=0)
    #run_desc += '-nopixelnorm'; G.update(use_pixelnorm=False)
    #run_desc += '-nowscale-lr0.0001'; G.update(use_wscale=False); D.update(use_wscale=False); schedule.update(G_learning_rate_max=0.0001, D_learning_rate_max=0.0001)
    #run_desc += '-noleakyrelu'; G.update(use_leakyrelu=False)
    #run_desc += '-nosmoothing'; train.update(G_smoothing=0.0)

    # Special modes.

    #run_desc += '-resume'; train.update(resume_run_id=100); schedule.update(lod_initial_resolution=1024)
    #run_desc += '-SPEEDTEST'; schedule.update(lod_initial_resolution=4, lod_training_kimg=3, lod_transition_kimg=3); train.update(total_kimg=(8*2+1)*3, tick_kimg_default=1, tick_kimg_overrides={}, image_snapshot_ticks=1000, network_snapshot_ticks=1000)
    #run_desc += '-SPEEDTEST0'; schedule.update(lod_initial_resolution=1024); train.update(total_kimg=10, tick_kimg_default=1, tick_kimg_overrides={}, image_snapshot_ticks=1000, network_snapshot_ticks=1000)
    #run_desc += '-VERBOSE'; train.update(tick_kimg_default=1, tick_kimg_overrides={}, image_snapshot_ticks=1, network_snapshot_ticks=100)

#----------------------------------------------------------------------------
# Utility scripts.
# To run, uncomment the appropriate line and launch train.py.

#train = dict(func='scripts.generate_fake_images', run_id=100, num_pngs=1000); run_desc = 'fake-images-' + str(train['run_id'])
#train = dict(func='scripts.generate_fake_images', run_id=100, grid_size=[15,8], num_pngs=10, image_shrink=4); run_desc = 'fake-grids-' + str(train['run_id'])
#train = dict(func='scripts.generate_interpolation_video', run_id=100, grid_size=[1,1], duration_sec=60.0, smoothing_sec=1.0); run_desc = 'interpolation-video-' + str(train['run_id'])
#train = dict(func='scripts.generate_training_video', run_id=100, duration_sec=20.0); run_desc = 'training-video-' + str(train['run_id'])

#train = dict(func='scripts.evaluate_metrics', run_id=100, log='metric-swd-16k.txt', metrics=['swd'], num_images=16384, real_passes=2); run_desc = train['log'].split('.')[0] + '-' + str(train['run_id'])
#train = dict(func='scripts.evaluate_metrics', run_id=100, log='metric-fid-50k.txt', metrics=['fid'], num_images=50000, real_passes=2); run_desc = train['log'].split('.')[0] + '-' + str(train['run_id'])
#train = dict(func='scripts.evaluate_metrics', run_id=100, log='metric-is-50k.txt', metrics=['is'], num_images=50000, real_passes=1); run_desc = train['log'].split('.')[0] + '-' + str(train['run_id'])
#train = dict(func='scripts.evaluate_metrics', run_id=100, log='metric-msssim-20k.txt', metrics=['msssim'], num_images=20000, real_passes=1); run_desc = train['log'].split('.')[0] + '-' + str(train['run_id'])

#----------------------------------------------------------------------------
# Number of GPUs to use.

run_desc += '-1gpu'; num_gpus = 1
#run_desc += '-2gpus'; num_gpus = 2
#run_desc += '-4gpus'; num_gpus = 4
#run_desc += '-8gpus'; num_gpus = 8

#----------------------------------------------------------------------------
