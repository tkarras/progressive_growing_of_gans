# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import pickle
import inspect
import numpy as np

import tfutil
import networks

#----------------------------------------------------------------------------
# Custom unpickler that is able to load network pickles produced by
# the old Theano implementation.

class LegacyUnpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == 'network' and name == 'Network':
            return tfutil.Network
        return super().find_class(module, name)

#----------------------------------------------------------------------------
# Import handler for tfutil.Network that silently converts networks produced
# by the old Theano implementation to a suitable format.

theano_gan_remap = {
    'G_paper':          'G_paper',
    'G_progressive_8':  'G_paper',
    'D_paper':          'D_paper',
    'D_progressive_8':  'D_paper'}

def patch_theano_gan(state):
    if 'version' in state or state['build_func_spec']['func'] not in theano_gan_remap:
        return state

    spec = dict(state['build_func_spec'])
    func = spec.pop('func')
    resolution = spec.get('resolution', 32)
    resolution_log2 = int(np.log2(resolution))
    use_wscale = spec.get('use_wscale', True)

    assert spec.pop('label_size',       0)          == 0
    assert spec.pop('use_batchnorm',    False)      == False
    assert spec.pop('tanh_at_end',      None)       is None
    assert spec.pop('mbstat_func',      'Tstdeps')  == 'Tstdeps'
    assert spec.pop('mbstat_avg',       'all')      == 'all'
    assert spec.pop('mbdisc_kernels',   None)       is None
    spec.pop(       'use_gdrop',        True)       # doesn't make a difference
    assert spec.pop('use_layernorm',    False)      == False
    spec[           'fused_scale']                  = False
    spec[           'mbstd_group_size']             = 16

    vars = []
    param_iter = iter(state['param_values'])
    relu = np.sqrt(2); linear = 1.0
    def flatten2(w): return w.reshape(w.shape[0], -1)
    def he_std(gain, w): return gain / np.sqrt(np.prod(w.shape[:-1]))
    def wscale(gain, w): return w * next(param_iter) / he_std(gain, w) if use_wscale else w
    def layer(name, gain, w): return [(name + '/weight', wscale(gain, w)), (name + '/bias', next(param_iter))]
    
    if func.startswith('G'):
        vars += layer('4x4/Dense', relu/4, flatten2(next(param_iter).transpose(1,0,2,3)))
        vars += layer('4x4/Conv', relu, next(param_iter).transpose(2,3,1,0)[::-1,::-1])
        for res in range(3, resolution_log2 + 1):
            vars += layer('%dx%d/Conv0' % (2**res, 2**res), relu, next(param_iter).transpose(2,3,1,0)[::-1,::-1])
            vars += layer('%dx%d/Conv1' % (2**res, 2**res), relu, next(param_iter).transpose(2,3,1,0)[::-1,::-1])
        for lod in range(0, resolution_log2 - 1):
            vars += layer('ToRGB_lod%d' % lod, linear, next(param_iter)[np.newaxis, np.newaxis])

    if func.startswith('D'):
        vars += layer('FromRGB_lod0', relu, next(param_iter)[np.newaxis, np.newaxis])
        for res in range(resolution_log2, 2, -1):
            vars += layer('%dx%d/Conv0' % (2**res, 2**res), relu, next(param_iter).transpose(2,3,1,0)[::-1,::-1])
            vars += layer('%dx%d/Conv1' % (2**res, 2**res), relu, next(param_iter).transpose(2,3,1,0)[::-1,::-1])
            vars += layer('FromRGB_lod%d' % (resolution_log2 - (res - 1)), relu, next(param_iter)[np.newaxis, np.newaxis])
        vars += layer('4x4/Conv', relu, next(param_iter).transpose(2,3,1,0)[::-1,::-1])
        vars += layer('4x4/Dense0', relu, flatten2(next(param_iter)[:,:,::-1,::-1]).transpose())
        vars += layer('4x4/Dense1', linear, next(param_iter))

    vars += [('lod', state['toplevel_params']['cur_lod'])]

    return {
        'version':          2,
        'name':             func,
        'build_module_src': inspect.getsource(networks),
        'build_func_name':  theano_gan_remap[func],
        'static_kwargs':    spec,
        'variables':        vars}

tfutil.network_import_handlers.append(patch_theano_gan)

#----------------------------------------------------------------------------
# Import handler for tfutil.Network that ignores unsupported/deprecated
# networks produced by older versions of the code.

def ignore_unknown_theano_network(state):
    if 'version' in state:
        return state

    print('Ignoring unknown Theano network:', state['build_func_spec']['func'])
    return {
        'version':          2,
        'name':             'Dummy',
        'build_module_src': 'def dummy(input, **kwargs): input.set_shape([None, 1]); return input',
        'build_func_name':  'dummy',
        'static_kwargs':    {},
        'variables':        []}

tfutil.network_import_handlers.append(ignore_unknown_theano_network)

#----------------------------------------------------------------------------
