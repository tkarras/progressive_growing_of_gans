# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import sys
import imp
import inspect
import copy
import collections
import numpy as np
import theano
from theano import tensor as T
import lasagne
import cPickle

# NOTE: Do not reference config.py here!
# Instead, specify all network parameters as build function arguments.

#----------------------------------------------------------------------------
# Convenience.

from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer, NINLayer
from lasagne.layers import Upscale2DLayer, Pool2DLayer, GlobalPoolLayer, MaxPool2DLayer
from lasagne.layers import ReshapeLayer, ElemwiseSumLayer, ConcatLayer, FlattenLayer
from lasagne.layers import NonlinearityLayer, ScaleLayer

linear,  ilinear  = lasagne.nonlinearities.linear,            lasagne.init.HeNormal(1.0)
relu,    irelu    = lasagne.nonlinearities.rectify,           lasagne.init.HeNormal('relu')
lrelu,   ilrelu   = lasagne.nonlinearities.LeakyRectify(0.2), lasagne.init.HeNormal('relu')
vlrelu            = lasagne.nonlinearities.LeakyRectify(0.3)
elu,     ielu     = lasagne.nonlinearities.elu,               lasagne.init.HeNormal('relu')
tanh,    itanh    = lasagne.nonlinearities.tanh,              lasagne.init.HeNormal(1.0)
sigmoid, isigmoid = lasagne.nonlinearities.sigmoid,           lasagne.init.HeNormal(1.0)
clip,    iclip    = lambda x: T.clip(x, 0, 1),                lasagne.init.HeNormal('relu')

def Tsum    (*args, **kwargs): return T.sum (*args, dtype=theano.config.floatX, acc_dtype=theano.config.floatX, **kwargs)
def Tmean   (*args, **kwargs): return T.mean(*args, dtype=theano.config.floatX, acc_dtype=theano.config.floatX, **kwargs)
def Tstd    (*args, **kwargs): return T.std (*args, **kwargs)
def Tstdeps (val, **kwargs):   return T.sqrt(Tmean(T.square(val - Tmean(val, **kwargs)), **kwargs) + 1.0e-8)
def Downscale2DLayer(incoming, scale_factor, **kwargs): return Pool2DLayer(incoming, pool_size=scale_factor, mode='average_exc_pad', **kwargs)

#----------------------------------------------------------------------------
# Wrapper class for Lasagne networks for robust pickling.

class Network(object):
    def __init__(self, **build_func_spec):
        self.build_func_spec    = build_func_spec       # dict(func='func_name', **kwargs)
        self.build_module_src   = inspect.getsource(sys.modules[__name__]) # For pickle import.
        self.input_layers       = [] # One or more.
        self.output_layers      = [] # One or more.
        self.input_shapes       = [] # Including minibatch dimension.
        self.output_shapes      = [] # Including minibatch dimension.
        self.input_shape        = () # For first input layer.
        self.output_shape       = () # For first output layer.
        #self.arbitrary_field   = ...# Arbitrary fields returned by the build func.
        self.__dict__.update(self._call_build_func(globals()))
        self._call_build_func_from_src() # Make sure that pickle import will work.

    def eval(self, *inputs, **kwargs): # eval(input) => output --OR-- eval(primary_input, secondary_input, ...) => primary_output, secondary_output, ...
        ignore_unused_inputs = kwargs.pop('ignore_unused_inputs', False)
        expect_num_outputs = kwargs.pop('expect_num_outputs', None)
        assert len(inputs) >= len(self.input_layers)
        assert len(inputs) == len(self.input_layers) or ignore_unused_inputs
        input_dict = dict(zip(self.input_layers, inputs[:len(self.input_layers)]))
        outputs = lasagne.layers.get_output(self.output_layers, input_dict, **kwargs)
        if expect_num_outputs is not None:
            outputs += [None] * max(expect_num_outputs - len(outputs), 0)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def eval_d(self, *inputs, **kwargs):
        return self.eval(*inputs, deterministic=True, **kwargs)

    def eval_nd(self, *inputs, **kwargs):
        return self.eval(*inputs, deterministic=False, **kwargs)

    def eval_multi(self, *inputs, **kwargs): # eval(input_batch1, input_batch2, ...) => output_batch1, output_batch2, ... --OR-- eval([list], [list]) => [list], [list]
        input_lists = [input if isinstance(input, list) or isinstance(input, tuple) else [input] for input in inputs]
        combo_inputs = [T.concatenate(spliced_input, axis=0) for spliced_input in zip(*input_lists)]
        combo_outputs = self.eval(*combo_inputs, **kwargs)
        combo_outputs = combo_outputs if isinstance(combo_outputs, tuple) else [combo_outputs]
        output_ranges = [sum(input_lists[j][0].shape[0] for j in xrange(i)) for i in xrange(len(input_lists))]
        output_ranges = [(begin, begin + input_list[0].shape[0]) for input_list, begin in zip(input_lists, output_ranges)]
        spliced_outputs = [[combo_output[begin : end] for begin, end in output_ranges] for combo_output in combo_outputs]
        output_lists = [outputs[0] if len(outputs) == 1 else outputs for outputs in zip(*spliced_outputs)]
        return output_lists[0] if len(output_lists) == 1 else tuple(output_lists)

    def find_layer(self, name):
        for layer in lasagne.layers.get_all_layers(self.output_layers):
            if layer.name == name:
                return layer
        return None

    def trainable_params(self):
        return lasagne.layers.get_all_params(self.output_layers, trainable=True)

    def toplevel_params(self): # returns dict(name=shared)
        return {name: value for name, value in self.__dict__.iteritems() if isinstance(value, theano.compile.SharedVariable)}

    def get_toplevel_param_values(self): # returns dict(name=value)
        return {name: shared.get_value() for name, shared in self.toplevel_params().iteritems()}

    def set_toplevel_param_values(self, value_dict): # accepts dict(name=value)
        for name, shared in self.toplevel_params().iteritems():
            if name in value_dict:
                shared.set_value(value_dict[name])

    def create_temporally_smoothed_version(self, beta=0.99, explicit_updates=True):
        # Create shallow copy of the network.
        net = Network.__new__(Network)
        net.__dict__.update(self.__dict__)
        layer_map = {layer: copy.copy(layer) for layer in lasagne.layers.get_all_layers(net.output_layers)}
        net.input_layers = [layer_map[layer] for layer in net.input_layers]
        net.output_layers = [layer_map[layer] for layer in net.output_layers]
        for layer in layer_map.itervalues():
            if hasattr(layer, 'input_layer'): layer.input_layer = layer_map[layer.input_layer]
            if hasattr(layer, 'input_layers'): layer.input_layers = [layer_map[input] for input in layer.input_layers]

        # Override trainable parameters with their smoothed versions.
        if explicit_updates: net.updates = []
        for layer in layer_map.itervalues():
            orig_params = layer.params
            param_map = dict()
            for name, orig in layer.__dict__.items():
                try:
                    if orig in orig_params and 'trainable' in orig_params[orig] and beta > 0.0:
                        smoothed = theano.shared(orig.get_value())
                        param_map[orig] = smoothed
                        updated = beta * smoothed + (1.0 - beta) * orig
                        if explicit_updates: # explicit_updates=True: You need to explicitly include net.updates in a Theano function to update the weights.
                            layer.__dict__[name] = smoothed
                            net.updates.append((smoothed, updated))
                        else: # explicit_updates=False: Weights are updated automatically every time the net is evaluated.
                            layer.__dict__[name + '_param'] = orig # for print_network_topology_info()
                            layer.__dict__[name] = updated
                            smoothed.default_update = updated
                except TypeError: # if orig is not hashable
                    pass
            layer.params = collections.OrderedDict()
            for param, tags in orig_params.iteritems():
                layer.params[param_map.get(param, param)] = copy.copy(tags)
        return net

    def _call_build_func(self, module_globals):
        func_params = dict(self.build_func_spec)
        func_name = func_params['func']
        del func_params['func']
        if 'subfunc' in func_params:
            func_params['subfunc'] = module_globals[func_params['subfunc']]     # str --> function
        func_result = module_globals[func_name](**func_params)

        # func_result can be one of the following:
        #   output_layer
        #   [first_output_layer, second_output_layer, ...]
        #   dict(output_layers=<one-or-more>)
        #   dict(input_layers=<one-or-more>, output_layers=<one-or-more>)
        #   dict(input_layers=<one-or-more>, output_layers=<one-or-more>, arbitray_field=arbitrary_value, ...)

        # Convert output layer list to canonical form.
        r = dict(func_result) if isinstance(func_result, dict) else dict(output_layers=func_result)
        assert 'output_layers' in r
        if isinstance(r['output_layers'], lasagne.layers.Layer):
            r['output_layers'] = [r['output_layers']]

        # Convert input layer list to canonical form.
        if 'input_layers' not in r:
            r['input_layers'] = [l for l in lasagne.layers.get_all_layers(r['output_layers']) if isinstance(l, InputLayer)]
        elif isinstance(r['input_layers'], lasagne.layers.Layer):
            r['input_layers'] = [r['input_layers']]

        # Check that input/output layers are specified correctly.
        assert isinstance(r['input_layers'], list) and len(r['input_layers']) >= 1
        assert isinstance(r['output_layers'], list) and len(r['output_layers']) >= 1
        assert all(isinstance(layer, InputLayer) for layer in r['input_layers'])

        # Fill in input/output shapes.
        r['input_shapes'] = lasagne.layers.get_output_shape(r['input_layers'])
        r['output_shapes'] = lasagne.layers.get_output_shape(r['output_layers'])
        r['input_shape'] = r['input_shapes'][0]
        r['output_shape'] = r['output_shapes'][0]
        return r

    def _call_build_func_from_src(self):
        tmp_module = imp.new_module('network_tmp_module')
        exec self.build_module_src in tmp_module.__dict__
        globals()['tmp_modules'] = globals().get('tmp_modules', []) + [tmp_module] # Work around issues with GC.
        return self._call_build_func(tmp_module.__dict__)

    def __getstate__(self): # Pickle export.
        return {
            'build_func_spec':  self.build_func_spec,
            'build_module_src': self.build_module_src,
            'param_values':     lasagne.layers.get_all_param_values(self.output_layers),
            'toplevel_params':  self.get_toplevel_param_values()}

    def __setstate__(self, state): # Pickle import.
        self.build_func_spec    = state['build_func_spec']
        self.build_module_src   = state['build_module_src']
        self.__dict__.update(self._call_build_func_from_src())
        lasagne.layers.set_all_param_values(self.output_layers, state['param_values'])
        self.set_toplevel_param_values(state.get('toplevel_params', dict()))

#----------------------------------------------------------------------------
# Mark all parameters in the last layer as non-trainable.

def non_trainable(net):
    for tags in net.params.itervalues():
        tags -= {'trainable', 'regularizable'}
    return net

#----------------------------------------------------------------------------
# Resize activation tensor 'v' of shape 'si' to match shape 'so'.

def resize_activations(v, si, so):
    assert len(si) == len(so) and si[0] == so[0]

    # Decrease feature maps.
    if si[1] > so[1]:
        v = v[:, :so[1]]

    # Shrink spatial axes.
    if len(si) == 4 and (si[2] > so[2] or si[3] > so[3]):
        assert si[2] % so[2] == 0 and si[3] % so[3] == 0
        ws = (si[2] / so[2], si[3] / so[3])
        v = T.signal.pool.pool_2d(v, ws=ws, stride=ws, ignore_border=True, pad=(0,0), mode='average_exc_pad')

    # Extend spatial axes.
    for i in xrange(2, len(si)):
        if si[i] < so[i]:
            assert so[i] % si[i] == 0
            v = T.extra_ops.repeat(v, so[i] / si[i], i)

    # Increase feature maps.
    if si[1] < so[1]:
        z = T.zeros((v.shape[0], so[1] - si[1]) + so[2:], dtype=v.dtype)
        v = T.concatenate([v, z], axis=1)
    return v

#----------------------------------------------------------------------------
# Resolution selector for fading in new layers during progressive growing.

class LODSelectLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, cur_lod, first_incoming_lod=0, ref_idx=0, **kwargs):
        super(LODSelectLayer, self).__init__(incomings, **kwargs)
        self.cur_lod = cur_lod
        self.first_incoming_lod = first_incoming_lod
        self.ref_idx = ref_idx

    def get_output_shape_for(self, input_shapes):
        return input_shapes[self.ref_idx]

    def get_output_for(self, inputs, min_lod=None, max_lod=None, **kwargs):
        v = [resize_activations(input, shape, self.input_shapes[self.ref_idx]) for input, shape in zip(inputs, self.input_shapes)]
        lo = np.clip(int(np.floor(min_lod - self.first_incoming_lod)), 0, len(v)-1) if min_lod is not None else 0
        hi = np.clip(int(np.ceil(max_lod - self.first_incoming_lod)), lo, len(v)-1) if max_lod is not None else len(v)-1
        t = self.cur_lod - self.first_incoming_lod
        r = v[hi]
        for i in xrange(hi-1, lo-1, -1): # i = hi-1, hi-2, ..., lo
            r = theano.ifelse.ifelse(T.lt(t, i+1), v[i] * ((i+1)-t) + v[i+1] * (t-i), r)
        if lo < hi:
            r = theano.ifelse.ifelse(T.le(t, lo), v[lo], r)
        return r

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

class PixelNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(PixelNormLayer, self).__init__(incoming, **kwargs)
    def get_output_for(self, v, **kwargs):
        return v / T.sqrt(Tmean(v**2, axis=1, keepdims=True) + 1.0e-8)

#----------------------------------------------------------------------------
# Applies equalized learning rate to the preceding layer.

class WScaleLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(WScaleLayer, self).__init__(incoming, **kwargs)
        W = incoming.W.get_value()
        scale = np.sqrt(np.mean(W ** 2))
        incoming.W.set_value(W / scale)
        self.scale = self.add_param(scale, (), name='scale', trainable=False)
        self.b = None
        if hasattr(incoming, 'b') and incoming.b is not None:
            b = incoming.b.get_value()
            self.b = self.add_param(b, b.shape, name='b', regularizable=False)
            del incoming.params[incoming.b]
            incoming.b = None
        self.nonlinearity = lasagne.nonlinearities.linear
        if hasattr(incoming, 'nonlinearity') and incoming.nonlinearity is not None:
            self.nonlinearity = incoming.nonlinearity
            incoming.nonlinearity = lasagne.nonlinearities.linear

    def get_output_for(self, v, **kwargs):
        v = v * self.scale
        if self.b is not None:
            pattern = ['x', 0] + ['x'] * (v.ndim - 2)
            v = v + self.b.dimshuffle(*pattern)
        return self.nonlinearity(v)

#----------------------------------------------------------------------------
# Minibatch stat concatenation layer. 
# - func is the function to use for the activations across minibatch
# - averaging tells how much averaging to use ('all', 'spatial', 'none')

class MinibatchStatConcatLayer(lasagne.layers.Layer):
    def __init__(self, incoming, func, averaging, **kwargs):
        super(MinibatchStatConcatLayer, self).__init__(incoming, **kwargs)
        self.func = func
        self.averaging = averaging

    def get_output_shape_for(self, input_shape):
        s = list(input_shape)
        if self.averaging == 'all': s[1] += 1
        elif self.averaging == 'flat': s[1] += 1
        elif self.averaging.startswith('group'): s[1] += int(self.averaging[len('group'):])
        else: s[1] *= 2
        return tuple(s)

    def get_output_for(self, input, **kwargs):
        s = list(input.shape)
        vals = self.func(input,axis=0,keepdims=True)                # per activation, over minibatch dim
        if self.averaging == 'all':                                 # average everything --> 1 value per minibatch
            vals = Tmean(vals,keepdims=True)
            reps = s; reps[1]=1
            vals = T.tile(vals,reps)
        elif self.averaging == 'spatial':                           # average spatial locations
            if len(s) == 4:
                vals = Tmean(vals,axis=(2,3),keepdims=True)
            reps = s; reps[1]=1
            vals = T.tile(vals,reps)
        elif self.averaging == 'none':                              # no averaging, pass on all information
            vals = T.repeat(vals,repeats=s[0],axis=0)
        elif self.averaging == 'gpool':                             # EXPERIMENTAL: compute variance (func) over minibatch AND spatial locations.
            if len(s) == 4:
                vals = self.func(input,axis=(0,2,3),keepdims=True)
            reps = s; reps[1]=1
            vals = T.tile(vals,reps)
        elif self.averaging == 'flat':
            vals = self.func(input,keepdims=True)                   # variance of ALL activations --> 1 value per minibatch
            reps = s; reps[1]=1
            vals = T.tile(vals,reps)
        elif self.averaging.startswith('group'):                    # average everything over n groups of feature maps --> n values per minibatch
            n = int(self.averaging[len('group'):])
            vals = vals.reshape((1, n, s[1]/n, s[2], s[3]))
            vals = Tmean(vals, axis=(2,3,4), keepdims=True)
            vals = vals.reshape((1, n, 1, 1))
            reps = s; reps[1] = 1
            vals = T.tile(vals, reps)
        else:
            raise ValueError('Invalid averaging mode', self.averaging)
        return T.concatenate([input, vals], axis=1)

#----------------------------------------------------------------------------
# Generalized dropout layer. Supports arbitrary subsets of axes and different
# modes. Mainly used to inject multiplicative Gaussian noise in the network.

class GDropLayer(lasagne.layers.Layer):
    def __init__(self, incoming, mode='mul', strength=0.4, axes=(0,1), normalize=False, **kwargs):
        super(GDropLayer, self).__init__(incoming, **kwargs)
        assert mode in ('drop', 'mul', 'prop')
        self.random     = theano.sandbox.rng_mrg.MRG_RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.mode       = mode
        self.strength   = strength
        self.axes       = [axes] if isinstance(axes, int) else list(axes)
        self.normalize  = normalize # If true, retain overall signal variance.
        self.gain       = None      # For experimentation.

    def get_output_for(self, input, deterministic=False, **kwargs):
        if self.gain is not None:
            input = input * self.gain
        if deterministic or not self.strength:
            return input

        in_shape  = self.input_shape
        in_axes   = range(len(in_shape))
        in_shape  = [in_shape[axis] if in_shape[axis] is not None else input.shape[axis] for axis in in_axes] # None => Theano expr
        rnd_shape = [in_shape[axis] for axis in self.axes]
        broadcast = [self.axes.index(axis) if axis in self.axes else 'x' for axis in in_axes]
        one       = T.constant(1)

        if self.mode == 'drop':
            p = one - self.strength
            rnd = self.random.binomial(tuple(rnd_shape), p=p, dtype=input.dtype) / p

        elif self.mode == 'mul':
            rnd = (one + self.strength) ** self.random.normal(tuple(rnd_shape), dtype=input.dtype)

        elif self.mode == 'prop':
            coef = self.strength * T.constant(np.sqrt(np.float32(self.input_shape[1])))
            rnd = self.random.normal(tuple(rnd_shape), dtype=input.dtype) * coef + one

        else:
            raise ValueError('Invalid GDropLayer mode', self.mode)

        if self.normalize:
            rnd = rnd / T.sqrt(Tmean(rnd ** 2, axis=1, keepdims=True))
        return input * rnd.dimshuffle(broadcast)

#----------------------------------------------------------------------------
# Layer normalization. Custom reimplementation based on the paper:
# https://arxiv.org/abs/1607.06450

class LayerNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, epsilon=1.0e-4, **kwargs):
        super(LayerNormLayer, self).__init__(incoming, **kwargs)
        self.epsilon = epsilon
        self.gain = self.add_param(np.float32(1.0), (), name='gain', trainable=True)
        self.b = None
        if hasattr(incoming, 'b') and incoming.b is not None: # steal bias
            b = incoming.b.get_value()
            self.b = self.add_param(b, b.shape, name='b', regularizable=False)
            del incoming.params[incoming.b]
            incoming.b = None
        self.nonlinearity = lasagne.nonlinearities.linear
        if hasattr(incoming, 'nonlinearity') and incoming.nonlinearity is not None: # steal nonlinearity
            self.nonlinearity = incoming.nonlinearity
            incoming.nonlinearity = lasagne.nonlinearities.linear

    def get_output_for(self, v, **kwargs):
        avg_axes = range(1, len(self.input_shape))
        v = v - Tmean(v, axis=avg_axes, keepdims=True) # subtract mean
        v = v * T.inv(T.sqrt(Tmean(T.square(v), axis=avg_axes, keepdims=True) + self.epsilon)) # divide by stdev
        v = v * self.gain # multiply by gain
        if self.b is not None:
            pattern = ['x', 0] + ['x'] * (v.ndim - 2)
            v = v + self.b.dimshuffle(*pattern) # apply bias
        return self.nonlinearity(v) # apply nonlinearity

#----------------------------------------------------------------------------
# Generator network template used in the paper.

def G_paper(
    num_channels        = 1,        # Overridden based on dataset.
    resolution          = 32,       # Overridden based on dataset.
    label_size          = 0,        # Overridden based on dataset.
    fmap_base           = 4096,
    fmap_decay          = 1.0,
    fmap_max            = 256,
    latent_size         = None,
    normalize_latents   = True,
    use_wscale          = True,
    use_pixelnorm       = True,
    use_leakyrelu       = True,
    use_batchnorm       = False,
    tanh_at_end         = None,
    **kwargs):

    R = int(np.log2(resolution))
    assert resolution == 2**R and resolution >= 4
    cur_lod = theano.shared(np.float32(0.0))
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(layer): return PixelNormLayer(layer, name=layer.name+'pn') if use_pixelnorm else layer
    def BN(layer): return lasagne.layers.batch_norm(layer) if use_batchnorm else layer
    def WS(layer): return WScaleLayer(layer, name=layer.name+'S') if use_wscale else layer
    if latent_size is None: latent_size = nf(0)
    (act, iact) = (lrelu, ilrelu) if use_leakyrelu else (relu, irelu)

    input_layers = [InputLayer(name='Glatents', shape=[None, latent_size])]
    net = input_layers[-1]
    if normalize_latents:
        net = PixelNormLayer(net, name='Glnorm')
    if label_size:
        input_layers += [InputLayer(name='Glabels', shape=[None, label_size])]
        net = ConcatLayer(name='Gina', incomings=[net, input_layers[-1]])

    net = ReshapeLayer(name='Ginb', incoming=net, shape=[[0], [1], 1, 1])
    net = PN(BN(WS(Conv2DLayer(net, name='G1a', num_filters=nf(1), filter_size=4, pad='full', nonlinearity=act, W=iact))))
    net = PN(BN(WS(Conv2DLayer(net, name='G1b', num_filters=nf(1), filter_size=3, pad=1,      nonlinearity=act, W=iact))))
    lods  = [net]

    for I in xrange(2, R): # I = 2, 3, ..., R-1
        net = Upscale2DLayer(net, name='G%dup' % I, scale_factor=2)
        net = PN(BN(WS(Conv2DLayer(net, name='G%da'  % I, num_filters=nf(I), filter_size=3, pad=1, nonlinearity=act, W=iact))))
        net = PN(BN(WS(Conv2DLayer(net, name='G%db'  % I, num_filters=nf(I), filter_size=3, pad=1, nonlinearity=act, W=iact))))
        lods += [net]

    lods = [WS(NINLayer(l, name='Glod%d' % i, num_units=num_channels, nonlinearity=linear, W=ilinear)) for i, l in enumerate(reversed(lods))]
    output_layer = LODSelectLayer(name='Glod', incomings=lods, cur_lod=cur_lod, first_incoming_lod=0)
    if tanh_at_end is not None:
        output_layer = NonlinearityLayer(output_layer, name='Gtanh', nonlinearity=tanh)
        if tanh_at_end != 1.0:
            output_layer = non_trainable(ScaleLayer(output_layer, name='Gtanhs', scales=lasagne.init.Constant(tanh_at_end)))
    return dict(input_layers=input_layers, output_layers=[output_layer], cur_lod=cur_lod)

#----------------------------------------------------------------------------
# Discriminator network template used in the paper.

def D_paper(
    num_channels    = 1,        # Overridden based on dataset.
    resolution      = 32,       # Overridden based on dataset.
    label_size      = 0,        # Overridden based on dataset.
    fmap_base       = 4096,
    fmap_decay      = 1.0,
    fmap_max        = 256,
    mbstat_func     = 'Tstdeps',
    mbstat_avg      = 'all',
    mbdisc_kernels  = None,
    use_wscale      = True,
    use_gdrop       = True,
    use_layernorm   = False,
    **kwargs):

    R = int(np.log2(resolution))
    assert resolution == 2**R and resolution >= 4
    cur_lod = theano.shared(np.float32(0.0))
    gdrop_strength = theano.shared(np.float32(0.0))
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def GD(layer): return GDropLayer(layer, name=layer.name+'gd', mode='prop', strength=gdrop_strength) if use_gdrop else layer
    def LN(layer): return LayerNormLayer(layer, name=layer.name+'ln') if use_layernorm else layer
    def WS(layer): return WScaleLayer(layer, name=layer.name+'ws') if use_wscale else layer

    input_layer = InputLayer(name='Dimages', shape=[None, num_channels, 2**R, 2**R])
    net = WS(NINLayer(input_layer, name='D%dx' % (R-1), num_units=nf(R-1), nonlinearity=lrelu, W=ilrelu))

    for I in xrange(R-1, 1, -1): # I = R-1, R-2, ..., 2
        net = LN(WS(Conv2DLayer     (GD(net),     name='D%db'   % I, num_filters=nf(I),   filter_size=3, pad=1, nonlinearity=lrelu, W=ilrelu)))
        net = LN(WS(Conv2DLayer     (GD(net),     name='D%da'   % I, num_filters=nf(I-1), filter_size=3, pad=1, nonlinearity=lrelu, W=ilrelu)))
        net =       Downscale2DLayer(net,         name='D%ddn'  % I, scale_factor=2)
        lod =       Downscale2DLayer(input_layer, name='D%dxs'  % (I-1), scale_factor=2**(R-I))
        lod =    WS(NINLayer        (lod,         name='D%dx'   % (I-1), num_units=nf(I-1), nonlinearity=lrelu, W=ilrelu))
        net =       LODSelectLayer  (             name='D%dlod' % (I-1), incomings=[net, lod], cur_lod=cur_lod, first_incoming_lod=R-I-1)

    if mbstat_avg is not None:
        net = MinibatchStatConcatLayer(net, name='Dstat', func=globals()[mbstat_func], averaging=mbstat_avg)

    net = LN(WS(Conv2DLayer(GD(net), name='D1b', num_filters=nf(1), filter_size=3, pad=1, nonlinearity=lrelu, W=ilrelu)))
    net = LN(WS(Conv2DLayer(GD(net), name='D1a', num_filters=nf(0), filter_size=4, pad=0, nonlinearity=lrelu, W=ilrelu)))

    if mbdisc_kernels:
        import minibatch_discrimination
        net = minibatch_discrimination.MinibatchLayer(net, name='Dmd', num_kernels=mbdisc_kernels)

    output_layers = [WS(DenseLayer(net, name='Dscores', num_units=1, nonlinearity=linear, W=ilinear))]
    if label_size:
        output_layers += [WS(DenseLayer(net, name='Dlabels', num_units=label_size, nonlinearity=linear, W=ilinear))]
    return dict(input_layers=[input_layer], output_layers=output_layers, cur_lod=cur_lod, gdrop_strength=gdrop_strength)

#----------------------------------------------------------------------------
# Cripped generator for MNIST mode recovery experiment.

def G_mnist_mode_recovery(
    num_channels        = 1,
    resolution          = 32,
    fmap_base           = 64,
    fmap_decay          = 1.0,
    fmap_max            = 256,
    latent_size         = None,
    label_size          = 10,
    normalize_latents   = True,
    use_wscale          = False,
    use_pixelnorm       = False,
    use_batchnorm       = True,
    tanh_at_end         = True,
    progressive         = False,
    **kwargs):

    R = int(np.log2(resolution))
    assert resolution == 2**R and resolution >= 4
    cur_lod = theano.shared(np.float32(0.0))
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(layer): return PixelNormLayer(layer, name=layer.name+'pn') if use_pixelnorm else layer
    def BN(layer): return lasagne.layers.batch_norm(layer) if use_batchnorm else layer
    def WS(layer): return WScaleLayer(layer, name=layer.name+'S') if use_wscale else layer
    if latent_size is None: latent_size = nf(0)

    input_layers = [InputLayer(name='Glatents', shape=[None, latent_size])]
    net = input_layers[-1]
    if normalize_latents:
        net = PixelNormLayer(net, name='Glnorm')
    if label_size:
        input_layers += [InputLayer(name='Glabels', shape=[None, label_size])]
        net = ConcatLayer (name='Gina', incomings=[net, input_layers[-1]])

    net = ReshapeLayer(name='Ginb', incoming=net, shape=[[0], [1], 1, 1])
    net = PN(BN(WS(Conv2DLayer(net, name='G1a', num_filters=64, filter_size=4, pad='full', nonlinearity=vlrelu, W=irelu))))

    lods  = [net]
    for I in xrange(2, R): # I = 2, 3, ..., R-1
        net = Upscale2DLayer(net, name='G%dup' % I, scale_factor=2)
        net = PN(BN(WS(Conv2DLayer(net, name='G%da'  % I, num_filters=nf(I-1), filter_size=3, pad=1, nonlinearity=vlrelu, W=irelu))))
        lods += [net]

    if progressive:
        lods = [WS(Conv2DLayer(l, name='Glod%d' % i, num_filters=num_channels, filter_size=3, pad=1, nonlinearity=linear, W=ilinear)) for i, l in enumerate(reversed(lods))]        # Should be this
        #lods = [WS(NINLayer(l, name='Glod%d' % i, num_units=num_channels, nonlinearity=linear, W=ilinear)) for i, l in enumerate(reversed(lods))]                                  # .. but this is better
        output_layer = LODSelectLayer(name='Glod', incomings=lods, cur_lod=cur_lod, first_incoming_lod=0)
    else:
        net = WS(Conv2DLayer(net, name='toRGB', num_filters=num_channels, filter_size=3, pad=1, nonlinearity=linear, W=ilinear))                                                    # Should be this
        #net = WS(NINLayer(net, name='toRGB', num_units=num_channels, nonlinearity=linear, W=ilinear))                                                                              # .. but this is better
        output_layer = net

    if tanh_at_end:
        output_layer = NonlinearityLayer(output_layer, name='Gtanh', nonlinearity=tanh)

    return dict(input_layers=input_layers, output_layers=[output_layer], cur_lod=cur_lod)

#----------------------------------------------------------------------------
# Cripped discriminator for MNIST mode recovery experiment.

def D_mnist_mode_recovery(
    num_channels    = 1,
    resolution      = 32,
    fmap_base       = 64,
    fmap_decay      = 1.0,
    fmap_max        = 256,
    mbstat_func     = 'Tstdeps',
    mbstat_avg      = None,         #'all',
    label_size      = 0,
    use_wscale      = False,
    use_gdrop       = False,
    use_layernorm   = False,
    use_batchnorm   = True,
    X               = 2,
    progressive     = False,
    **kwargs):

    R = int(np.log2(resolution))
    assert resolution == 2**R and resolution >= 4
    cur_lod = theano.shared(np.float32(0.0))
    gdrop_strength = theano.shared(np.float32(0.0))
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))) // X, fmap_max)
    def GD(layer): return GDropLayer(layer, name=layer.name+'gd', mode='prop', strength=gdrop_strength) if use_gdrop else layer
    def LN(layer): return LayerNormLayer(layer, name=layer.name+'ln') if use_layernorm else layer
    def WS(layer): return WScaleLayer(layer, name=layer.name+'ws') if use_wscale else layer
    def BN(layer): return lasagne.layers.batch_norm(layer) if use_batchnorm else layer

    net = input_layer = InputLayer(name='Dimages', shape=[None, num_channels, 2**R, 2**R])
    for I in xrange(R-1, 1, -1): # I = R-1, R-2, ..., 2     (i.e. 4,3,2)
        net = BN(LN(WS(Conv2DLayer     (GD(net),     name='D%da'   % I, num_filters=nf(I-1), filter_size=3, pad=1, nonlinearity=lrelu, W=ilrelu))))
        net =       Downscale2DLayer(net,         name='D%ddn'  % I, scale_factor=2)
        if progressive:
            lod =       Downscale2DLayer(input_layer, name='D%dxs'  % (I-1), scale_factor=2**(R-I))
            lod =    WS(NINLayer        (lod,         name='D%dx'   % (I-1), num_units=nf(I-1), nonlinearity=lrelu, W=ilrelu))
            net =       LODSelectLayer  (             name='D%dlod' % (I-1), incomings=[net, lod], cur_lod=cur_lod, first_incoming_lod=R-I-1)

    if mbstat_avg is not None:
        net = MinibatchStatConcatLayer(net, name='Dstat', func=globals()[mbstat_func], averaging=mbstat_avg)

    net = FlattenLayer(GD(net), name='Dflatten')
    output_layers = [WS(DenseLayer(net, name='Dscores', num_units=1, nonlinearity=linear, W=ilinear))]

    if label_size:
        output_layers += [WS(DenseLayer(net, name='Dlabels', num_units=label_size, nonlinearity=linear, W=ilinear))]
    return dict(input_layers=[input_layer], output_layers=output_layers, cur_lod=cur_lod, gdrop_strength=gdrop_strength)

#----------------------------------------------------------------------------
# Load a simple MNIST classifier.

def load_mnist_classifier(pkl_path):
    nl = lasagne.nonlinearities.LeakyRectify(0.1)

    net = InputLayer((None, 1, 32, 32))
    net = Conv2DLayer(net, 32, (3, 3), pad='same', nonlinearity=nl)
    net = Conv2DLayer(net, 32, (3, 3), pad='same', nonlinearity=nl)
    net = MaxPool2DLayer(net, (2, 2))
    net = Conv2DLayer(net, 55, (3, 3), pad='same', nonlinearity=nl)
    net = Conv2DLayer(net, 55, (3, 3), pad='same', nonlinearity=nl)
    net = MaxPool2DLayer(net, (2, 2))
    net = Conv2DLayer(net, 96, (3, 3), pad=0, nonlinearity=nl)
    net = Conv2DLayer(net, 96, (3, 3), pad=0, nonlinearity=nl)
    net = MaxPool2DLayer(net, (2, 2))
    net = DenseLayer(net, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

    with open(pkl_path, 'rb') as file:
        lasagne.layers.set_all_param_values(net, cPickle.load(file))
    return net

#----------------------------------------------------------------------------
