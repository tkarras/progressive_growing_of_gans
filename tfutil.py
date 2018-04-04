# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import inspect
import importlib
import imp
import numpy as np
from collections import OrderedDict
import tensorflow as tf

#----------------------------------------------------------------------------
# Convenience.

def run(*args, **kwargs): # Run the specified ops in the default session.
    return tf.get_default_session().run(*args, **kwargs)

def is_tf_expression(x):
    return isinstance(x, tf.Tensor) or isinstance(x, tf.Variable) or isinstance(x, tf.Operation)

def shape_to_list(shape):
    return [dim.value for dim in shape]

def flatten(x):
    with tf.name_scope('Flatten'):
        return tf.reshape(x, [-1])

def log2(x):
    with tf.name_scope('Log2'):
        return tf.log(x) * np.float32(1.0 / np.log(2.0))

def exp2(x):
    with tf.name_scope('Exp2'):
        return tf.exp(x * np.float32(np.log(2.0)))

def lerp(a, b, t):
    with tf.name_scope('Lerp'):
        return a + (b - a) * t

def lerp_clip(a, b, t):
    with tf.name_scope('LerpClip'):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)

def absolute_name_scope(scope): # Forcefully enter the specified name scope, ignoring any surrounding scopes.
    return tf.name_scope(scope + '/')

#----------------------------------------------------------------------------
# Initialize TensorFlow graph and session using good default settings.

def init_tf(config_dict=dict()):
    if tf.get_default_session() is None:
        tf.set_random_seed(np.random.randint(1 << 31))
        create_session(config_dict, force_as_default=True)

#----------------------------------------------------------------------------
# Create tf.Session based on config dict of the form
# {'gpu_options.allow_growth': True}

def create_session(config_dict=dict(), force_as_default=False):
    config = tf.ConfigProto()
    for key, value in config_dict.items():
        fields = key.split('.')
        obj = config
        for field in fields[:-1]:
            obj = getattr(obj, field)
        setattr(obj, fields[-1], value)
    session = tf.Session(config=config)
    if force_as_default:
        session._default_session = session.as_default()
        session._default_session.enforce_nesting = False
        session._default_session.__enter__()
    return session

#----------------------------------------------------------------------------
# Initialize all tf.Variables that have not already been initialized.
# Equivalent to the following, but more efficient and does not bloat the tf graph:
#   tf.variables_initializer(tf.report_unitialized_variables()).run()

def init_uninited_vars(vars=None):
    if vars is None: vars = tf.global_variables()
    test_vars = []; test_ops = []
    with tf.control_dependencies(None): # ignore surrounding control_dependencies
        for var in vars:
            assert is_tf_expression(var)
            try:
                tf.get_default_graph().get_tensor_by_name(var.name.replace(':0', '/IsVariableInitialized:0'))
            except KeyError:
                # Op does not exist => variable may be uninitialized.
                test_vars.append(var)
                with absolute_name_scope(var.name.split(':')[0]):
                    test_ops.append(tf.is_variable_initialized(var))
    init_vars = [var for var, inited in zip(test_vars, run(test_ops)) if not inited]
    run([var.initializer for var in init_vars])

#----------------------------------------------------------------------------
# Set the values of given tf.Variables.
# Equivalent to the following, but more efficient and does not bloat the tf graph:
#   tfutil.run([tf.assign(var, value) for var, value in var_to_value_dict.items()]

def set_vars(var_to_value_dict):
    ops = []
    feed_dict = {}
    for var, value in var_to_value_dict.items():
        assert is_tf_expression(var)
        try:
            setter = tf.get_default_graph().get_tensor_by_name(var.name.replace(':0', '/setter:0')) # look for existing op
        except KeyError:
            with absolute_name_scope(var.name.split(':')[0]):
                with tf.control_dependencies(None): # ignore surrounding control_dependencies
                    setter = tf.assign(var, tf.placeholder(var.dtype, var.shape, 'new_value'), name='setter') # create new setter
        ops.append(setter)
        feed_dict[setter.op.inputs[1]] = value
    run(ops, feed_dict)

#----------------------------------------------------------------------------
# Autosummary creates an identity op that internally keeps track of the input
# values and automatically shows up in TensorBoard. The reported value
# represents an average over input components. The average is accumulated
# constantly over time and flushed when save_summaries() is called.
#
# Notes:
# - The output tensor must be used as an input for something else in the
#   graph. Otherwise, the autosummary op will not get executed, and the average
#   value will not get accumulated.
# - It is perfectly fine to include autosummaries with the same name in
#   several places throughout the graph, even if they are executed concurrently.
# - It is ok to also pass in a python scalar or numpy array. In this case, it
#   is added to the average immediately.

_autosummary_vars = OrderedDict() # name => [var, ...]
_autosummary_immediate = OrderedDict() # name => update_op, update_value
_autosummary_finalized = False

def autosummary(name, value):
    id = name.replace('/', '_')
    if is_tf_expression(value):
        with tf.name_scope('summary_' + id), tf.device(value.device):
            update_op = _create_autosummary_var(name, value)
            with tf.control_dependencies([update_op]):
                return tf.identity(value)
    else: # python scalar or numpy array
        if name not in _autosummary_immediate:
            with absolute_name_scope('Autosummary/' + id), tf.device(None), tf.control_dependencies(None):
                update_value = tf.placeholder(tf.float32)
                update_op = _create_autosummary_var(name, update_value)
                _autosummary_immediate[name] = update_op, update_value
        update_op, update_value = _autosummary_immediate[name]
        run(update_op, {update_value: np.float32(value)})
        return value

# Create the necessary ops to include autosummaries in TensorBoard report.
# Note: This should be done only once per graph.
def finalize_autosummaries():
    global _autosummary_finalized
    if _autosummary_finalized:
        return
    _autosummary_finalized = True
    init_uninited_vars([var for vars in _autosummary_vars.values() for var in vars])
    with tf.device(None), tf.control_dependencies(None):
        for name, vars in _autosummary_vars.items():
            id = name.replace('/', '_')
            with absolute_name_scope('Autosummary/' + id):
                sum = tf.add_n(vars)
                avg = sum[0] / sum[1]
                with tf.control_dependencies([avg]): # read before resetting
                    reset_ops = [tf.assign(var, tf.zeros(2)) for var in vars]
                    with tf.name_scope(None), tf.control_dependencies(reset_ops): # reset before reporting
                        tf.summary.scalar(name, avg)

# Internal helper for creating autosummary accumulators.
def _create_autosummary_var(name, value_expr):
    assert not _autosummary_finalized
    v = tf.cast(value_expr, tf.float32)
    if v.shape.ndims is 0:
        v = [v, np.float32(1.0)]
    elif v.shape.ndims is 1:
        v = [tf.reduce_sum(v), tf.cast(tf.shape(v)[0], tf.float32)]
    else:
        v = [tf.reduce_sum(v), tf.reduce_prod(tf.cast(tf.shape(v), tf.float32))]
    v = tf.cond(tf.is_finite(v[0]), lambda: tf.stack(v), lambda: tf.zeros(2))
    with tf.control_dependencies(None):
        var = tf.Variable(tf.zeros(2)) # [numerator, denominator]
    update_op = tf.cond(tf.is_variable_initialized(var), lambda: tf.assign_add(var, v), lambda: tf.assign(var, v))
    if name in _autosummary_vars:
        _autosummary_vars[name].append(var)
    else:
        _autosummary_vars[name] = [var]
    return update_op

#----------------------------------------------------------------------------
# Call filewriter.add_summary() with all summaries in the default graph,
# automatically finalizing and merging them on the first call.

_summary_merge_op = None

def save_summaries(filewriter, global_step=None):
    global _summary_merge_op
    if _summary_merge_op is None:
        finalize_autosummaries()
        with tf.device(None), tf.control_dependencies(None):
            _summary_merge_op = tf.summary.merge_all()
    filewriter.add_summary(_summary_merge_op.eval(), global_step)

#----------------------------------------------------------------------------
# Utilities for importing modules and objects by name.

def import_module(module_or_obj_name):
    parts = module_or_obj_name.split('.')
    parts[0] = {'np': 'numpy', 'tf': 'tensorflow'}.get(parts[0], parts[0])
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module('.'.join(parts[:i]))
            relative_obj_name = '.'.join(parts[i:])
            return module, relative_obj_name
        except ImportError:
            pass
    raise ImportError(module_or_obj_name)

def find_obj_in_module(module, relative_obj_name):
    obj = module
    for part in relative_obj_name.split('.'):
        obj = getattr(obj, part)
    return obj

def import_obj(obj_name):
    module, relative_obj_name = import_module(obj_name)
    return find_obj_in_module(module, relative_obj_name)

def call_func_by_name(*args, func=None, **kwargs):
    assert func is not None
    return import_obj(func)(*args, **kwargs)

#----------------------------------------------------------------------------
# Wrapper for tf.train.Optimizer that automatically takes care of:
# - Gradient averaging for multi-GPU training.
# - Dynamic loss scaling and typecasts for FP16 training.
# - Ignoring corrupted gradients that contain NaNs/Infs.
# - Reporting statistics.
# - Well-chosen default settings.

class Optimizer:
    def __init__(
        self,
        name                = 'Train',
        tf_optimizer        = 'tf.train.AdamOptimizer',
        learning_rate       = 0.001,
        use_loss_scaling    = False,
        loss_scaling_init   = 64.0,
        loss_scaling_inc    = 0.0005,
        loss_scaling_dec    = 1.0,
        **kwargs):

        # Init fields.
        self.name               = name
        self.learning_rate      = tf.convert_to_tensor(learning_rate)
        self.id                 = self.name.replace('/', '.')
        self.scope              = tf.get_default_graph().unique_name(self.id)
        self.optimizer_class    = import_obj(tf_optimizer)
        self.optimizer_kwargs   = dict(kwargs)
        self.use_loss_scaling   = use_loss_scaling
        self.loss_scaling_init  = loss_scaling_init
        self.loss_scaling_inc   = loss_scaling_inc
        self.loss_scaling_dec   = loss_scaling_dec
        self._grad_shapes       = None          # [shape, ...]
        self._dev_opt           = OrderedDict() # device => optimizer
        self._dev_grads         = OrderedDict() # device => [[(grad, var), ...], ...]
        self._dev_ls_var        = OrderedDict() # device => variable (log2 of loss scaling factor)
        self._updates_applied   = False

    # Register the gradients of the given loss function with respect to the given variables.
    # Intended to be called once per GPU.
    def register_gradients(self, loss, vars):
        assert not self._updates_applied

        # Validate arguments.
        if isinstance(vars, dict):
            vars = list(vars.values()) # allow passing in Network.trainables as vars
        assert isinstance(vars, list) and len(vars) >= 1
        assert all(is_tf_expression(expr) for expr in vars + [loss])
        if self._grad_shapes is None:
            self._grad_shapes = [shape_to_list(var.shape) for var in vars]
        assert len(vars) == len(self._grad_shapes)
        assert all(shape_to_list(var.shape) == var_shape for var, var_shape in zip(vars, self._grad_shapes))
        dev = loss.device
        assert all(var.device == dev for var in vars)

        # Register device and compute gradients.
        with tf.name_scope(self.id + '_grad'), tf.device(dev):
            if dev not in self._dev_opt:
                opt_name = self.scope.replace('/', '_') + '_opt%d' % len(self._dev_opt)
                self._dev_opt[dev] = self.optimizer_class(name=opt_name, learning_rate=self.learning_rate, **self.optimizer_kwargs)
                self._dev_grads[dev] = []
            loss = self.apply_loss_scaling(tf.cast(loss, tf.float32))
            grads = self._dev_opt[dev].compute_gradients(loss, vars, gate_gradients=tf.train.Optimizer.GATE_NONE) # disable gating to reduce memory usage
            grads = [(g, v) if g is not None else (tf.zeros_like(v), v) for g, v in grads] # replace disconnected gradients with zeros
            self._dev_grads[dev].append(grads)

    # Construct training op to update the registered variables based on their gradients.
    def apply_updates(self):
        assert not self._updates_applied
        self._updates_applied = True
        devices = list(self._dev_grads.keys())
        total_grads = sum(len(grads) for grads in self._dev_grads.values())
        assert len(devices) >= 1 and total_grads >= 1
        ops = []
        with absolute_name_scope(self.scope):

            # Cast gradients to FP32 and calculate partial sum within each device.
            dev_grads = OrderedDict() # device => [(grad, var), ...]
            for dev_idx, dev in enumerate(devices):
                with tf.name_scope('ProcessGrads%d' % dev_idx), tf.device(dev):
                    sums = []
                    for gv in zip(*self._dev_grads[dev]):
                        assert all(v is gv[0][1] for g, v in gv)
                        g = [tf.cast(g, tf.float32) for g, v in gv]
                        g = g[0] if len(g) == 1 else tf.add_n(g)
                        sums.append((g, gv[0][1]))
                    dev_grads[dev] = sums

            # Sum gradients across devices.
            if len(devices) > 1:
                with tf.name_scope('SumAcrossGPUs'), tf.device(None):
                    for var_idx, grad_shape in enumerate(self._grad_shapes):
                        g = [dev_grads[dev][var_idx][0] for dev in devices]
                        if np.prod(grad_shape): # nccl does not support zero-sized tensors
                            g = tf.contrib.nccl.all_sum(g)
                        for dev, gg in zip(devices, g):
                            dev_grads[dev][var_idx] = (gg, dev_grads[dev][var_idx][1])

            # Apply updates separately on each device.
            for dev_idx, (dev, grads) in enumerate(dev_grads.items()):
                with tf.name_scope('ApplyGrads%d' % dev_idx), tf.device(dev):

                    # Scale gradients as needed.
                    if self.use_loss_scaling or total_grads > 1:
                        with tf.name_scope('Scale'):
                            coef = tf.constant(np.float32(1.0 / total_grads), name='coef')
                            coef = self.undo_loss_scaling(coef)
                            grads = [(g * coef, v) for g, v in grads]

                    # Check for overflows.
                    with tf.name_scope('CheckOverflow'):
                        grad_ok = tf.reduce_all(tf.stack([tf.reduce_all(tf.is_finite(g)) for g, v in grads]))

                    # Update weights and adjust loss scaling.
                    with tf.name_scope('UpdateWeights'):
                        opt = self._dev_opt[dev]
                        ls_var = self.get_loss_scaling_var(dev)
                        if not self.use_loss_scaling:
                            ops.append(tf.cond(grad_ok, lambda: opt.apply_gradients(grads), tf.no_op))
                        else:
                            ops.append(tf.cond(grad_ok,
                                lambda: tf.group(tf.assign_add(ls_var, self.loss_scaling_inc), opt.apply_gradients(grads)),
                                lambda: tf.group(tf.assign_sub(ls_var, self.loss_scaling_dec))))

                    # Report statistics on the last device.
                    if dev == devices[-1]:
                        with tf.name_scope('Statistics'):
                            ops.append(autosummary(self.id + '/learning_rate', self.learning_rate))
                            ops.append(autosummary(self.id + '/overflow_frequency', tf.where(grad_ok, 0, 1)))
                            if self.use_loss_scaling:
                                ops.append(autosummary(self.id + '/loss_scaling_log2', ls_var))

            # Initialize variables and group everything into a single op.
            self.reset_optimizer_state()
            init_uninited_vars(list(self._dev_ls_var.values()))
            return tf.group(*ops, name='TrainingOp')

    # Reset internal state of the underlying optimizer.
    def reset_optimizer_state(self):
        run([var.initializer for opt in self._dev_opt.values() for var in opt.variables()])

    # Get or create variable representing log2 of the current dynamic loss scaling factor.
    def get_loss_scaling_var(self, device):
        if not self.use_loss_scaling:
            return None
        if device not in self._dev_ls_var:
            with absolute_name_scope(self.scope + '/LossScalingVars'), tf.control_dependencies(None):
                self._dev_ls_var[device] = tf.Variable(np.float32(self.loss_scaling_init), name='loss_scaling_var')
        return self._dev_ls_var[device]

    # Apply dynamic loss scaling for the given expression.
    def apply_loss_scaling(self, value):
        assert is_tf_expression(value)
        if not self.use_loss_scaling:
            return value
        return value * exp2(self.get_loss_scaling_var(value.device))

    # Undo the effect of dynamic loss scaling for the given expression.
    def undo_loss_scaling(self, value):
        assert is_tf_expression(value)
        if not self.use_loss_scaling:
            return value
        return value * exp2(-self.get_loss_scaling_var(value.device))

#----------------------------------------------------------------------------
# Generic network abstraction.
#
# Acts as a convenience wrapper for a parameterized network construction
# function, providing several utility methods and convenient access to
# the inputs/outputs/weights.
#
# Network objects can be safely pickled and unpickled for long-term
# archival purposes. The pickling works reliably as long as the underlying
# network construction function is defined in a standalone Python module
# that has no side effects or application-specific imports.

network_import_handlers = []    # Custom import handlers for dealing with legacy data in pickle import.
_network_import_modules = []    # Temporary modules create during pickle import.

class Network:
    def __init__(self,
        name=None,          # Network name. Used to select TensorFlow name and variable scopes.
        func=None,          # Fully qualified name of the underlying network construction function.
        **static_kwargs):   # Keyword arguments to be passed in to the network construction function.

        self._init_fields()
        self.name = name
        self.static_kwargs = dict(static_kwargs)

        # Init build func.
        module, self._build_func_name = import_module(func)
        self._build_module_src = inspect.getsource(module)
        self._build_func = find_obj_in_module(module, self._build_func_name)

        # Init graph.
        self._init_graph()
        self.reset_vars()

    def _init_fields(self):
        self.name               = None          # User-specified name, defaults to build func name if None.
        self.scope              = None          # Unique TF graph scope, derived from the user-specified name.
        self.static_kwargs      = dict()        # Arguments passed to the user-supplied build func.
        self.num_inputs         = 0             # Number of input tensors.
        self.num_outputs        = 0             # Number of output tensors.
        self.input_shapes       = [[]]          # Input tensor shapes (NC or NCHW), including minibatch dimension.
        self.output_shapes      = [[]]          # Output tensor shapes (NC or NCHW), including minibatch dimension.
        self.input_shape        = []            # Short-hand for input_shapes[0].
        self.output_shape       = []            # Short-hand for output_shapes[0].
        self.input_templates    = []            # Input placeholders in the template graph.
        self.output_templates   = []            # Output tensors in the template graph.
        self.input_names        = []            # Name string for each input.
        self.output_names       = []            # Name string for each output.
        self.vars               = OrderedDict() # All variables (localname => var).
        self.trainables         = OrderedDict() # Trainable variables (localname => var).
        self._build_func        = None          # User-supplied build function that constructs the network.
        self._build_func_name   = None          # Name of the build function.
        self._build_module_src  = None          # Full source code of the module containing the build function.
        self._run_cache         = dict()        # Cached graph data for Network.run().
        
    def _init_graph(self):
        # Collect inputs.
        self.input_names = []
        for param in inspect.signature(self._build_func).parameters.values():
            if param.kind == param.POSITIONAL_OR_KEYWORD and param.default is param.empty:
                self.input_names.append(param.name)
        self.num_inputs = len(self.input_names)
        assert self.num_inputs >= 1

        # Choose name and scope.
        if self.name is None:
            self.name = self._build_func_name
        self.scope = tf.get_default_graph().unique_name(self.name.replace('/', '_'), mark_as_used=False)
        
        # Build template graph.
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            assert tf.get_variable_scope().name == self.scope
            with absolute_name_scope(self.scope): # ignore surrounding name_scope
                with tf.control_dependencies(None): # ignore surrounding control_dependencies
                    self.input_templates = [tf.placeholder(tf.float32, name=name) for name in self.input_names]
                    out_expr = self._build_func(*self.input_templates, is_template_graph=True, **self.static_kwargs)
            
        # Collect outputs.
        assert is_tf_expression(out_expr) or isinstance(out_expr, tuple)
        self.output_templates = [out_expr] if is_tf_expression(out_expr) else list(out_expr)
        self.output_names = [t.name.split('/')[-1].split(':')[0] for t in self.output_templates]
        self.num_outputs = len(self.output_templates)
        assert self.num_outputs >= 1
        
        # Populate remaining fields.
        self.input_shapes   = [shape_to_list(t.shape) for t in self.input_templates]
        self.output_shapes  = [shape_to_list(t.shape) for t in self.output_templates]
        self.input_shape    = self.input_shapes[0]
        self.output_shape   = self.output_shapes[0]
        self.vars           = OrderedDict([(self.get_var_localname(var), var) for var in tf.global_variables(self.scope + '/')])
        self.trainables     = OrderedDict([(self.get_var_localname(var), var) for var in tf.trainable_variables(self.scope + '/')])

    # Run initializers for all variables defined by this network.
    def reset_vars(self):
        run([var.initializer for var in self.vars.values()])

    # Run initializers for all trainable variables defined by this network.
    def reset_trainables(self):
        run([var.initializer for var in self.trainables.values()])

    # Get TensorFlow expression(s) for the output(s) of this network, given the inputs.
    def get_output_for(self, *in_expr, return_as_list=False, **dynamic_kwargs):
        assert len(in_expr) == self.num_inputs
        all_kwargs = dict(self.static_kwargs)
        all_kwargs.update(dynamic_kwargs)
        with tf.variable_scope(self.scope, reuse=True):
            assert tf.get_variable_scope().name == self.scope
            named_inputs = [tf.identity(expr, name=name) for expr, name in zip(in_expr, self.input_names)]
            out_expr = self._build_func(*named_inputs, **all_kwargs)
        assert is_tf_expression(out_expr) or isinstance(out_expr, tuple)
        if return_as_list:
            out_expr = [out_expr] if is_tf_expression(out_expr) else list(out_expr)
        return out_expr

    # Get the local name of a given variable, excluding any surrounding name scopes.
    def get_var_localname(self, var_or_globalname):
        assert is_tf_expression(var_or_globalname) or isinstance(var_or_globalname, str)
        globalname = var_or_globalname if isinstance(var_or_globalname, str) else var_or_globalname.name
        assert globalname.startswith(self.scope + '/')
        localname = globalname[len(self.scope) + 1:]
        localname = localname.split(':')[0]
        return localname

    # Find variable by local or global name.
    def find_var(self, var_or_localname):
        assert is_tf_expression(var_or_localname) or isinstance(var_or_localname, str)
        return self.vars[var_or_localname] if isinstance(var_or_localname, str) else var_or_localname

    # Get the value of a given variable as NumPy array.
    # Note: This method is very inefficient -- prefer to use tfutil.run(list_of_vars) whenever possible.
    def get_var(self, var_or_localname):
        return self.find_var(var_or_localname).eval()
        
    # Set the value of a given variable based on the given NumPy array.
    # Note: This method is very inefficient -- prefer to use tfutil.set_vars() whenever possible.
    def set_var(self, var_or_localname, new_value):
        return set_vars({self.find_var(var_or_localname): new_value})

    # Pickle export.
    def __getstate__(self):
        return {
            'version':          2,
            'name':             self.name,
            'static_kwargs':    self.static_kwargs,
            'build_module_src': self._build_module_src,
            'build_func_name':  self._build_func_name,
            'variables':        list(zip(self.vars.keys(), run(list(self.vars.values()))))}

    # Pickle import.
    def __setstate__(self, state):
        self._init_fields()

        # Execute custom import handlers.
        for handler in network_import_handlers:
            state = handler(state)

        # Set basic fields.
        assert state['version'] == 2
        self.name = state['name']
        self.static_kwargs = state['static_kwargs']
        self._build_module_src = state['build_module_src']
        self._build_func_name = state['build_func_name']
        
        # Parse imported module.
        module = imp.new_module('_tfutil_network_import_module_%d' % len(_network_import_modules))
        exec(self._build_module_src, module.__dict__)
        self._build_func = find_obj_in_module(module, self._build_func_name)
        _network_import_modules.append(module) # avoid gc
        
        # Init graph.
        self._init_graph()
        self.reset_vars()
        set_vars({self.find_var(name): value for name, value in state['variables']})

    # Create a clone of this network with its own copy of the variables.
    def clone(self, name=None):
        net = object.__new__(Network)
        net._init_fields()
        net.name = name if name is not None else self.name
        net.static_kwargs = dict(self.static_kwargs)
        net._build_module_src = self._build_module_src
        net._build_func_name = self._build_func_name
        net._build_func = self._build_func
        net._init_graph()
        net.copy_vars_from(self)
        return net

    # Copy the values of all variables from the given network.
    def copy_vars_from(self, src_net):
        assert isinstance(src_net, Network)
        name_to_value = run({name: src_net.find_var(name) for name in self.vars.keys()})
        set_vars({self.find_var(name): value for name, value in name_to_value.items()})

    # Copy the values of all trainable variables from the given network.
    def copy_trainables_from(self, src_net):
        assert isinstance(src_net, Network)
        name_to_value = run({name: src_net.find_var(name) for name in self.trainables.keys()})
        set_vars({self.find_var(name): value for name, value in name_to_value.items()})

    # Create new network with the given parameters, and copy all variables from this network.
    def convert(self, name=None, func=None, **static_kwargs):
        net = Network(name, func, **static_kwargs)
        net.copy_vars_from(self)
        return net

    # Construct a TensorFlow op that updates the variables of this network
    # to be slightly closer to those of the given network.
    def setup_as_moving_average_of(self, src_net, beta=0.99, beta_nontrainable=0.0):
        assert isinstance(src_net, Network)
        with absolute_name_scope(self.scope):
            with tf.name_scope('MovingAvg'):
                ops = []
                for name, var in self.vars.items():
                    if name in src_net.vars:
                        cur_beta = beta if name in self.trainables else beta_nontrainable
                        new_value = lerp(src_net.vars[name], var, cur_beta)
                        ops.append(var.assign(new_value))
                return tf.group(*ops)

    # Run this network for the given NumPy array(s), and return the output(s) as NumPy array(s).
    def run(self, *in_arrays,
        return_as_list  = False,    # True = return a list of NumPy arrays, False = return a single NumPy array, or a tuple if there are multiple outputs.
        print_progress  = False,    # Print progress to the console? Useful for very large input arrays.
        minibatch_size  = None,     # Maximum minibatch size to use, None = disable batching.
        num_gpus        = 1,        # Number of GPUs to use.
        out_mul         = 1.0,      # Multiplicative constant to apply to the output(s).
        out_add         = 0.0,      # Additive constant to apply to the output(s).
        out_shrink      = 1,        # Shrink the spatial dimensions of the output(s) by the given factor.
        out_dtype       = None,     # Convert the output to the specified data type.
        **dynamic_kwargs):          # Additional keyword arguments to pass into the network construction function.

        assert len(in_arrays) == self.num_inputs
        num_items = in_arrays[0].shape[0]
        if minibatch_size is None:
            minibatch_size = num_items
        key = str([list(sorted(dynamic_kwargs.items())), num_gpus, out_mul, out_add, out_shrink, out_dtype])

        # Build graph.
        if key not in self._run_cache:
            with absolute_name_scope(self.scope + '/Run'), tf.control_dependencies(None):
                in_split = list(zip(*[tf.split(x, num_gpus) for x in self.input_templates]))
                out_split = []
                for gpu in range(num_gpus):
                    with tf.device('/gpu:%d' % gpu):
                        out_expr = self.get_output_for(*in_split[gpu], return_as_list=True, **dynamic_kwargs)
                        if out_mul != 1.0:
                            out_expr = [x * out_mul for x in out_expr]
                        if out_add != 0.0:
                            out_expr = [x + out_add for x in out_expr]
                        if out_shrink > 1:
                            ksize = [1, 1, out_shrink, out_shrink]
                            out_expr = [tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') for x in out_expr]
                        if out_dtype is not None:
                            if tf.as_dtype(out_dtype).is_integer:
                                out_expr = [tf.round(x) for x in out_expr]
                            out_expr = [tf.saturate_cast(x, out_dtype) for x in out_expr]
                        out_split.append(out_expr)
                self._run_cache[key] = [tf.concat(outputs, axis=0) for outputs in zip(*out_split)]

        # Run minibatches.
        out_expr = self._run_cache[key]
        out_arrays = [np.empty([num_items] + shape_to_list(expr.shape)[1:], expr.dtype.name) for expr in out_expr]
        for mb_begin in range(0, num_items, minibatch_size):
            if print_progress:
                print('\r%d / %d' % (mb_begin, num_items), end='')
            mb_end = min(mb_begin + minibatch_size, num_items)
            mb_in = [src[mb_begin : mb_end] for src in in_arrays]
            mb_out = tf.get_default_session().run(out_expr, dict(zip(self.input_templates, mb_in)))
            for dst, src in zip(out_arrays, mb_out):
                dst[mb_begin : mb_end] = src

        # Done.
        if print_progress:
            print('\r%d / %d' % (num_items, num_items))
        if not return_as_list:
            out_arrays = out_arrays[0] if len(out_arrays) == 1 else tuple(out_arrays)
        return out_arrays

    # Returns a list of (name, output_expr, trainable_vars) tuples corresponding to
    # individual layers of the network. Mainly intended to be used for reporting.
    def list_layers(self):
        patterns_to_ignore = ['/Setter', '/new_value', '/Shape', '/strided_slice', '/Cast', '/concat']
        all_ops = tf.get_default_graph().get_operations()
        all_ops = [op for op in all_ops if not any(p in op.name for p in patterns_to_ignore)]
        layers = []

        def recurse(scope, parent_ops, level):
            prefix = scope + '/'
            ops = [op for op in parent_ops if op.name == scope or op.name.startswith(prefix)]

            # Does not contain leaf nodes => expand immediate children.
            if level == 0 or all('/' in op.name[len(prefix):] for op in ops):
                visited = set()
                for op in ops:
                    suffix = op.name[len(prefix):]
                    if '/' in suffix:
                        suffix = suffix[:suffix.index('/')]
                    if suffix not in visited:
                        recurse(prefix + suffix, ops, level + 1)
                        visited.add(suffix)

            # Otherwise => interpret as a layer.
            else:
                layer_name = scope[len(self.scope)+1:]
                layer_output = ops[-1].outputs[0]
                layer_trainables = [op.outputs[0] for op in ops if op.type.startswith('Variable') and self.get_var_localname(op.name) in self.trainables]
                layers.append((layer_name, layer_output, layer_trainables))

        recurse(self.scope, all_ops, 0)
        return layers

    # Print a summary table of the network structure.
    def print_layers(self, title=None, hide_layers_with_no_params=False):
        if title is None: title = self.name
        print()
        print('%-28s%-12s%-24s%-24s' % (title, 'Params', 'OutputShape', 'WeightShape'))
        print('%-28s%-12s%-24s%-24s' % (('---',) * 4))

        total_params = 0
        for layer_name, layer_output, layer_trainables in self.list_layers():
            weights = [var for var in layer_trainables if var.name.endswith('/weight:0')]
            num_params = sum(np.prod(shape_to_list(var.shape)) for var in layer_trainables)
            total_params += num_params
            if hide_layers_with_no_params and num_params == 0:
                continue

            print('%-28s%-12s%-24s%-24s' % (
                layer_name,
                num_params if num_params else '-',
                layer_output.shape,
                weights[0].shape if len(weights) == 1 else '-'))

        print('%-28s%-12s%-24s%-24s' % (('---',) * 4))
        print('%-28s%-12s%-24s%-24s' % ('Total', total_params, '', ''))
        print()

    # Construct summary ops to include histograms of all trainable parameters in TensorBoard.
    def setup_weight_histograms(self, title=None):
        if title is None: title = self.name
        with tf.name_scope(None), tf.device(None), tf.control_dependencies(None):
            for localname, var in self.trainables.items():
                if '/' in localname:
                    p = localname.split('/')
                    name = title + '_' + p[-1] + '/' + '_'.join(p[:-1])
                else:
                    name = title + '_toplevel/' + localname
                tf.summary.histogram(name, var)

#----------------------------------------------------------------------------
