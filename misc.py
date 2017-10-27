# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import glob
import math
import types
import datetime
import cPickle
import numpy as np
from collections import OrderedDict 
import PIL.Image

#----------------------------------------------------------------------------
# Convenience wrappers for pickle.

def load_pkl(filename):
    with open(filename, 'rb') as file:
        return cPickle.load(file)

def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        cPickle.dump(obj, file, protocol=cPickle.HIGHEST_PROTOCOL)

#----------------------------------------------------------------------------
# Image save utils.

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) / grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in xrange(num):
        x = (idx % grid_w) * img_w
        y = (idx / grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid

def convert_to_pil_image(image, drange=[0,1]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.round(image).clip(0, 255).astype(np.uint8)
    format = 'RGB' if image.ndim == 3 else 'L'
    return PIL.Image.fromarray(image, format)

def save_image(image, filename, drange=[0,1]):
    convert_to_pil_image(image, drange).save(filename)

def save_image_grid(images, filename, drange=[0,1], grid_size=None):
    convert_to_pil_image(create_image_grid(images, grid_size), drange).save(filename)

#----------------------------------------------------------------------------
# Training utils.

def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def rampdown_linear(epoch, num_epochs, rampdown_length):
    if epoch >= num_epochs - rampdown_length:
        return float(num_epochs - epoch) / rampdown_length
    else:
        return 1.0

def format_time(seconds):
    s = int(np.round(seconds))
    if s < 60:         return '%ds'                % (s)
    elif s < 60*60:    return '%dm %02ds'          % (s / 60, s % 60)
    elif s < 24*60*60: return '%dh %02dm %02ds'    % (s / (60*60), (s / 60) % 60, s % 60)
    else:              return '%dd %dh %02dm'      % (s / (24*60*60), (s / (60*60)) % 24, (s / 60) % 60)

#----------------------------------------------------------------------------
# Logging of stdout and stderr to a file.

class OutputLogger(object):
    def __init__(self):
        self.file = None
        self.buffer = ''

    def set_log_file(self, filename):
        assert self.file is None
        self.file = open(filename, 'wt')
        if self.buffer is not None:
            self.file.write(self.buffer)
            self.buffer = None

    def write(self, data):
        if self.file is not None:
            self.file.write(data)
        if self.buffer is not None:
            self.buffer += data

    def flush(self):
        if self.file is not None:
            self.file.flush()

class TeeOutputStream(object):
    def __init__(self, child_streams, autoflush=False):
        self.child_streams = child_streams
        self.autoflush = autoflush
 
    def write(self, data):
        for stream in self.child_streams:
            stream.write(data)
        if self.autoflush:
            self.flush()

    def flush(self):
        for stream in self.child_streams:
            stream.flush()

output_logger = None

def init_output_logging():
    global output_logger
    if output_logger is None:
        output_logger = OutputLogger()
        sys.stdout = TeeOutputStream([sys.stdout, output_logger], autoflush=True)
        sys.stderr = TeeOutputStream([sys.stderr, output_logger], autoflush=True)

def set_output_log_file(filename):
    if output_logger is not None:
        output_logger.set_log_file(filename)

#----------------------------------------------------------------------------
# Reporting of results.

def create_result_subdir(result_dir, run_desc):

    # Select run ID and create subdir.
    while True:
        run_id = 0
        for fname in glob.glob(os.path.join(result_dir, '*')):
            try:
                fbase = os.path.basename(fname)
                ford = int(fbase[:fbase.find('-')])
                run_id = max(run_id, ford + 1)
            except ValueError:
                pass

        result_subdir = os.path.join(result_dir, '%03d-%s' % (run_id, run_desc))
        try:
            os.makedirs(result_subdir)
            break
        except OSError:
            if os.path.isdir(result_subdir):
                continue
            raise

    print "Saving results to", result_subdir
    set_output_log_file(os.path.join(result_subdir, 'log.txt'))

    # Export run details.
    try:
        import config
        with open(os.path.join(result_subdir, 'run.txt'), 'wt') as f:
            f.write('%-16s%s\n' % ('Date', datetime.datetime.today()))
            f.write('%-16s%s\n' % ('Device', config.theano_flags['device']))
            f.write('%-16s%s\n' % ('Working dir', os.getcwd()))
            f.write('%-16s%s\n' % ('Executable', sys.argv[0]))
            f.write('%-16s%s\n' % ('Arguments', ' '.join(sys.argv[1:])))
    except:
        pass

    # Export config.
    try:
        import config
        with open(os.path.join(result_subdir, 'config.txt'), 'wt') as fout:
            for k, v in sorted(config.__dict__.iteritems()):
                if not k.startswith('_'):
                    fout.write("%s = %s\n" % (k, str(v)))
    except:
        pass

    return result_subdir

#----------------------------------------------------------------------------
# Network topology info.

def print_network_topology_info(layers):
    import lasagne

    print
    print "%-16s%-28s%-10s%-20s%-20s%s" % ('LayerName', 'LayerType', 'Params', 'OutputShape', 'WeightShape', 'Activation')
    print "%-16s%-28s%-10s%-20s%-20s%s" % (('---',) * 6)
    total_params = 0

    for l in lasagne.layers.get_all_layers(layers):
        type_str        = type(l).__name__
        nparams         = sum(np.prod(p.get_value().shape) for p in l.get_params(trainable=True))
        total_params    += nparams
        outshape        = lasagne.layers.get_output_shape(l)
        try:
            weights = l.W.get_value()
        except AttributeError:
            try:
                weights = l.W_param.get_value()
            except AttributeError:
                weights = np.zeros(())
        weight_str      = shape_to_str(weights.shape)
        act_str         = '' if not hasattr(l, 'nonlinearity') else l.nonlinearity.__name__ if isinstance(l.nonlinearity, types.FunctionType) else type(l.nonlinearity).__name__
        print "%-16s%-28s%-10d%-20s%-20s%s" % (l.name, type_str, nparams, shape_to_str(outshape), weight_str, act_str)

    print "%-16s%-28s%-10s%-20s%-20s%s" % (('---',) * 6)
    print "%-16s%-28s%-10d%-20s%-20s%s" % ('Total', '', total_params, '', '', '')
    print

def shape_to_str(shape):
    str = ['%d' % v if v else '?' for v in shape]
    return ', '.join(str) if len(str) else ''

#----------------------------------------------------------------------------
# Locating results.

def locate_result_subdir(run_id):
    if isinstance(run_id, str) and os.path.isdir(run_id):
        return run_id

    searchdirs = []
    searchdirs += ['.']
    searchdirs += ['results']
    searchdirs += ['networks']

    import config
    for searchdir in searchdirs:
        dir = os.path.join(config.result_dir, searchdir, str(run_id))
        if os.path.isdir(dir):
            return dir
        dirs = glob.glob(os.path.join(config.result_dir, searchdir, '%s-*' % str(run_id)))
        if len(dirs) == 1 and os.path.isdir(dirs[0]):
            return dirs[0]
    raise IOError('Cannot locate result subdir for run', run_id)

def list_network_pkls(result_subdir):
    pkls = sorted(glob.glob(os.path.join(result_subdir, 'network-*.pkl')))
    if len(pkls) >= 1 and os.path.basename(pkls[0]) == 'network-final.pkl':
        pkls.append(pkls[0])
        del pkls[0]
    return pkls

def locate_network_pkl(result_subdir, snapshot=None):
    if isinstance(snapshot, str) and os.path.isfile(snapshot):
        return snapshot

    pkls = list_network_pkls(result_subdir)
    if len(pkls) >= 1 and snapshot is None:
        return pkls[-1]
    for pkl in pkls:
        try:
            name = os.path.splitext(os.path.basename(pkl))[0]
            number = int(name.split('-')[-1])
            if number == snapshot:
                return pkl
        except ValueError: pass
        except IndexError: pass
    raise IOError('Cannot locate network pkl for snapshot', snapshot)
    
#----------------------------------------------------------------------------
