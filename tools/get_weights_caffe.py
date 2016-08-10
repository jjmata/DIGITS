#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
import sys
import os
import json
import logging
import argparse

import h5py
import numpy as np

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import digits.config
digits.config.load_config()
from digits import utils, log
from digits.inference.errors import InferenceError

# must call digits.config.load_config() before caffe to set the path
import caffe

logger = logging.getLogger('digits.tools.inference')

def get_weights(output_dir,net):
    """
    Get weights from a pretrained model
    """
    f = h5py.File(output_dir+'/filters.hdf5','a')

    #layers = net.layer
    #if len(layers) == 0:
    #    layers = net.layers

    # Save param keys to file:
    num_outputs = len(net.params)
    for index, layer in enumerate(net.params):

        shape = net.params[layer][0].data.shape
        raw_data = np.reshape(np.array(net.params[layer][0].data),shape)

        vis_data = utils.image.reshape_data_for_vis(raw_data,'BGR')
        dset = f.create_dataset(layer, data=utils.image.normalize_data(vis_data))

        # TODO: Add more stats
        dset.attrs['stats'] = json.dumps({"shape": shape, "num_activations": shape[0]})

        logger.info('Processed %s/%s', index, num_outputs)

    f.close()

def run(output_dir, model_def_path, weights_path):

    # net = caffe.proto.caffe_pb2.NetParameter()
    # with open(weights_path, 'rb') as infile:
    #    net.MergeFromString(infile.read())
    caffe.set_mode_cpu()
    net = caffe.Net(model_def_path,weights_path,caffe.TEST)
    get_weights(output_dir, net)

    logger.info('Saved data to %s', output_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Weights tool for pretrained models - DIGITS')

    ### Positional arguments
    parser.add_argument('output_dir',
            help='Directory to write outputs to')

    parser.add_argument('model_def_path',
            help='Path to model definition',
            )

    parser.add_argument('weights_path',
            help='Path to weights',
            )

    args = vars(parser.parse_args())

    try:
        run(
            args['output_dir'],
            args['model_def_path'],
            args['weights_path']
            )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
