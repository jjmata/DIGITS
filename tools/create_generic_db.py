#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

import argparse
# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
import lmdb
import logging
import numpy as np
import os
import sys

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()
from digits import extensions
from digits import utils, log
from digits.job import Job

# Run load_config() first to set the path to Caffe
import caffe.io
import caffe_pb2

logger = logging.getLogger('digits.tools.create_dataset')

BATCH_SIZE = 256

class DbWriter(object):
    """
    Abstract class for writing to databases
    """

    def __init__(self, output_dir):
        self._dir = output_dir

    def write_batch(self, batch):
        raise NotImplementedError

class LmdbWriter(DbWriter):

    def __init__(self, dataset_dir, stage, **kwargs):
        self.stage = stage
        os.makedirs(os.path.join(dataset_dir, stage))
        super(LmdbWriter, self).__init__(dataset_dir, **kwargs)

        # create LMDB for features
        self.feature_db = self.create_lmdb("features")
        # will create LMDB for labels later if necessary
        self.label_db = None

    def create_lmdb(self, db_type):
        sub_dir = os.path.join(self.stage, db_type)
        db_dir = os.path.join(self._dir, sub_dir)
        db = lmdb.open(
            db_dir,
            map_async=True,
            max_dbs=0)
        logger.info('Created %s db for stage %s in %s' % (db_type, self.stage, sub_dir))
        return db

    def write_batch(self, batch):
        # encode data into datum objects
        feature_datums = []
        label_datums = []
        for idx, (feature, label) in enumerate(batch):
            key = "%d" % idx
            if label.size > 1:
                # label is not a scalar
                if self.label_db is None:
                    self.label_db = self.create_lmdb("labels")
                label_datums.append((key,caffe.io.array_to_datum(label, 0)))
                # setting label to 0 - it will be unused as there is a dedicated label DB
                label = 0
            else:
                label = label[0]
            feature_datums.append((key,caffe.io.array_to_datum(feature, label)))
        self.write_datums(self.feature_db, feature_datums)
        if len(label_datums) > 0:
            self.write_datums(self.label_db, label_datums)

    def write_datums(self, db, batch):
        try:
            with db.begin(write=True) as lmdb_txn:
                for key, datum in batch:
                    lmdb_txn.put(key, datum.SerializeToString())
        except lmdb.MapFullError:
            # double the map_size
            curr_limit = db.info()['map_size']
            new_limit = curr_limit*2
            logger.info('Doubling LMDB map size to %sMB ...' % (new_limit>>20,))
            try:
                db.set_mapsize(new_limit) # double it
            except AttributeError as e:
                version = tuple(int(x) for x in lmdb.__version__.split('.'))
                if version < (0,87):
                    raise Error('py-lmdb is out of date (%s vs 0.87)' % lmdb.__version__)
                else:
                    raise e
            # try again
            self.write_datums(db, batch)

class DbCreator(object):

    def __init__(self):
        self.label_shape = None
        self.feature_shape = None

    def create_db(self, extension, stage, dataset_dir):
        # retrieve itemized list of entries
        entry_ids = extension.itemize_entries(stage)
        entry_count = len(entry_ids)

        if entry_count > 0:
            logger.info('Found %d entries for stage %s' % (entry_count, stage) )
            # create db
            db = LmdbWriter(dataset_dir, stage)
            data = []
            feature_sum = None
            processed_count = 0
            for entry in entry_ids:
                feature, label = extension.format_entry(entry)
                # check feature and label shapes
                if self.feature_shape is None:
                    # restrict to 3D data (Caffe Datum objects)
                    assert feature.ndim == 3, "Expecting 3D data"
                    self.feature_shape = feature.shape
                    feature_sum = np.zeros(self.feature_shape, np.float64)
                    logger.info('Feature shape for stage %s: %s' % (stage, repr(self.feature_shape)))
                else:
                    assert self.feature_shape == feature.shape
                if self.label_shape is None:
                    # restrict to 3D data (Caffe Datum objects) or scalars
                    assert label.ndim == 3 or label.size == 1, "Expecting 3D data or scalar"
                    self.label_shape = label.shape
                    logger.info('Label shape for stage %s: %s' % (stage, repr(self.label_shape)))
                else:
                    assert self.label_shape == label.shape
                feature_sum += feature
                data.append( (feature, label) )
                processed_count += 1
                if len(data) >= BATCH_SIZE:
                    db.write_batch(data)
                    data = []
                    logger.info('Processed %d/%d' % (processed_count, entry_count))
            if len(data) >= 0:
                db.write_batch(data)
            # write mean file
            self.save_mean(feature_sum, entry_count, dataset_dir, stage)

    def save_mean(self, feature_sum, entry_count, dataset_dir, stage):
        """
        Save mean to file
        """
        data = np.around(feature_sum / entry_count).astype(np.uint8)
        mean_file = os.path.join(stage, 'mean.binaryproto')
        # Transform to caffe's format requirements
        if data.ndim == 3:
            if data.shape[0] == 3:
                # channel swap
                # XXX see issue #59
                data = data[[2,1,0],...]
        elif mean.ndim == 2:
            # Add a channels axis
            data = data[np.newaxis,:,:]

        blob = caffe_pb2.BlobProto()
        blob.num = 1
        blob.channels, blob.height, blob.width = data.shape
        blob.data.extend(data.astype(float).flat)

        with open(os.path.join(dataset_dir, mean_file), 'wb') as outfile:
            outfile.write(blob.SerializeToString())

        logger.info('Created mean file for stage %s in %s' % (stage, mean_file) )

"""
Create a generic DB
"""
def create_generic_db(jobs_dir, dataset_id, stage):

    # job directory defaults to that defined in DIGITS config
    if jobs_dir == 'none':
        jobs_dir = digits.config.config_value('jobs_dir')

    # load dataset job
    dataset_dir = os.path.join(jobs_dir, dataset_id)
    assert os.path.isdir(dataset_dir), "Dataset dir %s does not exist" % dataset_dir
    dataset = Job.load(dataset_dir)

    # create instance of extension
    extension_id = dataset.extension_id
    extension_class = extensions.data.get_extension(extension_id)
    extension = extension_class(**dataset.extension_userdata)

    db_creator = DbCreator()

    stages = [utils.constants.TRAIN_DB, utils.constants.VAL_DB]
    db_creator.create_db(extension, stage, dataset_dir)

    logger.info('Dataset creation Done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset creation tool - DIGITS')

    ### Positional arguments

    parser.add_argument('dataset',
            help='Dataset Job ID')

    ### Optional arguments
    parser.add_argument('-j', '--jobs_dir',
            default='none',
            help='Jobs directory (default: from DIGITS config)',
            )

    parser.add_argument('-s', '--stage',
            default='train',
            help='Stage (train, val, test)',
            )

    args = vars(parser.parse_args())

    try:
        create_generic_db(
            args['jobs_dir'],
            args['dataset'],
            args['stage'],
                )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise

