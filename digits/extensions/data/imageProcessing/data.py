# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import math
import numpy as np
import os
import random

from digits.utils import image, subclass, override, constants
from ..interface import DataIngestionInterface
from .forms import DatasetForm

TEMPLATE = "template.html"


@subclass
class DataIngestion(DataIngestionInterface):
    """
    A data ingestion extension for an image processing dataset
    """

    def __init__(self, **kwargs):
        """
        Init
        """
        super(DataIngestion, self).__init__(**kwargs)

        self.random_indices = None

        if not 'seed' in self.userdata:
            # choose random seed and add to userdata so it gets persisted
            self.userdata['seed'] = random.randint(0, 1000)

        random.seed(self.userdata['seed'])

    @override
    def encode_entry(self, entry):
        """
        Return numpy.ndarray
        """
        source_image_file = entry[0]
        target_image_file = entry[1]

        source_image = self.encode_PIL_Image(
            image.load_image(source_image_file))
        target_image = self.encode_PIL_Image(
            image.load_image(target_image_file))

        return source_image, target_image

    def encode_PIL_Image(self, image):
        if self.channel_conversion != 'none':
            if image.mode != self.channel_conversion:
                # convert to different image mode if necessary
                image = image.convert(self.channel_conversion)
        # convert to numpy array
        image = np.array(image)
        # add channel axis if input is grayscale image
        if image.ndim == 2:
            image = image[..., np.newaxis]
        elif image.ndim != 3:
            raise ValueError("Unhandled number of channels: %d" % image.ndim)
        # transpose to CHW
        image = image.transpose(2, 0, 1)
        return image

    @staticmethod
    @override
    def get_category():
        return "Images"

    @staticmethod
    @override
    def get_id():
        return "image-processing"

    @staticmethod
    @override
    def get_dataset_form():
        return DatasetForm()

    @staticmethod
    @override
    def get_dataset_template(form):
        """
        parameters:
        - form: form returned by get_dataset_form(). This may be populated
        with values if the job was cloned
        returns:
        - (template, context) tuple
          - template is a Jinja template to use for rendering dataset creation
          options
          - context is a dictionary of context variables to use for rendering
          the form
        """
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(os.path.join(extension_dir, TEMPLATE), "r").read()
        context = {'form': form}
        return (template, context)

    @staticmethod
    @override
    def get_title():
        return "Processing"

    @override
    def itemize_entries(self, stage):
        if stage == constants.TEST_DB:
            # don't retun anything for the test stage
            return []

        # get image file names
        source_image_list = self.make_image_list(self.source_folder)
        target_image_list = self.make_image_list(self.target_folder)
        if len(source_image_list) != len(target_image_list):
            raise ValueError(
                "Expect same number of images in source and target folders (%d!=%d)" % (len(source_image_list), len(target_image_list)))

        return zip(
            self.split_image_list(source_image_list, stage),
            self.split_image_list(target_image_list, stage))

    def make_image_list(self, folder):
        image_files = []
        for dirpath, dirnames, filenames in os.walk(folder, followlinks=True):
            for filename in filenames:
                if filename.lower().endswith(image.SUPPORTED_EXTENSIONS):
                    image_files.append('%s' % os.path.join(folder, filename))
        if len(image_files) == 0:
            raise ValueError("Unable to find supported images in %s" % folder)
        return sorted(image_files)

    def split_image_list(self, list, stage):
        if self.random_indices is None:
            self.random_indices = random.shuffle(range(len(list)))
        pct_val = int(self.folder_pct_val)
        n_val_entries = int(math.floor(len(list) * pct_val / 100))
        if stage == constants.VAL_DB:
            return list[:n_val_entries]
        elif stage == constants.TRAIN_DB:
            return list[n_val_entries:]
        else:
            raise ValueError("Unknown stage: %s" % stage)
