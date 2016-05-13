# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import cv2 as cv
import math
import numpy as np
import operator
import os
import random

from digits.utils import image, subclass, override, constants
from ..interface import DataIngestionInterface
from .forms import DatasetForm
from .kitty import *

TEMPLATE = "template.html"

@subclass
class DataIngestion(DataIngestionInterface):
    """
    A data ingestion extension for a KITTI object detection dataset
    """

    def __init__(self, **kwargs):
        """
        Init
        """
        super(DataIngestion, self).__init__(**kwargs)

        # this instance is automatically populated with form field attributes by superclass constructor

        if ((self.val_image_folder == '') ^ (self.val_label_folder == '')):
            raise ValueError("You must specify either both val_image_folder and val_label_folder or none")

        if ((self.resize_image_width is None) ^ (self.resize_image_height is None)):
            raise ValueError("You must specify either both resize_image_width and resize_image_height or none")

        if self.resize_image_width is not None:
            self.resize_ratio = {
                                'width':  self.resize_image_width/self.padding_image_width,
                                'height': self.resize_image_height/self.padding_image_height,
                                }
        else:
            self.resize_ratio = {
                                'width':  1,
                                'height': 1,
                                }

        # this will be set when we know the phase we are encoding
        self.ground_truth = None

    @override
    def encode_entry(self, entry):
        """
        Return numpy.ndarray
        """
        image_filename = entry

        # (1) image part

        # load from file
        img = cv.imread(image_filename, cv.CV_LOAD_IMAGE_UNCHANGED)

        # pad
        img = self.pad_image(img)

        imgSizeX,imgSizeY = img.shape[1], img.shape[0]

        # resize
        img = resizeImage_onlyForImage(img, self.resize_ratio['width'], self.resize_ratio['height'])

        x_offset = img.shape[1] - imgSizeX
        y_offset = img.shape[0] - imgSizeY

        # If cropped image is larger than original image, output image will be zero-padded by the offset amount.
        if img.shape[1] > imgSizeX:
           x_offset /= 2
        if img.shape[0] > imgSizeY:
           y_offset /= 2
        img = cropImage_onlyForImage(img, x_offset, y_offset, imgSizeX, imgSizeY)

        if img.ndim == 2:
            # grayscale
            img = img[np.newaxis, :, :]
            if img.dtype == 'uint16':
                img = img.astype(float)
        else:
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("Unsupported image shape: %s" % repr(img.shape))
            # HWC -> CHW
            img = img.transpose(2,0,1)
            # BGR to RGB
            img = img[[2,1,0],...]

        # (2) label part

        # make sure label exists
        try:
            label_id = int(os.path.splitext(os.path.basename(entry))[0])
        except:
            raise ValueError("Unable to extract numerical id from file name %s" % entry)

        if not label_id in self.datasrc_annotation_dict:
            raise ValueError("Label key %s not found in label folder" % label_id)
        annotations = self.datasrc_annotation_dict[label_id]

        # collect bbox list into bboxList
        bboxList = []

        for bbox in annotations:
            # retrieve all vars defining groundtruth, and interpret all
            # serialized values as float32s:
            np_bbox = np.array(bbox.gt_to_lmdb_format())
            bboxList.append(np_bbox)

        bboxList = sorted(
            bboxList,
            key=operator.itemgetter(GroundTruthObj.lmdb_format_length()-1)
        )

        bboxList.reverse()

        # adjust bboxes according to image cropping
        bboxList = resize_bbox_list(bboxList, self.resize_ratio['width'], self.resize_ratio['height'])
        bboxList = cropImage_onlyForLabel(bboxList, x_offset, y_offset, imgSizeX, imgSizeY)

        # return data
        feature = img
        label = np.asarray(bboxList)

        # LMDB compaction: now label (aka bbox) is the joint datum
        datum_bbox = caffe.io.bbox_to_datum(label,
                                            0,
                                            max_bboxes=self.max_bboxes,
                                            bbox_width=GroundTruthObj.lmdb_format_length())
        # convert to numpy array
        label = caffe.io.datum_to_array(datum_bbox)

        return feature, label

    @staticmethod
    @override
    def get_category():
        return "Images"

    @staticmethod
    @override
    def get_id():
        return "image-kitti-object-detection"

    @staticmethod
    @override
    def get_dataset_form():
        return DatasetForm()

    @staticmethod
    @override
    def get_dataset_template(form):
        """
        parameters:
        - form: form returned by get_dataset_form(). This may be populated with values if the job was cloned
        return:
        - (template, context) tuple
          template is a Jinja template to use for rendering dataset creation options
          context is a dictionary of context variables to use for rendering the form
        """
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(os.path.join(extension_dir, TEMPLATE), "r").read()
        context = {'form': form}
        return (template, context)

    @staticmethod
    @override
    def get_title():
        return "KITTI Object Detection"

    @override
    def itemize_entries(self, stage):
        """
        return list of image file names to encode for specified stage
        """
        if stage == constants.TEST_DB:
            # don't retun anything for the test stage
            return []
        elif stage == constants.TRAIN_DB:
            # load ground truth
            self.load_ground_truth(self.train_label_folder)
            # get training image file names
            return self.make_image_list(self.train_image_folder)
        elif stage == constants.VAL_DB:
            if self.val_image_folder != '':
                # load ground truth
                self.load_ground_truth(self.val_label_folder)
                # get validation image file names
                return self.make_image_list(self.val_image_folder)
            else:
                # no validation folder was specified
                return []
        else:
            raise ValueError("Unknown stage: %s" % stage)

    def load_ground_truth(self, folder):
        """
        load ground truth from specified folder
        """
        datasrc = groundtruth(folder)
        datasrc.load_gt_obj()
        self.datasrc_annotation_dict = datasrc.objects_all

        scene_files = []
        for key in self.datasrc_annotation_dict:
            scene_files.append(key)

        # determine largest label height:
        self.max_bboxes = max([len(annotation) for annotation in self.datasrc_annotation_dict.values()])

    def make_image_list(self, folder):
        """
        find all supported images within specified folder and return list of file names
        """
        image_files = []
        for dirpath, dirnames, filenames in os.walk(folder, followlinks=True):
            for filename in filenames:
                if filename.lower().endswith(image.SUPPORTED_EXTENSIONS):
                    image_files.append('%s' % os.path.join(folder, filename))
        if len(image_files) == 0:
            raise ValueError("Unable to find supported images in %s" % folder)
        # shuffle
        random.shuffle(image_files)
        return image_files

    def pad_image(self, img):
        """
        pad a single image to the dimensions specified in form
        """
        src_shape = img.shape
        src_height = src_shape[0]
        src_width = src_shape[1]

        if self.padding_image_width < src_width:
            raise ValueError("Source image width %d is greater than padding width %d" % (src_width, self.padding_image_width) )

        if self.padding_image_height < src_height:
            raise ValueError("Source image height %d is greater than padding height %d" % (src_height, self.padding_image_height) )

        x_pad = self.padding_image_width - src_width
        y_pad = self.padding_image_height - src_height

        pad_image = cv.copyMakeBorder(img, 0, y_pad, 0, x_pad,
                                  cv.BORDER_CONSTANT, value=(0, 0, 0))
        return pad_image
