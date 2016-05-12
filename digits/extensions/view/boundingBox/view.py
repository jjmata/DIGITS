# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import numpy as np
import os
import PIL.Image
import PIL.ImageDraw

import digits
from digits.utils import subclass, override
from .forms import ConfigForm
from ..interface import VisualizationInterface

CONFIG_TEMPLATE = "config_template.html"
VIEW_TEMPLATE = "view_template.html"

@subclass
class Visualization(VisualizationInterface):
    """
    A visualization extension to display bounding boxes
    """

    def __init__(self, dataset, **kwargs):
        """
        Init
        """
        # arrow config
        color = kwargs['box_color']
        if color == "red":
            self.color = (255,0,0)
        elif color == "green":
            self.color = (0,255,0)
        elif color == "blue":
            self.color = (0,0,255)
        else:
            raise ValueError("unknown color: %s" % color)
        self.line_width = int(kwargs['line_width'])

        # memorize view template for later use
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        self.view_template = open(os.path.join(extension_dir, VIEW_TEMPLATE), "r").read()

    @staticmethod
    def get_config_form():
        return ConfigForm()

    @staticmethod
    def get_config_template(form):
        """
        parameters:
        - form: form returned by get_config_form(). This may be populated with values if the job was cloned
        return:
        - (template, context) tuple
          template is a Jinja template to use for rendering config options
          context is a dictionary of context variables to use for rendering the form
        """
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(os.path.join(extension_dir, CONFIG_TEMPLATE), "r").read()
        context = {'form': form}
        return (template, context)

    @staticmethod
    def get_id():
        return 'image-bounding-boxes'

    @staticmethod
    def get_title():
        return 'Bounding boxes'

    @override
    def get_view_template(self, data):
        """
        return:
        - (template, context) tuple
          template is a Jinja template to use for rendering config options
          context is a dictionary of context variables to use for rendering the form
        """
        return self.view_template, {'image': data['image']}

    @override
    def process_data(self, input_id, input_data, output_data):
        """
        Process one inference and return data to visualize
        """
        # get source image
        image = PIL.Image.fromarray(input_data).convert('RGB')
        draw = PIL.ImageDraw.Draw(image)

        # create arrays in expected format
        bboxes = []
        print output_data
        outputs = output_data[output_data.keys()[0]]
        for output in outputs:
            # last number is confidence
            if output[-1] > 0:
                box = ((output[0], output[1]), (output[2], output[3]))
                bboxes.append(box)
        digits.utils.image.add_bboxes_to_image(image, bboxes, self.color, self.line_width)
        image_html = digits.utils.image.embed_image_html(image)
        return {'image': image_html}

    @staticmethod
    def supports_dataset(data_extension_id):
        """
        returns true if view extension supports datasets created with specified data extension id
        extension_id may be None if dataset was not created using a data extension
        """
        return data_extension_id in [digits.extensions.data.kittiObject.data.DataIngestion.get_id()]





