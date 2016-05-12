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
    A visualization extension to display raw data
    """

    def __init__(self, dataset, **kwargs):
        """
        Init
        """
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
        return (template, {})

    @staticmethod
    def get_id():
        return 'raw-data'

    @staticmethod
    def get_title():
        return 'Raw Data'

    @override
    def get_view_template(self, data):
        """
        return:
        - (template, context) tuple
          template is a Jinja template to use for rendering config options
          context is a dictionary of context variables to use for rendering the form
        """
        return self.view_template, {'data': data}

    @override
    def process_data(self, input_id, input_data, output_data):
        """
        Process one inference and return data to visualize
        """
        return output_data

    @staticmethod
    def supports_dataset(data_extension_id):
        """
        returns true if view extension supports datasets created with specified data extension id
        extension_id may be None if dataset was not created using a data extension
        """
        # supports all datasets
        return True





