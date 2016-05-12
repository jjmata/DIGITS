# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import digits
from digits.utils import subclass, override
from .forms import ConfigForm
from ..interface import VisualizationInterface

import os

CONFIG_TEMPLATE = "config_template.html"
VIEW_TEMPLATE = "view_template.html"

@subclass
class Visualization(VisualizationInterface):
    """
    A visualization extension
    """

    def __init__(self, **kwargs):
        """
        Init
        """
        pass

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
        return digits.extensions.data.imageGradients.data.DataIngestion.get_id()

    @staticmethod
    def get_title():
        return digits.extensions.data.imageGradients.data.DataIngestion.get_title()

    @override
    def get_view_template(self):
        """
        return:
        - (template, context) tuple
          template is a Jinja template to use for rendering config options
          context is a dictionary of context variables to use for rendering the form
        """
        raise NotImplementedError

    @override
    def process_data(self, dataset, input_data, inference_data):
        raise NotImplementedError

    @staticmethod
    def supports_dataset(data_extension_id):
        """
        returns true if view extension supports datasets created with specified data extension id
        extension_id may be None if dataset was not created using a data extension
        """
        return data_extension_id == Visualization.get_id()





