# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from digits.utils import subclass, override

import os

class VisualizationInterface(object):
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
        raise NotImplementedError

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
        raise NotImplementedError

    @staticmethod
    def get_id():
        raise NotImplementedError

    @staticmethod
    def get_title():
        raise NotImplementedError

    def get_view_template(self):
        """
        return:
        - (template, context) tuple
          template is a Jinja template to use for rendering config options
          context is a dictionary of context variables to use for rendering the form
        """
        raise NotImplementedError

    def process_data(self, dataset, input_data, inference_data):
        raise NotImplementedError

    @staticmethod
    def supports_dataset(data_extension_id):
        """
        returns true if view extension supports datasets created with specified data extension id
        extension_id may be None if dataset was not created using a data extension
        """
        raise NotImplementedError





