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
        The config template shows a form with view config options
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

    def get_summary_template(self):
        """
        This returns a summary of the job. This method is called after all entries have been processed.
        return:
        - (template, context) tuple
          template is a Jinja template to use for rendering the summary, or None if there is no summary to display
          context is a dictionary of context variables to use for rendering the form
        """
        return None, None

    @staticmethod
    def get_title():
        raise NotImplementedError

    def get_view_template(self):
        """
        The view template shows the visualization of one inference output
        parameters:
        - data: the data returned by process_data()
        return:
        - (template, context) tuple
          template is a Jinja template to use for rendering config options
          context is a dictionary of context variables to use for rendering the form
        """
        raise NotImplementedError

    def process_data(self, dataset, input_data, inference_data):
        raise NotImplementedError
