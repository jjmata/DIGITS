# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from digits.utils import subclass, override

import os

class DataIngestionInterface(object):
    """
    A data ingestion extension
    """

    def __init__(self, **kwargs):
        """
        Init
        """

        # save all data there - no other fields will be persisted
        self.userdata = kwargs

        # populate instance from userdata dictionary
        for k, v in self.userdata.items():
            setattr(self, k, v)

    def encode_entry(self, entry):
        """
        Return numpy.ndarray
        """
        raise NotImplementedError

    @staticmethod
    def get_category():
        raise NotImplementedError

    @staticmethod
    def get_dataset_form():
        raise NotImplementedError

    @staticmethod
    def get_dataset_template():
        raise NotImplementedError

    @staticmethod
    def get_id():
        raise NotImplementedError

    @staticmethod
    def get_inference_form():
        raise NotImplementedError

    @staticmethod
    def get_inference_template():
        raise NotImplementedError

    @staticmethod
    def get_title():
        raise NotImplementedError

    def get_user_data(self):
        """
        return serializable user data
        """
        return self.userdata

    def itemize_entries(self, stage):
        raise NotImplementedError





