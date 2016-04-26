# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

data_extensions = {
}

def get_extensions():
    """
    return set of data data extensions
    """
    return data_extensions

def get_extension(extension_id):
    """
    return extension associated with specified extension_id
    """
    for extension in data_extensions:
        if extension.get_id() == extension_id:
            return extension
    return None
