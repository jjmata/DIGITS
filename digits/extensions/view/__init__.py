# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import imageGradients
from . import rawData

view_extensions = {
	imageGradients.Visualization,
	rawData.Visualization,
}

def get_extensions():
	"""
	return set of visualization extensions
	"""
	return view_extensions

def get_extension(extension_id):
	"""
	return extension associated with specified extension_id
	"""
	for extension in view_extensions:
		if extension.get_id() == extension_id:
			return extension
	return None
