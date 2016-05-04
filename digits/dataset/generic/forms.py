# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os.path

import wtforms
from wtforms import validators

from ..forms import DatasetForm
from digits import utils
from digits.utils.forms import validate_required_iff

class GenericDatasetForm(DatasetForm):
    """
    Defines the form used to create a new GenericDatasetJob
    """

    feature_encoding = utils.forms.SelectField('Feature Encoding',
            default = 'png',
            choices = [
                ('none', 'None'),
                ('png', 'PNG (lossless)'),
                ('jpg', 'JPEG (lossy, 90% quality)'),
                ],
            tooltip = "Using either of these compression formats can save disk space, but can also require marginally more time for training."
            )

    label_encoding = utils.forms.SelectField('Label Encoding',
            default = 'none',
            choices = [
                ('none', 'None'),
                ('png', 'PNG (lossless)'),
                ('jpg', 'JPEG (lossy, 90% quality)'),
                ],
            tooltip = "Using either of these compression formats can save disk space, but can also require marginally more time for training."
            )

    batch_size = utils.forms.IntegerField('Encoder batch size',
            validators=[
                validators.DataRequired(),
                validators.NumberRange(min=1),
                ],
            default = 32,
            tooltip = "Encode data in batches of specified number of entries"
            )

    num_threads = utils.forms.IntegerField('Number of encoder threads',
            validators=[
                validators.DataRequired(),
                validators.NumberRange(min=1),
                ],
            default = 4,
            tooltip = "Use specified number of encoder threads"
            )

    backend = wtforms.SelectField('DB backend',
            choices = [
                ('lmdb', 'LMDB'),
                ],
            default='lmdb',
            )

