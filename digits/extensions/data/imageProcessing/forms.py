# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from digits import utils
from digits.utils import subclass
from flask.ext.wtf import Form
import os
import wtforms
from wtforms import validators

@subclass
class DatasetForm(Form):
    """
    A form used to create an image processing dataset
    """

    def validate_folder_path(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) or not os.path.isdir(field.data):
                raise validators.ValidationError('Folder does not exist or is not reachable')
            else:
                return True

    source_folder = utils.forms.StringField(u'Source image folder',
            validators=[
                validators.DataRequired(),
                validate_folder_path,
                ],
            tooltip = "Indicate a folder full of images."
            )

    target_folder = utils.forms.StringField(u'Target image folder',
            validators=[
                validators.DataRequired(),
                validate_folder_path,
                ],
            tooltip = "Indicate a folder full of images. There must be one image per image in the source image folder. Image names do not matter but the images should match those of the source folder, when sorted alphanumerically."
            )

    folder_pct_val = utils.forms.IntegerField(u'% for validation',
            default=10,
            validators=[
                validators.DataRequired(),
                validators.NumberRange(min=0, max=100)
                ],
            tooltip = "You can choose to set apart a certain percentage of images from the training images for the validation set."
            )


