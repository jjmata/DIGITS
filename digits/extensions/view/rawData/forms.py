# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from digits import utils
from digits.utils import subclass
from flask.ext.wtf import Form
import os
import wtforms
from wtforms import validators

@subclass
class ConfigForm(Form):
    pass
