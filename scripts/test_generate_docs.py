# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import sys
import tempfile
import itertools
import unittest
import platform

try:
    import flask.ext.autodoc
except ImportError as e:
    raise unittest.SkipTest('Flask-Autodoc not installed')

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import digits.config; digits.config.load_config()
from digits.webapp import app, _doc as doc
from . import generate_docs as _

def check_doc_file(generator, doc_filename):
    """
    Checks that the output generated by generator matches the contents of doc_filename
    """
    # overcome temporary file permission errors
    tmp_file_name = tempfile.mkstemp(suffix='.md')
    os.close(tmp_file_name[0])
    with open(tmp_file_name[1],'w+') as tmp_file:
        generator.generate(tmp_file_name[1])
        tmp_file.seek(0)
        with open(doc_filename) as doc_file:
            # memory friendly
            for doc_line, tmp_line in itertools.izip(doc_file, tmp_file):
                doc_line = doc_line.strip()
                tmp_line = tmp_line.strip()
                if doc_line.startswith('*Generated') and \
                        tmp_line.startswith('*Generated'):
                    # If the date is different, that's not a problem
                    pass
                elif doc_line != tmp_line:
                    print '(Previous)', doc_line
                    print '(New)     ', tmp_line
                    raise RuntimeError('%s needs to be regenerated. Use scripts/generate_docs.py' % doc_filename)
    os.remove(tmp_file_name[1])

def test_api_md():
    with app.app_context():
        generator = _.ApiDocGenerator(doc)
        check_doc_file(generator,
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'API.md'))

def test_flask_routes_md():
    with app.app_context():
        generator = _.FlaskRoutesDocGenerator(doc)
        check_doc_file(generator,
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'FlaskRoutes.md'))


