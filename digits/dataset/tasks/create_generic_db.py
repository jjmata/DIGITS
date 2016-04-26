# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import re
import sys

import digits
from digits import device_query
from digits.task import Task
from digits.utils import subclass, override, constants
from digits.utils.image import embed_image_html
from digits.status import Status

@subclass
class CreateGenericDbTask(Task):
    """
    A task to create a db using a user-defined extension
    """

    def __init__(self, job, backend, stage, **kwargs):
        """
        Arguments:
        """
        self.job = job
        self.backend = backend
        self.stage = stage
        self.create_db_log_file = "create_%s_db.log" % stage
        self.dbs = {'features': None, 'labels': None}
        self.entry_count = 0
        self.feature_shape = None
        self.label_shape = None
        self.mean_file = None
        super(CreateGenericDbTask, self).__init__(**kwargs)

    @override
    def name(self):
        return 'Create %s DB' % self.stage

    @override
    def __getstate__(self):
        state = super(CreateGenericDbTask, self).__getstate__()
        if 'create_db_log' in state:
            # don't save file handle
            del state['create_db_log']
        return state

    @override
    def __setstate__(self, state):
        super(CreateGenericDbTask, self).__setstate__(state)

    @override
    def before_run(self):
        super(CreateGenericDbTask, self).before_run()
        # create log file
        self.create_db_log = open(self.path(self.create_db_log_file), 'a')
        # save job before spawning sub-process
        self.job.save()

    @override
    def process_output(self, line):
        self.create_db_log.write('%s\n' % line)
        self.create_db_log.flush()

        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False

        # keep track of which databases were created
        match = re.match(r'Created (features|labels) db for stage %s in (.*)' % self.stage, message)
        if match:
            db_type = match.group(1)
            self.dbs[db_type] = match.group(2)
            return True

        # mean file
        match = re.match(r'Created mean file for stage %s in (.*)' % self.stage, message)
        if match:
            self.mean_file = match.group(1)
            return True

        # entry counts
        match = re.match(r'Found (\d+) entries for stage %s' % self.stage, message)
        if match:
            count = int(match.group(1))
            self.entry_count = count
            return True

        # feature shape
        match = re.match(r'Feature shape for stage %s: (.*)' % self.stage, message)
        if match:
            self.feature_shape = eval(match.group(1))
            return True

        # feature shape
        match = re.match(r'Label shape for stage %s: (.*)' % self.stage, message)
        if match:
            self.label_shape = eval(match.group(1))
            return True

        # progress
        match = re.match(r'Processed (\d+)\/(\d+)', message)
        if match:
            self.progress = float(match.group(1))/int(match.group(2))
            self.emit_progress_update()
            return True

        return False

    @override
    def after_run(self):
        super(CreateGenericDbTask, self).after_run()

        self.create_db_log.close()

    @override
    def offer_resources(self, resources):
        reserved_resources = {}
        # we need one CPU resource from data_ingestion_task_pool
        cpu_key = 'data_ingestion_task_pool'
        if cpu_key not in resources:
            return None
        for resource in resources[cpu_key]:
            if resource.remaining() >= 1:
                reserved_resources[cpu_key] = [(resource.identifier, 1)]
                return reserved_resources
        return None

    @override
    def task_arguments(self, resources, env):

        args = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(digits.__file__))), 'tools', 'create_generic_db.py'),
            self.job.id(),
            '--stage=%s' % self.stage
            ]

        return args


