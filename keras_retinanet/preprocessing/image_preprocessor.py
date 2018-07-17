"""
Copyright 2018 CVision AI (https://cvisionai.com)

"""

from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path

from .csv_generator import CSVGenerator

class ImagePreProcessor(CSVGenerator):
    def __init__(
        self,
        csv_data_file,
        csv_class_file,
        mean_image_file,
        image_data_generator,
        group_queue,
        image_queue,
        base_dir=None,
        **kwargs
    ):
        self.group_queue = group_queue
        self.image_queue = image_queue
        super(ImagePreProcessor, self).__init__(
            csv_data_file,
            csv_class_file,
            mean_image_file,
            image_data_generator,
            base_dir,
            group_queue=group_queue,
            **kwargs
        )

    def __next__(self):
        return self.next()

    def next(self):
        group = self.group_queue.get(True,None)
        self.image_queue.put(self.compute_input_output(group))

    def start(self):
        """ Starts pushing data onto queue.
        """
        while True:
            self.__next__()
