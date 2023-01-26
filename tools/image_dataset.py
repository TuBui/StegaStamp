#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
imagefolder loader
inspired from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
@author: Tu Bui @surrey.ac.uk
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import io
import time
import pandas as pd 
import numpy as np
import random
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple
# import torch
from .base_lmdb import PILlmdb, ArrayDatabase
# from . import debug


# def worker_init_fn(worker_id):
#     # to be passed to torch.utils.data.DataLoader to fix the 
#     #  random seed issue with numpy in multi-worker settings
#     torch_seed = torch.initial_seed()
#     random.seed(torch_seed + worker_id)
#     if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
#         torch_seed = torch_seed % 2**30
#     np.random.seed(torch_seed + worker_id)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageDataset():
    r"""
    Customised Image Folder class for pytorch.
    Accept lmdb and a csv list as the input.
    Usage:
        dataset = ImageDataset(img_dir, img_list)
        dataset.set_transform(some_pytorch_transforms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
            num_workers=4, worker_init_fn=worker_init_fn)
        for x,y in loader:
            # x and y is input and target (dict), the keys can be customised.
    """
    _repr_indent = 4
    def __init__(self, data_dir, data_list, transform=None, target_transform=None, **kwargs):
        super().__init__()
        self.set_transform(transform, target_transform)
        self.build_data(data_dir, data_list, **kwargs)
        self.kwargs = kwargs

    def set_transform(self, transform, target_transform=None):
        self.transform, self.target_transform = transform, target_transform

    def build_data(self, data_dir, data_list, **kwargs):
        """
        Args:
            data_list    (text file) must have at least 3 fields: id, path and label

        This method must create an attribute self.samples containing ID, input and target samples; and another attribute N storing the dataset size

        Optional attributes: classes (list of unique classes), group (useful for 
        metric learning)
        """
        self.data_dir, self.list = data_dir, data_list
        if ('dtype' in kwargs) and (kwargs['dtype'].lower() == 'array'):
            data = ArrayDatabase(data_dir, data_list)
        else:
            data = PILlmdb(data_dir, data_list, **kwargs)
        self.N = len(data)
        self.classes = np.unique(data.labels)
        self.samples = {'x': data, 'y': data.labels}
        # assert isinstance(data_list, str) or isinstance(data_list, pd.DataFrame)
        # df = pd.read_csv(data_list) if isinstance(data_list, str) else data_list
        # assert 'id' in df and 'label' in df, f'[DATA] Error! {data_list} must contains "id" and "label".'
        # ids = df['id'].tolist()
        # labels = np.array(df['label'].tolist())
        # data = PILlmdb(data_dir)
        # assert set(ids).issubset(set(data.keys))  # ids should exist in lmdb
        # self.N = len(ids)
        # self.classes, inds = np.unique(labels, return_index=True)
        # self.samples = {'id': ids, 'x': data, 'y': labels}

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            dict: (x: sample, y: target, **kwargs)
        """
        x, y = self.samples['x'][index], self.samples['y'][index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return {'x': x}, {'y': y}

    def __len__(self) -> int:
        # raise NotImplementedError
        return self.N 

    def __repr__(self) -> str:
        head = "\nDataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if hasattr(self, 'data_dir') and self.data_dir is not None:
            body.append("data_dir location: {}".format(self.data_dir))
        if hasattr(self, 'kwargs'):
            body.append(f'kwargs: {self.kwargs}')
        body += self.extra_repr().splitlines()
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self) -> str:
        return ""

