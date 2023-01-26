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
import time
import pandas as pd 
import numpy as np
import random
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple
import torch
from torchvision import transforms
# from . import debug


def worker_init_fn(worker_id):
    # to be passed to torch.utils.data.DataLoader to fix the 
    #  random seed issue with numpy in multi-worker settings
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageFolder(torch.utils.data.Dataset):
    r"""
    Customised Image Folder class for pytorch.
    Usually accept image directory and a csv list as the input.
    Usage:
        dataset = ImageFolder(img_dir, img_list)
        dataset.set_transform(some_pytorch_transforms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
            num_workers=4, worker_init_fn=worker_init_fn)
        for x,y in loader:
            # x and y is input and target (dict), the keys can be customised.
    """
    _repr_indent = 4
    def __init__(self, data_dir, data_list, loader=pil_loader, transform=None, target_transform=None, **kwargs):
        self.root = data_dir
        self.loader = loader
        self.set_transform(transform, target_transform)
        self.build_data(data_list, data_dir)

    def set_transform(self, transform, target_transform=None):
        self.transform, self.target_transform = transform, target_transform

    def build_data(self, data_list, data_dir=None):
        """
        Args:
            data_list    (text file) must have at least 2 fields: path and label

        This method must create an attribute self.samples containing input and target samples; and another attribute N storing the dataset size

        Optional attributes: classes (list of unique classes), group (useful for 
        metric learning), N (dataset length)
        """
        assert isinstance(data_list, str) or isinstance(data_list, pd.DataFrame)
        df = pd.read_csv(data_list) if isinstance(data_list, str) else data_list
        assert 'path' in df and 'label' in df, f'[DATA] Error! {data_list} must contains "path" and "label".'
        paths = df['path'].tolist()
        labels = np.array(df['label'].tolist())
        self.N = len(labels)
        
        self.classes, inds = np.unique(labels, return_index=True)
        # class name to class index dict
        if '/' in paths[0] and os.path.exists(os.path.join(self.root, paths[0])):  # data organized by class name
            cnames = [paths[i].split('/')[0] for i in inds]
            self.class_to_idx = {key: val for key, val in zip(cnames, self.classes)}
        # class index to all samples within that class
        self.group = {}  # group by class index
        for key in self.classes:
            self.group[key] = np.nonzero(labels==key)[0]
        # self.labels = labels

        # check if data label avai
        # self.dlabels = np.array(df['dlabel'].tolist())
        # self.group_d = {}
        # for key in list(set(self.dlabels)):
        #     self.group_d[key] = np.nonzero(self.dlabels==key)[0]
        # self.dclasses = np.unique(self.dlabels)

        # self.samples = [(s[0], (s[1], s[2])) for s in zip(paths, self.labels, self.dlabels)]
        self.samples = {'x': paths, 'y': labels}

    @staticmethod
    def apply_transform(transform, x):
        if isinstance(transform, list):
            for t in transform:
                x = t(x)
        elif transform is not None:
            x = transform(x)
        return x

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            dict: (x: sample, y: target, **kwargs)
        """
        path, y = self.samples['x'][index], self.samples['y'][index]
        full_path = os.path.join(self.root, path)
        sample = self.loader(full_path)
        sample = self.apply_transform(self.transform, sample)
        y = self.apply_transform(self.target_transform, y)

        return {'x': sample}, {'y': y}

    def __len__(self) -> int:
        # raise NotImplementedError
        return self.N 

    def __repr__(self) -> str:
        head = "\nDataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
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


class MixupFolder(ImageFolder):
    _repr_indent = 4
    def __init__(self, data_dir, data_list, loader=pil_loader, transform=None, target_transform=None, hps=None):
        # self.mixup_beta = hps.mixup_beta
        self.mixup_level = hps.mixup_level  # 0: post aug, -1: pre aux
        assert self.mixup_level in [-1,0], 'This dataset class is for pre/post/no augment only. Mixuplevel must be in [0,-1]'
        self.mixup_samples = hps.mixup_samples
        self.mixup_same_label = hps.mixup_same_label
        self.mixup_ratio = [hps.mixup_beta] * (hps.mixup_samples+1)
        self.dct_target = hps.do_dct_target
        if hps.do_dct_target:
            from .image_tools import DCT
            self.dct = DCT(log=True)
                    
        self.do_hier_classify = hps.do_hier_classify
        self.do_compound_loss = hps.do_compound_loss
        super().__init__(data_dir, data_list, loader, transform, target_transform)


    def __getitem__(self, index: int) -> Any:
        ids = [index]
        if self.mixup_samples > 0:
            if self.mixup_same_label:  # compound from same labels
                label = self.samples['y_gan'][index]
                ids += np.random.choice(self.group[label], self.mixup_samples).tolist()
            else:
                ids += np.random.choice(self.N, self.mixup_samples).tolist()
        x , y_gans, y_sems = [], [], []
        for i in ids:
            path, y_gan, y_sem = self.samples['x'][i], self.samples['y_gan'][i], self.samples['y_sem'][i]
            full_path = os.path.join(self.root, path)
            sample = self.loader(full_path)
            if self.target_transform is not None:
                y_gan = self.target_transform(y_gan)
                y_sem = self.target_transform(y_sem)
            y_gans.append(y_gan)
            y_sems.append(y_sem)
            x.append(sample)
        # transform & mixup x
        beta = np.random.dirichlet(self.mixup_ratio)
        beta_t = torch.tensor(beta[:,None,None,None])
        if self.transform is not None:
            if isinstance(self.transform, list):
                x = [self.transform[0](x_) for x_ in x]
                if self.mixup_level==0:
                    x = [self.transform[1](x_) for x_ in x]
                    x = [self.transform[2](x_) for x_ in x]
                    x = torch.sum(torch.stack(x)*beta_t, dim=0).float()
                else:  # preaug
                    x = np.array([np.array(x_) for x_ in x])
                    x = (x*beta[:,None,None,None]).sum(axis=0)
                    x = Image.fromarray(np.uint8(x))
                    x = self.transform[2](self.transform[1](x))
            else:
                if self.mixup_level==0:
                    x = [self.transform(x_) for x_ in x]
                    x = torch.sum(torch.stack(x)*beta_t, dim=0).float()
                else:
                    x = np.array([np.array(x_) for x_ in x])
                    x = (x*beta[:,None,None,None]).sum(axis=0)
                    x = Image.fromarray(np.uint8(x))
                    x = self.transform(x)
        x = {'x': x, 'beta': beta}
        if self.mixup_same_label or self.mixup_samples==0:
            y_gan, y_sem = y_gans[0], y_sems[0]  # y_sem isn't needed
            y_out = {'y_gan': y_gan, 'y_sem': y_sem, 'y_det': 1 if y_gan else 0, 'beta': beta}
        else:
            y_gan = np.array(y_gans)  # gan 
            y_sem = np.array(y_sems)  # sem 
            y_out = {'y_gan': y_gan, 'y_sem': y_sem, 'y_det': np.int64(y_gan > 0), 'beta': beta}
        return x, y_out 

    @staticmethod
    def collate_fn(batch):
        # batch is a list of (x,y)
        x = {}
        for key in batch[0][0].keys():
            val = torch.stack([torch.as_tensor(b[0][key]) for b in batch])
            x[key] = val 

        y = {}
        for key in batch[0][1].keys():
            val = torch.stack([torch.as_tensor(b[1][key]) for b in batch])
            y[key] = val
        return x, y 


    # def get_random_batch_np(self, bsz):
    #     # for Yu paper
    #     out = [self[i] for i in np.random.choice(self.N, bsz)]
    #     x = np.array([x_[0]['x'] for x_ in out])
    #     y = np.array([x_[1]['y_gan'] for x_ in out])
    #     if bsz==1:
    #         x = x[None,...]
    #         y = y[None,...]
    #     return x, y


class SiameseFolder(torch.utils.data.Dataset):
    """
    dataset class for siamese contrastive loss
    Each index returns a pair of images + its class labels
    you can infer the relevantness of the pair using the class labels
    """
    _repr_indent = 4
    def __init__(self, data_folder1, data_folder2, train=True, transform=None):
        self.data1 = data_folder1
        self.data2 = data_folder2
        self.train = train
        self.classes = self.data1.classes  # categories array
        self.data1_labels = self.data1.labels
        self.data2_group = self.data2.group  # group by class
        self.class_set = set(self.classes)
        self.set_transform(transform)
        if not train:  # fixed pair for test
            rng = np.random.RandomState(29)
            pos_ids = [(i, rng.choice(self.data2_group[self.data1_labels[i]])) \
                    for i in range(0, len(self.data1),2)]
            neg_ids = [(i, rng.choice(self.data2_group[np.random.choice(list(self.class_set - set([self.data1_labels[i]])))])) \
                    for i in range(1, len(self.data1),2)]
            self.test_ids = pos_ids + neg_ids

    def set_transform(self, transform):
        if isinstance(transform, dict):
            self.data1.set_transform(transform['sketch'])
            self.data2.set_transform(transform['image'])
        else:
            self.data1.set_transform(transform)
            self.data2.set_transform(transform)

    def __len__(self):
        return len(self.data1)

    def __repr__(self) -> str:
        head = "\nDataset " + self.__class__.__name__ + \
                " consisting of two following subsets:"
        data1 = self.data1.__repr__()
        data2 = self.data2.__repr__()
        return '\n'.join([head, data1, data2])

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.train:
            y = np.random.choice(2)
            img1, label1 = self.data1[index]
            if y:  # relevant
                img2, label2 = self.data2[np.random.choice(self.data2_group[label1])]
                assert label1==label2, 'Error! Sanity check failed.'
            else:  # not relevant
                rnd_class = np.random.choice(list(self.class_set - set([label1])))
                img2, label2 = self.data2[np.random.choice(self.data2_group[rnd_class])]
                assert label2!=label1, "Error! Sanity check non-rel failed."
        else:
            img1, label1 = self.data1[self.test_ids[index][0]]
            img2, label2 = self.data2[self.test_ids[index][1]]
            # y = int(label1==label2)
        return (img1, img2), (label1, label2)


class MixupFolder2(ImageFolder):
    """
    mixup prior augmentation
    dataset_version=2
    """
    _repr_indent = 4
    def __init__(self, data_dir, data_list, loader=pil_loader, transform=None, target_transform=None, hps=None):
        self.mixup_beta = hps.mixup_beta
        self.mixup_samples = hps.mixup_samples
        self.mixup_ratio = [hps.mixup_beta] * (hps.mixup_samples+1)
        self.dct_target = hps.do_dct_target
        if hps.do_dct_target:
            from .image_tools import DCT
            self.dct = DCT(log=True)
                    
        self.do_hier_classify = hps.do_hier_classify
        self.do_compound_loss = hps.do_compound_loss
        assert not (self.do_hier_classify & self.do_compound_loss)  # cant both true
        super().__init__(data_dir, data_list, loader, transform, target_transform)

    def __getitem__(self, index):
        ids = [index]
        if self.mixup_samples > 0:
            beta = np.random.dirichlet(self.mixup_ratio)
            beta_t = torch.tensor(beta[:,None,None,None])
            if self.do_compound_loss:  # compound from same labels
                label = self.samples['y_gan'][index]
                ids += np.random.choice(self.group[label], self.mixup_samples).tolist()
            else:
                ids += np.random.choice(self.N, self.mixup_samples).tolist()
        x , y_gans, y_sems = [], [], []
        for i in ids:
            path, y_gan, y_sem = self.samples['x'][i], self.samples['y_gan'][i], self.samples['y_sem'][i]
            full_path = os.path.join(self.root, path)
            sample = self.loader(full_path)
            if self.transform is not None:
                if isinstance(self.transform, list):
                    x_pre = self.transform[0](sample)
                    x.append(x_pre)
                    # x_post = self.transform[1](x_pre)
                    # x = self.transform[2](x_post)
                else:
                    x.append(self.transform(sample))
            if self.target_transform is not None:
                y_gan = self.target_transform(y_gan)
                y_sem = self.target_transform(y_sem)
            y_gans.append(y_gan)
            y_sems.append(y_sem)
        # mix up
        if self.mixup_samples > 0:
            x = np.array([np.array(im) for im in x])  # (len(ids),H,W,C)
            x = (x*beta[:,None,None,None]).sum(axis=0)
            x_pre = Image.fromarray(np.uint8(x))
        else:
            x_pre = x[0]
        if isinstance(self.transform, list):
            x_post = self.transform[1](x_pre)
            x = self.transform[2](x_post)
        else:
            x = x_pre

        if self.do_compound_loss:
            y_gan, y_sem = y_gans[0], y_sems[0]  # y_sem isn't needed
            y_out = {'y_gan': y_gan, 'y_sem': y_sem, 'y_det': 1 if y_gan else 0}
        else:
            y_gan = np.array(y_gans)  # gan 
            y_sem = np.array(y_sems)  # sem 
            y_out = {'y_gan': y_gan, 'y_sem': y_sem, 'beta': beta}
        if self.dct_target:
            y_out['y_dct'] = self.transform[2](self.dct(x_pre))
            y_out['x_pre'] = self.transform[2](x_pre)
        return {'x': x}, y_out   


class MixupFolder3(ImageFolder):
    """
    return image separately (for CNN mixup and CNNTrueMixup)
    """
    _repr_indent = 4
    def __init__(self, data_dir, data_list, loader=pil_loader, transform=None, target_transform=None, hps=None):
        self.mixup_samples = hps.mixup_samples
        self.mixup_same_label = hps.mixup_same_label
        if self.mixup_samples > 0 and (not self.mixup_same_label):  # mixup
            self.mixup_ratio = [hps.mixup_beta] * (hps.mixup_samples+1)
            self.dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor(self.mixup_ratio))
        self.dct_target = hps.do_dct_target
        if hps.do_dct_target:
            from .image_tools import DCT
            self.dct = DCT(log=True)
                    
        super().__init__(data_dir, data_list, loader, transform, target_transform)

    def __getitem__(self, index):
        ids = [index]
        if self.mixup_samples > 0:
            if self.mixup_same_label:  # compound from same labels
                label = self.samples['y_gan'][index]
                ids += np.random.choice(self.group[label], self.mixup_samples).tolist()
            else:
                ids += np.random.choice(self.N, self.mixup_samples).tolist()
        x , y_gans, y_sems = [], [], []
        for i in ids:
            path, y_gan, y_sem = self.samples['x'][i], self.samples['y_gan'][i], self.samples['y_sem'][i]
            full_path = os.path.join(self.root, path)
            sample = self.loader(full_path)
            if self.transform is not None:
                if isinstance(self.transform, list):
                    x_pre = self.transform[0](sample)
                    x_post = self.transform[1](x_pre)
                    x.append(self.transform[2](x_post))
                else:
                    x.append(self.transform(sample))
            if self.target_transform is not None:
                y_gan = self.target_transform(y_gan)
                y_sem = self.target_transform(y_sem)
            y_gans.append(y_gan)
            y_sems.append(y_sem)
        x = {'x': torch.stack(x)}  # (n,c,h,w)
        if self.mixup_same_label:
            y_gan, y_sem = y_gans[0], y_sems[0]  # y_sem isn't needed
            y_out = {'y_gan': y_gan, 'y_sem': y_sem, 'y_det': 1 if y_gan else 0}
        else:
            x['beta'] = self.dirichlet.sample()
            y_gan = np.array(y_gans)  # gan 
            y_sem = np.array(y_sems)  # sem 
            y_out = {'y_gan': y_gan, 'y_sem': y_sem, 'y_det': np.int64(y_gan > 0), 'beta': x['beta'].clone()}
        
        return x, y_out   

    @staticmethod
    def collate_fn(batch):
        # batch is a list of (x,y)
        x = {}
        for key in batch[0][0].keys():
            if key=='x':
                val = torch.cat([b[0]['x'] for b in batch])
            else:
                val = torch.stack([torch.as_tensor(b[0][key]) for b in batch])
            x[key] = val 

        y = {}
        for key in batch[0][1].keys():
            val = torch.stack([torch.as_tensor(b[1][key]) for b in batch])
            y[key] = val
        return x, y 


# class MixupFolder(ImageFolder):
#     _repr_indent = 4
#     def __init__(self, data_dir, data_list, loader=pil_loader, transform=None, target_transform=None, hps=None):
#         self.mixup_beta = hps.mixup_beta
#         self.mixup_samples = hps.mixup_samples
#         self.mixup_ratio = [hps.mixup_beta] * (hps.mixup_samples+1)
#         self.dct_target = hps.do_dct_target
#         if hps.do_dct_target:
#             from .image_tools import DCT
#             self.dct = DCT(log=True)
                    
#         self.do_hier_classify = hps.do_hier_classify
#         self.do_compound_loss = hps.do_compound_loss
#         assert not (self.do_hier_classify & self.do_compound_loss)  # cant both true
#         super().__init__(data_dir, data_list, loader, transform, target_transform)

#     def get_datum(self, index):
#         path, y_gan, y_sem = self.samples['x'][index], self.samples['y_gan'][index], self.samples['y_sem'][index]
#         full_path = os.path.join(self.root, path)
#         sample = self.loader(full_path)
#         if self.transform is not None:
#             if isinstance(self.transform, list):
#                 x_pre = self.transform[0](sample)
#                 x_post = self.transform[1](x_pre)
#                 x = self.transform[2](x_post)
#             else:
#                 x = self.transform(sample)
#         if self.target_transform is not None:
#             y_gan = self.target_transform(y_gan)
#             y_sem = self.target_transform(y_sem)
#         out = {'x': x, 'y_gan': y_gan, 'y_sem': y_sem}
#         if self.dct_target:
#             out['y_dct'] = self.transform[2](self.dct(x_pre))
#             out['x_pre'] = self.transform[2](x_pre)
#         return out

#     def __getitem__(self, index: int) -> Any:
#         if self.mixup_samples==0:
#             sample = self.get_datum(index)
#             sample['y_det'] = 1 if sample['y_gan'] else 0
#             return {'x': sample['x']}, {key: val for key, val in sample.items() if key != 'x'}
#         beta = np.random.dirichlet(self.mixup_ratio)
#         beta_t = torch.tensor(beta[:,None,None,None])
#         if self.do_compound_loss:  # compound from same labels
#             label = self.samples['y_gan'][index]
#             ids = [index] + np.random.choice(self.group[label], self.mixup_samples).tolist()
#         else:
#             ids = [index] + np.random.choice(self.N, self.mixup_samples).tolist()
#         samples = [self.get_datum(i) for i in ids]
        
#         x = torch.stack([s['x'] for s in samples])
#         x = torch.sum(beta_t*x, dim=0).float()

#         if self.dct_target:
#             y_dct = torch.stack([s['y_dct'] for s in samples])
#             y_dct = torch.sum(beta_t*y_dct, dim=0).float()
#             x_pre = torch.stack([s['x_pre'] for s in samples])
#             x_pre = torch.sum(beta_t*x_pre, dim=0).float()
#         else:
#             y_dct = x_pre = None
#         if self.do_compound_loss:
#             y_gan = samples[0]['y_gan']  # =label
#             y_sem = samples[0]['y_sem']
#             y_out = {'y_gan': y_gan, 'y_sem': y_sem, 'y_det': 1 if y_gan else 0}
#         else:
#             y_gan = np.array([s['y_gan'] for s in samples])  # gan 
#             y_sem = np.array([s['y_sem'] for s in samples])  # sem 
#             y_out = {'y_gan': y_gan, 'y_sem': y_sem, 'beta': beta}
#         if y_dct is not None:
#             y_out['y_dct'] = y_dct
#             y_out['x_pre'] = x_pre

#         if self.do_hier_classify:  # fake and gan attribution
#             real = y_gan==0  # real index
#             y_gan0 = np.int64(~real)  # 0 real; 1 fake
#             beta0 = beta
#             y_gan1 = np.maximum(y_gan - 1,0)  # reduce index by 1, real img will have index 0 (with weight 0)
#             beta1 = beta.copy()
#             beta1[real] = 0.
#             y_out.update(y_gan0=y_gan0, y_gan1=y_gan1, beta0=beta0, beta1=beta1)

#         return {'x': x}, y_out    

#     # def get_random_batch_np(self, bsz):
#     #     # for Yu paper
#     #     out = [self[i] for i in np.random.choice(self.N, bsz)]
#     #     x = np.array([x_[0]['x'] for x_ in out])
#     #     y = np.array([x_[1]['y_gan'] for x_ in out])
#     #     if bsz==1:
#     #         x = x[None,...]
#     #         y = y[None,...]
#     #     return x, y

class TripletFolder(torch.utils.data.Dataset):
    """
    dataset class for triplet loss
    Each index returns a triplet of images + its class labels
    """
    _repr_indent = 4
    def __init__(self, data_folder1, data_folder2, train=True, transform=None):
        self.data1 = data_folder1
        self.data2 = data_folder2
        self.train = train
        self.classes = self.data1.classes  # categories array
        self.data1_labels = self.data1.labels
        self.data2_group = self.data2.group  # group by class
        self.class_set = set(self.classes)
        self.set_transform(transform)
        if not train:  # fixed triplet for test
            rng = np.random.RandomState(29)
            pos_ids = [rng.choice(self.data2_group[self.data1_labels[i]]) \
                    for i in range(0, len(self.data1))]
            neg_ids = [rng.choice(self.data2_group[np.random.choice(list(self.class_set - set([self.data1_labels[i]])))]) \
                    for i in range(0, len(self.data1))]
            self.test_ids = (pos_ids, neg_ids)

    def set_transform(self, transform):
        if isinstance(transform, dict):
            self.data1.set_transform(transform['sketch'])
            self.data2.set_transform(transform['image'])
        else:
            self.data1.set_transform(transform)
            self.data2.set_transform(transform)

    def __len__(self):
        return len(self.data1)

    def __repr__(self) -> str:
        head = "\nDataset " + self.__class__.__name__ + \
                " consisting of two following subsets:"
        data1 = self.data1.__repr__()
        data2 = self.data2.__repr__()
        return '\n'.join([head, data1, data2])

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        anchor, label_a = self.data1[index]
        if self.train:
            pos_id = np.random.choice(self.data2_group[label_a])
            pos, label_p = self.data2[pos_id]
            neg_class = np.random.choice(list(self.class_set - set([label_a])))
            neg, label_n = self.data2[np.random.choice(self.data2_group[neg_class])]
        else:  # validation
            pos, label_p = self.data2[self.test_ids[0][index]]
            neg, label_n = self.data2[self.test_ids[1][index]]
        assert label_a==label_p and label_a!=label_n, 'Error! Triplet sampling sanity check fails.'
        return (anchor, pos, neg), (label_a, label_p, label_n)


class TripletFolderInstance(torch.utils.data.Dataset):
    """
    triplet dataset class designed specifically for instance level data e.g. Sketchy 
    Each index returns a pair of images, its class labels and a relevant indicator y
        where y=1 if the pair is relevant else 0
    """
    _repr_indent = 4
    def __init__(self, data_folder1, data_folder2, train=True, transform=None, class_rate=0.2, index_branch=1):
        """
        class_rate: rate of sampling different classes for negatives
        """
        super().__init__()
        assert index_branch in [0,1], 'Error! Index branch must be 0 (anchor) or 1 (pos)'
        self.index_branch = index_branch
        self.data1 = data_folder1
        self.data2 = data_folder2
        self.train = train
        self.class_rate = class_rate
        self.classes = self.data1.classes  # categories array

        self.data1_labels = self.data1.labels
        self.data1_labels_ins = self.data1.labels_ins
        self.data2_labels = self.data2.labels
        self.data2_labels_ins = self.data2.labels_ins

        self.data1_group = self.data1.group 
        self.data1_group_ins = self.data1.group_ins
        self.data1_group_set = {key: set(val) for key, val in self.data1_group.items()}
        self.data2_group = self.data2.group  # group by class
        self.data2_group_ins = self.data2.group_ins  # group by instance-level class
        self.data2_group_set = {key: set(val) for key, val in self.data2_group.items()}  # convert data2_group from list to set

        self.class_set = set(self.classes)
        self.n = len(self.data1) if index_branch==0 else len(self.data2)
        self.set_transform(transform)
        if not train:  # fixed triplet for test
            rng = np.random.RandomState(29)
            if self.index_branch==0:  # anchor is index branch
                pos_ids, neg_ids = [], []
                for i in range(0, len(self.data1)):
                    if rng.rand() < self.class_rate:  # coarse level
                        pos_ids.append(rng.choice(self.data2_group[self.data1_labels[i]]))
                        neg_class = rng.choice(list(self.class_set - set([self.data1_labels[i]])))
                        neg_ids.append(rng.choice(self.data2_group[neg_class]))
                    else:  # instance level
                        a_label, a_label_ins = self.data1_labels[i], self.data1_labels_ins[i]
                        pos_ids.append(rng.choice(self.data2_group_ins[a_label_ins]))
                        # neg has same class label but different instance label
                        neg_ids.append(rng.choice(list(self.data2_group_set[a_label] - set(self.data2_group_ins[a_label_ins]))))
                self.test_ids = (pos_ids, neg_ids)
            else:  # positive is index branch
                anc_ids, neg_ids = [], []
                for i in range(0, len(self.data2)):
                    if rng.rand() < self.class_rate:  # coarse level
                        anc_ids.append(rng.choice(self.data1_group[self.data2_labels[i]]))
                        neg_class = rng.choice(list(self.class_set - set([self.data2_labels[i]])))
                        neg_ids.append(rng.choice(self.data2_group[neg_class]))
                    else:  # instance level
                        p_label, p_label_ins = self.data2_labels[i], self.data2_labels_ins[i]
                        anc_ids.append(rng.choice(self.data1_group_ins[p_label_ins]))
                        # neg has same class label but different instance label
                        neg_ids.append(rng.choice(list(self.data2_group_set[p_label] - set(self.data2_group_ins[p_label_ins]))))
                self.test_ids = (anc_ids, neg_ids)

    def set_transform(self, transform):
        if isinstance(transform, dict):
            self.data1.set_transform(transform['sketch'])
            self.data2.set_transform(transform['image'])
        else:
            self.data1.set_transform(transform)
            self.data2.set_transform(transform)

    def __len__(self):
        return self.n

    def __repr__(self) -> str:
        head = "\nDataset " + self.__class__.__name__ + \
                " consisting of two following subsets:"
        data1 = self.data1.__repr__()
        data2 = self.data2.__repr__()
        return '\n'.join([head, data1, data2])

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.index_branch==0:
            anchor, label_a = self.data1[index]
            label_a_ins = self.data1_labels_ins[index]
            if self.train:
                if np.random.rand() < self.class_rate:
                    pos_id = np.random.choice(self.data2_group[label_a])
                    neg_class = np.random.choice(list(self.class_set - set([label_a])))
                    neg_id = np.random.choice(self.data2_group[neg_class])
                else:
                    pos_id = np.random.choice(self.data2_group_ins[label_a_ins])
                    neg_id = np.random.choice(list(self.data2_group_set[label_a] - set(self.data2_group_ins[label_a_ins])))
                    assert label_a_ins==self.data2_labels_ins[pos_id] and label_a_ins!=self.data2_labels_ins[neg_id], \
                        'Error!Ins-level triplet sampling sanity check fails.'
            else:  # validation
                pos_id, neg_id = self.test_ids[0][index], self.test_ids[1][index]
            pos, label_p = self.data2[pos_id]
            neg, label_n = self.data2[neg_id]

        else:
            pos, label_p = self.data2[index]
            label_p_ins = self.data2_labels_ins[index]
            if self.train:
                if np.random.rand() < self.class_rate:
                    anc_id = np.random.choice(self.data1_group[label_p])
                    neg_class = np.random.choice(list(self.class_set - set([label_p])))
                    neg_id = np.random.choice(self.data2_group[neg_class])
                else:
                    anc_id = np.random.choice(self.data1_group_ins[label_p_ins])
                    neg_id = np.random.choice(list(self.data2_group_set[label_p] - set(self.data2_group_ins[label_p_ins])))
                    assert label_p_ins==self.data1_labels_ins[anc_id] and label_p_ins!=self.data2_labels_ins[neg_id], \
                        'Error!Ins-level triplet sampling sanity check fails.'
            else:  # validation
                anc_id, neg_id = self.test_ids[0][index], self.test_ids[1][index]

            anchor, label_a = self.data1[anc_id]
            neg, label_n = self.data2[neg_id]

        return (anchor, pos, neg), (label_a, label_p, label_n)