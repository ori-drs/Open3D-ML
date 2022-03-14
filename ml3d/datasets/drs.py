import numpy as np
import open3d as o3d
from pathlib import Path
import os, sys, glob
from os.path import join, exists, dirname, abspath, isdir
import logging

from .base_dataset import BaseDataset, BaseDatasetSplit

log = logging.getLogger(__name__)

class DRSDataset(BaseDataset):
    def __init__(self, 
                 dataset_path, 
                 name="DRSDataset", 
                 task='segmentation',                 
                 cache_dir='./logs/cache',
                 use_cache=False,
                 class_weights=[
                     3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                     650464, 791496, 88727, 1284130, 229758, 2272837
                 ],
                 num_points=40960,
                 test_result_folder='./test',
                 **kwargs):
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         task=task,                 
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         num_points=num_points,
                         class_weights=class_weights,
                         test_result_folder=test_result_folder,
                         **kwargs)
        assert isdir(dataset_path), f"Invalid dataset path {dataset_path}"
        # self.label_to_names = self.get_label_to_names()
        # self.num_classes = len(self.label_to_names)
        print("self.cfg.dataset_path is ", self.cfg.dataset_path)
        self.all_files = [f for f in glob.glob(self.cfg.dataset_path + "/*.pcd")]

    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'ceiling',
            1: 'floor',
            2: 'wall',
            3: 'beam',
            4: 'column',
            5: 'window',
            6: 'door',
            7: 'table',
            8: 'chair',
            9: 'sofa',
            10: 'bookcase',
            11: 'board',
            12: 'clutter'
        }
        return label_to_names

    def get_split(self, split):
        return DRSDatasetSplit(self, split=split)



    def get_split_list(self, split):
        # for now, return all files
        if split in ['test', 'testing']:
            return self.all_files
        elif split in ['val', 'validation']:
            return self.all_files
        elif split in ['train', 'training']:
            return self.all_files
        elif split in ['all']:
            return self.all_files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(self, attr):
        # checks whether attr['name'] is already tested.
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.npy')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        # save results['predict_labels'] to file.
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        pred = np.array(pred)

        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        store_path = join(path, self.name, name + '.npy')
        make_dir(Path(store_path).parent)
        np.save(store_path, pred)
        log.info("Saved {} in {}.".format(name, store_path))


class DRSDatasetSplit(BaseDatasetSplit):
    def __init__(self, dataset, split='train'):
        # collect list of files relevant to split.
        super().__init__(dataset, split=split)

        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset


    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        path = self.path_list[idx]
        pcd = o3d.io.read_point_cloud(path)
        pc = np.asarray(pcd.points)
        points = np.array([[x, y, z] for x,y,z in pc]).astype(np.float32)
        features = (np.ones((points.shape[0], 3)) * 125).astype(np.float32)
        labels = (np.ones((points.shape[0], 1))).astype(np.int32)
        # labels = np.array(pc[:, 6], dtype=np.int32).reshape((-1,))

        return {'point': points, 'feat': features, 'label': labels}

    def get_attr(self, idx):
        path = self.path_list[idx]
        name = path.split('/')[-1]
        return {'name': name, 'path': path, 'split': self.split}