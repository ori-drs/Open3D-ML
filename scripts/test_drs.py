import os
import open3d.ml as _ml3d
import open3d.ml.tf as ml3d
import numpy as np
import pathlib
from os.path import join
from tqdm import tqdm

# create a folder and move some pcd files into it
DATA_SET_PATH = "/home/lintong/data/S3DIS/george_building"
CKPT_FOLDER = "/home/lintong/code/Open3D-ML/models/s3dis/ckpt-1"

curret_file_dir = pathlib.Path(__file__).parent.resolve()
cfg_file = join(curret_file_dir, "../ml3d/configs/randlanet_s3dis.yml")
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.RandLANet(**cfg.model)
cfg.dataset['dataset_path'] = DATA_SET_PATH
dataset = ml3d.datasets.DRSDataset(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

# download the weights.
ckpt_folder = CKPT_FOLDER
pipeline.load_ckpt(ckpt_path=ckpt_folder)

## initialise visualisation
drs_labels = ml3d.datasets.DRSDataset.get_label_to_names()
viz = ml3d.vis.Visualizer()
lut = ml3d.vis.LabelLUT()
for val in sorted(drs_labels.keys()):
    lut.add_label(drs_labels[val], val)
viz.set_lut("labels", lut)
viz.set_lut("pred", lut)


vis_d_list = []
# drs dataset returns all pcd files in the folder
test_split = dataset.get_split('test')
for idx in tqdm(range(len(test_split)), desc='test'):
    attr = test_split.get_attr(idx)
    data = test_split.get_data(idx)
    # returns dict with 'predict_labels' and 'predict_scores'.
    pred_results = pipeline.run_inference(data)
    pred_label = (pred_results["predict_labels"]).astype(np.int32)
    # create a dictionary for visualization
    vis_d = {
        "name": attr['name'],
        "points": data['point'], # n x 3
        "labels": data['label'], # n
        "colors": data['feat'],
        "pred": pred_label, # n
    }
    vis_d_list.append(vis_d)

viz.visualize(vis_d_list)   
