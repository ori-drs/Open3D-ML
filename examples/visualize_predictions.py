import os

import open3d.ml as _ml3d
import open3d.ml.tf as ml3d
from open3d.ml.tf.vis import Visualizer, LabelLUT
from open3d.ml.tf.datasets import S3DIS

import numpy as np


DATA_SET_PATH = "set_path_to_dataset_here"

# Setup the visualizer and the lookup table
s3dis_labels = S3DIS.get_label_to_names()
viz = Visualizer()
lut = LabelLUT()
for val in sorted(s3dis_labels.keys()):
    lut.add_label(s3dis_labels[val], val)
viz.set_lut("labels", lut)
viz.set_lut("pred", lut)

# load config file
# -- If using a custom script use these two lines
# OPEN3D_ML_ROOT = os.environ['OPEN3D_ML_ROOT']
# cfg_file = os.path.join(OPEN3D_ML_ROOT, "ml3d/configs/randlanet_s3dis.yml")

# -- Else use this line
cfg_file = "Open3D-ML/ml3d/configs/randlanet_s3dis.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

# download checkpoint if necessay
randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_s3dis_202201071330utc.zip"
ckpt_folder = "./logs/"
ckpt_path = ckpt_folder + "RandLANet_S3DIS_tf/randlanet_s3dis_202201071330utc/ckpt-1"
if not os.path.exists(ckpt_path):
    cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
    os.system(cmd)

# load the dataset
cfg.dataset['dataset_path'] = DATA_SET_PATH
dataset = ml3d.datasets.S3DIS(cfg.dataset.pop('dataset_path', None), **cfg.dataset)

# load the model from the checkpoint
model = ml3d.models.RandLANet(**cfg.model)
pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)
pipeline.load_ckpt(ckpt_path=ckpt_path)

# get the test split and sample point
test_split = dataset.get_split("test")
sample_data = test_split.get_data(0)
sample_attr = test_split.get_attr(0)

# predict results on the sample point
pred_results = pipeline.run_inference(sample_data)
pred_label = (pred_results["predict_labels"]).astype(np.int32)

# create a dictionary for visualization
vis_d = {
        "name": sample_attr["name"],
        "points": sample_data['point'], # n x 3
        "labels": sample_data['label'], # n
        "colors": sample_data['feat'],
        "pred": pred_label, # n
    }

# send to visualizer. Note: it takes lists of dicts
viz.visualize([vis_d])