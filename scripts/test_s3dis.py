import os
import open3d.ml as _ml3d
import open3d.ml.tf as ml3d
import numpy as np
import pathlib
from os.path import join


DATA_SET_PATH = "/home/lintong/data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/"
CKPT_FOLDER = "/home/lintong/code/Open3D-ML/models/s3dis/ckpt-1"
VISUALISE_ONLY = True

curret_file_dir = pathlib.Path(__file__).parent.resolve()
cfg_file = join(curret_file_dir, "../ml3d/configs/randlanet_s3dis.yml")
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.RandLANet(**cfg.model)
cfg.dataset['dataset_path'] = DATA_SET_PATH
dataset = ml3d.datasets.S3DIS(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

# download the weights.
ckpt_folder = CKPT_FOLDER
pipeline.load_ckpt(ckpt_path=ckpt_folder)

if VISUALISE_ONLY:
    ## one dataset visualisation example
    s3dis_labels = ml3d.datasets.S3DIS.get_label_to_names()
    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    for val in sorted(s3dis_labels.keys()):
        lut.add_label(s3dis_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)

    test_split = dataset.get_split("test")
    data = test_split.get_data(0)

    # run inference on a single example.
    result = pipeline.run_inference(data)

    pred_label = (result['predict_labels']).astype(np.int32)
    vis_points = []
    vis_d = {
        "name": "S3dis",
        "points": data['point'],
        "labels": data['label'],
        "pred": pred_label,
    }
    vis_points.append(vis_d)
    v.visualize(vis_points)
else:
    # evaluate performance on the test set; this will write logs to './logs'.
    pipeline.run_test()