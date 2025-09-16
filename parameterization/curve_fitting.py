import argparse
from plyfile import PlyData, PlyElement
import os
import numpy as np
import trimesh
import time
import json
import point_cloud_utils as pcu
from tqdm import tqdm
from glob import glob
import torch
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path",default="/data1/yjx/research_data/mini_dataset_param/", type=str)
    # parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()

def read_points(dir_path):
    points_list = []
    for object in os.listdir(dir_path):
        file_path= os.path.join(dir_path, object,"point_cloud","iteration_30000", "point_cloud_segment.ply")
        plydata = PlyData.read(file_path)
        centers = np.stack((np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])), axis=1)
        mask = np.asarray(plydata.elements[0]["r"])
        centers = centers[mask>10]
        point_cloud = trimesh.PointCloud(centers)
    
        output_file = os.path.join(os.path.join(dir_path, object), f"curve_points.ply")
        point_cloud.export(output_file)
        points_list.append(point_cloud)
    return points_list

def curve_fitting(points,save_dir):
    pass






if __name__ == "__main__":
    args = parse_args()
    points_list = read_points(args.dir_path)
    # for points in points_list:
    #     curve_fitting(points,save_dir=args.dir_path)

