import os
import subprocess
import torch

processes=[]
i=0
for t in os.listdir("/data1/yjx/research_data/ABC-NEF"):
    gpu_id = i % torch.cuda.device_count()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    i+=1

    full_source_path = os.path.join("/data1/yjx/research_data/ABC-NEF", t)
    full_target_path = os.path.join("/data1/yjx/research_data/ABC-NEF-2DGS", t)
    processes.append(subprocess.Popen(["python", "train.py", "-s", full_source_path, "-m", full_target_path], env=env))

for p in processes:
    p.wait()

