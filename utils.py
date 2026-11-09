import os
import glob
import torch
import pickle
import random
import shutil
import datetime
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def save_results(metrics, fn):
    with open(fn, 'wb') as f:
        pickle.dump(metrics, f)

def read_results(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def check_finish(path, tasks):
    try:
        result_dict = read_results(path)
        assert len(result_dict) == len(tasks)
        for task_name in tasks:
            assert result_dict[task_name].info["finished"] == True
        return True
    except:
        return False

def check_entry(metrics, indices):
    for metric_name, metric_value in metrics.data.items():
        if metric_value[indices].sum() == 0:
            return False
    return True

def ratio2param(ratios):
    ratios = np.array(ratios)
    return [(ratios[0] / ratios[:2].sum()).item(), (ratios[-1] / ratios[2:].sum()).item(), (ratios[:2].sum() / ratios.sum()).item()]

def param2ratio(param):
    alpha1, alpha0, beta = param
    return [alpha1 * beta, (1 - alpha1) * beta, (1 - alpha0) * (1 - beta), alpha0 * (1 - beta)]

def get_subgroup_masks(y, a):
    mask_11 = ((y == 1) & (a == 1))
    mask_10 = ((y == 1) & (a == 0))
    mask_01 = ((y == 0) & (a == 1))
    mask_00 = ((y == 0) & (a == 0))
    return mask_11, mask_10, mask_01, mask_00

def get_stats(y, a):
    masks = get_subgroup_masks(np.array(y), np.array(a))
    grp_sizes = np.array([m.sum() for m in masks])
    ratio = grp_sizes / grp_sizes.sum()
    param = ratio2param(ratio)
    print(f'grp sizes: {grp_sizes}, ratio: {ratio}, param: {param}')

def rm_empty_folder(root, remove=False):
    all_folders = glob.glob(os.path.join(root, "*/*/*/*"))
    for folder in all_folders:
        if os.path.exists(os.path.join(folder, "result.pkl")) == False and len(glob.glob(os.path.join(folder, "ckpt/*.pth"))) == 0:
            shutil.rmtree(folder) if remove else print(folder)
