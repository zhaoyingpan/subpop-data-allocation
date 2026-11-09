import torch
import numpy as np
from omegaconf import OmegaConf
from utils import get_subgroup_masks

class AverageMeter(object):
    def __init__(self, variable_list, metrics_names, seed_dim, config):
        assert seed_dim != -1
        self.variables = variable_list
        self.num_per_variable = tuple(len(var_list) for var_list in variable_list)
        self.metrics_names = metrics_names
        self.seed_dim = seed_dim
        
        config_dict = OmegaConf.to_container(config, resolve=True)
        self.info = config_dict
        self.info["finished"] = False

        self.data = {name: self.reset(name) for name in metrics_names}
        # self.log = {"tr_loss": np.zeros(self.num_per_variable + (config.train.num_epochs,)), "tr_acc": np.zeros(self.num_per_variable + (config.train.num_epochs, 4))}
        self.log = {"tr_loss": np.zeros(self.num_per_variable), "tr_acc": np.zeros(self.num_per_variable)}
        self.model_paths = np.zeros(self.num_per_variable, dtype=object)
    
    def reset(self, name):
        return np.zeros(self.num_per_variable + (4,)) if "acc_grps" in name else np.zeros(self.num_per_variable)

    def update(self, name, val, indices):
        # indices need to be a tuple
        self.data[name][indices] = val
    
    def update_paths(self, path, indices):
        self.model_paths[indices] = path
    
    def update_log(self, log_values, indices):
        self.log["tr_loss"][indices] = log_values[0]
        self.log["tr_acc"][indices] = log_values[1]

    def seeds_avg(self):
        return {name: data.mean(axis=self.seed_dim) for name, data in self.data.items()}
    
    def seeds_var(self):
        return {name: data.var(axis=self.seed_dim) for name, data in self.data.items()}

class GroupAcc():
    def __init__(self):
        self.total_correct = 0
        self.total_n = 0
        self.correct_grps = np.zeros(4)
        self.n_grps = np.zeros(4)
    
    def update(self, pred, y, a):
        self.total_correct += pred.float().sum().item()
        self.total_n += y.size(0)

        mask_11, mask_10, mask_01, mask_00 = get_subgroup_masks(y=y, a=a)
        self.correct_grps[0] += pred[mask_11].float().sum().item()
        self.n_grps[0] += mask_11.float().sum().item()
    
        self.correct_grps[1] += pred[mask_10].float().sum().item()
        self.n_grps[1] += mask_10.float().sum().item()
    
        self.correct_grps[2] += pred[mask_01].float().sum().item()
        self.n_grps[2] += mask_01.float().sum().item()
    
        self.correct_grps[3] += pred[mask_00].float().sum().item()
        self.n_grps[3] += mask_00.float().sum().item()
    
    def get_values(self):
        avg_acc = self.total_correct / self.total_n if self.total_n != 0 else 0
        acc_11 = self.correct_grps[0] / self.n_grps[0] if self.n_grps[0] != 0 else 0
        acc_10 = self.correct_grps[1] / self.n_grps[1] if self.n_grps[1] != 0 else 0
        acc_01 = self.correct_grps[2] / self.n_grps[2] if self.n_grps[2] != 0 else 0
        acc_00 = self.correct_grps[3] / self.n_grps[3] if self.n_grps[3] != 0 else 0
        return avg_acc, np.array([acc_11, acc_10, acc_01, acc_00])


