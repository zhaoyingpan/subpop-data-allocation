import os
import time
import torch
import logging
import torch.nn.functional as F
from omegaconf import OmegaConf

from methods.methods import ERM
from eval import AverageMeter, GroupAcc
from datasets.get_data import get_data
from datasets.fast_dataloader import InfiniteDataLoader, FastDataLoader
from utils import reset_seed, read_results, save_results, get_timestamp, check_finish, check_entry

class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.dataset = config.data.dataset
        self.data_root = config.data.data_root
        if config.data.params_dict is not None and len(config.data.params_dict) == 0:
            raise ValueError("data.params_dict is empty; add at least one configuration")
        self.params_dict = {k: list(v) for k, v in config.data.params_dict.items()} if config.data.params_dict is not None else {"original": "original"}
        self.model_type = config.train.model_type
        if isinstance(config.train.num_steps, int):
            self.num_steps = config.train.num_steps
        else:
            self.num_steps = list(config.train.num_steps)
        self.lr = config.train.lr
        self.batch_size = config.train.batch_size
        self.seed_list = list(config.train.seed_list)
        self.device = torch.device("cuda") if config.train.num_gpus > 0 else torch.device("cpu")
        self.num_workers = config.train.num_workers
        self.val_freq = config.train.val_freq
        self.resume = config.train.resume

        self.result_root = config.log.result_root
        self.disable_saving = config.log.disable_saving

        self.finished = False

        self.metrics_names = ["va_loss", "va_acc_grps", "te_loss", "te_acc_grps"]
        self.tasks = list(self.params_dict.keys())
        
        self.init_result_dict()


        logging.info(f"Exp dir: {self.result_dir}")
        logging.info(f"Logs saved to: {self.log_path}")
        logging.info(f"Results saved to: {self.result_path}")
        logging.info(f"Checkpoints saved to: {self.ckpt_dir}")
        print(self.result_path)

    def get_exp_name(self):
        return f"{self.timestamp}_{self.config.name}"
        
    def init_result_dict(self):
        if self.resume is not None and os.path.isfile(os.path.join(self.resume, "result.pkl")):
            if check_finish(path=os.path.join(self.resume, "result.pkl"), tasks=self.tasks):
                self.finished = True
            else:
                self.result_dir = self.resume
                self.ckpt_dir = os.path.join(self.resume, "ckpt")
                self.result_path = os.path.join(self.resume, "result.pkl")
                self.timestamp = get_timestamp()
                self.log_path = os.path.join(self.resume, f"{self.timestamp}_resume_exp.log")

                logging.basicConfig(filename=self.log_path, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
                logging.info(f"resuming from {self.resume}")

                
                self.metrics_names = ["va_loss", "va_acc_grps", "te_loss", "te_acc_grps"]

                self.result_dict = read_results(self.result_path)
                for task in self.tasks:
                    self.result_dict.setdefault(task, AverageMeter(variable_list=[self.seed_list], metrics_names=self.metrics_names, seed_dim=0, config=self.config))

        elif self.resume is None:
            self.timestamp = get_timestamp()
            self.exp_name = self.get_exp_name()
            self.result_dict = {task: AverageMeter(variable_list=[self.seed_list], metrics_names=self.metrics_names, seed_dim=0, config=self.config) for task in self.tasks}
            self.result_dir = os.path.join(self.result_root, self.dataset, self.model_type, "ERM", self.exp_name)
            self.ckpt_dir = os.path.join(self.result_dir, "ckpt")
            self.result_path = os.path.join(self.result_dir, "result.pkl")
            self.log_path = os.path.join(self.result_dir, "exp.log")
            os.makedirs(self.result_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            OmegaConf.save(self.config, os.path.join(self.result_dir, "config.yaml"))

            logging.basicConfig(filename=self.log_path, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.info(f"Exp name: {self.exp_name}")

        elif os.path.isfile(self.resume) == False:
            raise FileNotFoundError(f"{self.resume} does not exist")

    def mark_finished(self):
        self.finished = True
        for task_name, task_dict in self.result_dict.items():
            task_dict.info["finished"] = True
    
    def trial(self, datasets, num_steps=None):
        trainset, valset, testset = datasets
        train_loader = InfiniteDataLoader(trainset, weights=None, batch_size=self.batch_size, num_workers=self.num_workers)
        val_loader = FastDataLoader(valset, batch_size=self.batch_size, num_workers=self.num_workers)
        test_loader = FastDataLoader(testset, batch_size=self.batch_size, num_workers=self.num_workers)

        method = ERM(
            config=self.config,
            num_classes=trainset.num_labels,
            num_attributes=trainset.num_attributes,
            group_sizes=trainset.group_sizes,
            num_steps=num_steps,
        )
        
        method.to(self.device)
        train_minibatches_iterator = iter(train_loader)
        num_steps = self.num_steps if num_steps is None else num_steps
        for step in range(num_steps):
            method.train()
            i, x, y, a = next(train_minibatches_iterator)
            minibatch_device = (i, x.to(self.device), y.to(self.device), a.to(self.device))
            step_vals = method.update(minibatch_device, step)

            if step % self.val_freq == 0:
                va_loss, va_acc, va_acc_grps = self.compute_acc(dataloader=val_loader, model=method)
                logging.info(f"[step {step}/{self.num_steps}] loss: {step_vals['loss']:.4f}, val loss: {va_loss:.4f}, val acc: {va_acc:.4f}, va grp acc: {va_acc_grps[0]:.4f} {va_acc_grps[1]:.4f} {va_acc_grps[2]:.4f} {va_acc_grps[3]:.4f}")
        
        te_loss, te_acc, te_acc_grps = self.compute_acc(dataloader=test_loader, model=method)
        
        return method, [va_loss, va_acc_grps, te_loss, te_acc_grps], [step_vals['loss'], 0]

    def sweep_exp(self):
        start = time.time()
        for seed_idx, seed in enumerate(self.seed_list):
            for task_idx, task in enumerate(self.tasks):
                logging.info(f"start {task_idx + 1}/{len(self.tasks)} {task} exp, seed {seed_idx + 1}/{len(self.seed_list)}")
                indices = seed_idx

                if check_entry(metrics=self.result_dict[task], indices=indices):
                    logging.info(f"skipping exp: seed {seed}, task {task}")
                    continue

                reset_seed(seed)
                datasets = get_data(param=self.params_dict[task], config=self.config, seed=seed)
                param = datasets[0].param
                if isinstance(self.num_steps, list):
                    method, metrics_values, log_values = self.trial(datasets=datasets, num_steps=self.num_steps[task_idx])
                else:
                    method, metrics_values, log_values = self.trial(datasets=datasets)
                
                if self.disable_saving == False:
                    model_path = self.save_model(model=method, task=task, seed=seed)
                else:
                    model_path = None

                self.update_results(metrics_values=metrics_values, log_values=log_values, model_path=model_path, task=task, indices=indices)

                save_results(self.result_dict, self.result_path)
                logging.info(f"finished {task_idx + 1}/{len(self.tasks)} {task} exp, seed {seed_idx + 1}/{len(self.seed_list)}")
                method.cpu()
                del method, datasets
                import gc
                gc.collect()
                torch.cuda.empty_cache()

        logging.info(f"Eclipsed time: {time.time() - start}")
        self.mark_finished()
        if isinstance(self.params_dict[task], str):
            self.params_dict[task] = param
            self.config.data.params_dict = self.params_dict
            OmegaConf.save(self.config, os.path.join(self.result_dir, "config.yaml"))
    
    def compute_acc(self, dataloader, model):
        total_loss = 0
        total_n = 0
        acc_meter = GroupAcc()
        model.eval()
        with torch.no_grad():
            for _, x, y, a in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                output = model.predict(x.to(self.device))
                loss = F.cross_entropy(output, y)
                pred = (torch.argmax(output.detach(), dim=1) == y).cpu()
                acc_meter.update(pred, y.cpu(), a)
        
                total_loss += loss.item() * y.size(0)
                total_n += y.size(0)
        
        avg_loss = total_loss / total_n
        avg_acc, acc_grps = acc_meter.get_values()
        
        return avg_loss, avg_acc, acc_grps

    def update_results(self, metrics_values, log_values, model_path, task, indices):
        assert len(self.metrics_names) == len(metrics_values)
        assert len(log_values) == 2
        for metrics_name, metrics_value in zip(self.metrics_names, metrics_values):
            self.result_dict[task].update(name=metrics_name, val=metrics_value, indices=indices)

        self.result_dict[task].update_log(log_values=log_values, indices=indices)

        self.result_dict[task].update_paths(path=model_path, indices=indices)

    def save_model(self, model, task, seed):
        param = self.params_dict[task]
        param_str = str(task) + "_" + "_".join(f"{p:.4f}" for p in param) if isinstance(param, list) else param
        model_path = os.path.join(self.ckpt_dir, f"task_{param_str}_seed_{seed:d}.pth")
        torch.save(model.state_dict(), model_path)
        return model_path
