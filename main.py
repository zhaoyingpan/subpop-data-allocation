import os
import argparse
from omegaconf import OmegaConf
from experiment_runner import ExperimentRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some variables.")
    parser.add_argument("--base_config_path", type=str, default="configs/base.yaml")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet18", "resnet50", "vit", "bert", "roberta", "deberta"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    if args.resume is not None:
        config = OmegaConf.load(os.path.join(args.resume, "config.yaml"))
        config.train.resume = args.resume
    else:
        if args.config_path is not None:
            base_conf = OmegaConf.load(args.base_config_path)
            override_conf = OmegaConf.load(args.config_path)
            config = OmegaConf.merge(base_conf, override_conf)
            config.data.dataset = args.dataset if args.dataset is not None else config.data.dataset
        else:
            assert args.dataset is not None
            config = OmegaConf.load(args.base_config_path)
            config.data.dataset = args.dataset

        config.train.batch_size = args.batch_size if args.batch_size is not None else config.train.batch_size
        config.name = args.exp_name if args.exp_name is not None else "ERM"
        config.train.model_type = args.model

        if args.steps is not None:
            config.train.num_steps = args.steps
        elif config.train.num_steps is None:
            default_steps = {
                "Waterbirds": 801,
                "waterbirds_syn": 801,
                "CelebA": 2001,
                "CelebA_subset": 2001,
                "CivilComments": 5001,
                "CivilComments_subset": 5001,
                "urbancars_syn": 2001,
            }
            if config.data.dataset not in default_steps:
                raise ValueError(
                    f"No default step count for dataset {config.data.dataset!r}; "
                    "provide --steps or set train.num_steps in the config."
                )
            config.train.num_steps = default_steps[config.data.dataset]
        
    exp_runner = ExperimentRunner(config=config)
    exp_runner.sweep_exp()
