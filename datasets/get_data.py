from datasets import subpop
from datasets.waterbirds_syn import WaterBirds_biased
from datasets.subpop_alter import CelebA_subset, CivilComments_subset
from datasets.urbancars_syn import get_urbancars_syn

def get_subpop_datasets(param, config, seed):
    trainset = vars(subpop)[config.data.dataset](config=config, split='tr', param=param, seed=seed)
    valset = vars(subpop)[config.data.dataset](config=config, split='va', param="original", seed=seed)
    testset = vars(subpop)[config.data.dataset](config=config, split='te', param="original", seed=seed)
    return trainset, valset, testset

def get_waterbirds_syn(param, config, seed, num_samples=None, initial_setup=None):
    trainset = WaterBirds_biased(param=param, data_root=config.data.data_root, split="tr", seed=seed, num_samples=num_samples, initial_setup=initial_setup)
    valset = WaterBirds_biased(param=(0.5, 0.5, 0.5), data_root=config.data.data_root, split="va", seed=seed)
    testset = WaterBirds_biased(param=(0.5, 0.5, 0.5), data_root=config.data.data_root, split="te", seed=seed)
    return trainset, valset, testset

def get_subset(param, config, seed):
    if config.data.dataset == "CelebA_subset":
        dataset_class = CelebA_subset
    elif config.data.dataset == "CivilComments_subset":
        dataset_class = CivilComments_subset
    else:
        raise ValueError(f"Unsupported subset dataset: {config.data.dataset}")
    trainset = dataset_class(config=config, split='tr', param=param, seed=seed)
    valset = dataset_class(config=config, split='va', param="original", seed=seed)
    testset = dataset_class(config=config, split='te', param="original", seed=seed)
    return trainset, valset, testset

def get_data(param, config, seed=None):
    if isinstance(param, list) and len(param) == 3:
        if min(param) < 0 or max(param) > 1:
            raise ValueError("Configuration parameters must lie in [0, 1]")

    if config.data.dataset == "waterbirds_syn":
        if not isinstance(param, list):
            raise ValueError("Waterbirds parameters must be provided as a list")
        if len(param) == 3:
            trainset, valset, testset = get_waterbirds_syn(param=param, config=config, seed=seed)
        elif len(param) == 4:
            trainset, valset, testset = get_waterbirds_syn(param=param[:3], config=config, seed=seed, num_samples=param[3])
        elif len(param) == 8:
            trainset, valset, testset = get_waterbirds_syn(param=param[:3], config=config, seed=seed, num_samples=param[3], initial_setup=param[4:])
        else:
            raise ValueError("Waterbirds expects 3, 4, or 8 parameters")
    elif config.data.dataset == 'urbancars_syn':
        if not isinstance(param, list):
            raise ValueError("UrbanCars parameters must be provided as a list")
        if len(param) == 3:
            trainset, valset, testset = get_urbancars_syn(param=param, seed=seed)
        elif len(param) == 4:
            trainset, valset, testset = get_urbancars_syn(param=param[:3], seed=seed, num_samples=param[3])
        else:
            raise ValueError("UrbanCars expects 3 or 4 parameters")
    elif config.data.dataset in ["CelebA_subset", "CivilComments_subset"]:
        trainset, valset, testset = get_subset(param=param, config=config, seed=seed)
    elif config.data.dataset in ("Waterbirds", "CelebA", "CivilComments"):
        trainset, valset, testset = get_subpop_datasets(param=param, config=config, seed=seed)
    else:
        raise ValueError(f"Unsupported dataset: {config.data.dataset}")
    
    return trainset, valset, testset


