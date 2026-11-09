import json
import random
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, Subset, ConcatDataset

from utils import param2ratio, get_subgroup_masks, reset_seed
from datasets.urbancars_utils import crop_mask_and_img, rescale_cropped_mask_and_img, Places, StanfordCars


class UrbanCarsSyn(Dataset):
    def __init__(self, param, split, seed, num_samples=None):
        reset_seed(seed)
        self.param = param
        self.ratio = param2ratio(param)
        self.seed = seed
        
        self.split = split
        self.meta_data_fpath = './data/Whac-A-Mole/create_datasets/urbancars/metadata/urbancars/meta_data.json'
        self.num_train = num_samples if num_samples is not None else 4000
        self.num_val, self.num_test = 500, 500
        self.target_image_size = 224
        self.ratio_dict = {
            (0, 0): self.ratio[3],
            (0, 1): self.ratio[2],
            (1, 0): self.ratio[1],
            (1, 1): self.ratio[0],
        }

        self.setup()
        self._gen_composition()

        masks = get_subgroup_masks(np.array(self.y), np.array(self.a))
        self.num_labels = 2
        self.num_attributes = 2
        self.group_sizes = [m.sum() for m in masks]
        
    def setup(self):

        with open(self.meta_data_fpath) as f:
            meta_data = json.load(f)

        self.obj_dict_list = meta_data["object"]
        self.bg_dict_list = meta_data["background"]
        assert (
            len(self.obj_dict_list)
            == len(self.bg_dict_list)
        )
        if self.split == "train":
            split_indices = list(range(self.num_train))
            num_img_split = self.num_train
        elif self.split == "val":
            split_indices = list(
                range(self.num_train, self.num_train + self.num_val)
            )
            num_img_split = self.num_val
        elif self.split == "test":
            split_indices = list(
                range(
                    self.num_train + self.num_val,
                    self.num_train + self.num_val + self.num_test,
                )
            )
            num_img_split = self.num_test
        else:
            raise NotImplementedError

        self.num_img_split = num_img_split

        self.obj_cat_name_to_iter_loader = {}
        for obj_dict in self.obj_dict_list:
            obj_name = obj_dict["name"]
            obj_id = obj_dict["id"]
            stanford_cars_train = StanfordCars(
                "./data/Whac-A-Mole/data",
                obj_id,
                split="train",
                transform=T.ToTensor(),
                mask_transform=T.ToTensor(),
            )
            stanford_cars_test = StanfordCars(
                "./data/Whac-A-Mole/data",
                obj_id,
                split="test",
                transform=T.ToTensor(),
                mask_transform=T.ToTensor(),
            )
            stanford_cars = ConcatDataset(
                [stanford_cars_train, stanford_cars_test]
            )
            assert (
                len(stanford_cars)
                >= self.num_train + self.num_val + self.num_test
            )
            subsample_indices = list(range(len(stanford_cars)))
            random.Random(self.seed).shuffle(subsample_indices)
            stanford_cars = Subset(stanford_cars, subsample_indices)
            stanford_cars = Subset(stanford_cars, split_indices)
            loader = torch.utils.data.DataLoader(
                stanford_cars,
                batch_size=1,
            )
            self.obj_cat_name_to_iter_loader[obj_name] = iter(loader)

        self.bg_cat_name_to_iter_loader = {}
        for bg_dict in self.bg_dict_list:
            bg_cat_name = bg_dict["name"]

            places_dataset_list = []

            for places_dict in bg_dict["class_list"]:
                places_bg_name = places_dict["name"]
                places_dataset = Places(
                    "./data/Whac-A-Mole/data",
                    places_bg_name,
                )
                places_dataset_list.append(places_dataset)

            concat_places_dataset = ConcatDataset(places_dataset_list)
            assert (
                len(concat_places_dataset)
                >= self.num_train + self.num_val + self.num_test
            )

            subsample_indices = list(range(len(concat_places_dataset)))
            random.Random(self.seed).shuffle(subsample_indices)
            concat_places_dataset = Subset(
                concat_places_dataset, subsample_indices
            )
            concat_places_dataset = Subset(concat_places_dataset, split_indices)
            loader = torch.utils.data.DataLoader(
                concat_places_dataset,
                batch_size=1,
            )
            self.bg_cat_name_to_iter_loader[bg_cat_name] = iter(loader)

        self.obj_scale = 0.5
        self.object_center_pos = (
            self.target_image_size // 2,
            self.target_image_size // 2,
        )

    def _gen_composition(self):

        self.x, self.y, self.a = [], [], []
        for factor_composition in self.ratio_dict.keys():
            idx_obj, idx_bg = factor_composition
            num_img = int(self.ratio_dict[factor_composition] * self.num_img_split)

            obj_dict = self.obj_dict_list[idx_obj]
            bg_dict = self.bg_dict_list[idx_bg]
            obj_name = obj_dict["name"]
            bg_name = bg_dict["name"]

            obj_loader = self.obj_cat_name_to_iter_loader[obj_name]
            bg_loader = self.bg_cat_name_to_iter_loader[bg_name]

            for _ in tqdm(range(num_img), leave=False, dynamic_ncols=True):
                obj_data_dict = next(obj_loader)
                image, mask = obj_data_dict["image"], obj_data_dict["mask"]
                image.squeeze_(0)
                mask.squeeze_(0)

                cropped_mask, cropped_image = crop_mask_and_img(mask, image)

                (rescaled_mask, rescaled_img,) = rescale_cropped_mask_and_img(
                    cropped_mask=cropped_mask,
                    cropped_img=cropped_image,
                    scale=self.obj_scale,
                    target_side_size=self.target_image_size,
                    object_center_pos=self.object_center_pos,
                )

                bg_image = next(bg_loader)
                bg_image.squeeze_(0)

                self.x.append(rescaled_mask * rescaled_img + (1 - rescaled_mask) * bg_image)
                self.y.append(idx_obj)
                self.a.append(idx_bg)
        self.y, self.a = torch.tensor(self.y).to(torch.long), torch.tensor(self.a).to(torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return idx, self.x[idx], self.y[idx], self.a[idx]


def get_urbancars_syn(param, seed, num_samples=None):
    trainset = UrbanCarsSyn(param=param, split='train', seed=seed, num_samples=num_samples)
    valset = UrbanCarsSyn(param=[0.5, 0.5, 0.5], split='val', seed=seed)
    testset = UrbanCarsSyn(param=[0.5, 0.5, 0.5], split='test', seed=seed)
    return trainset, valset, testset
