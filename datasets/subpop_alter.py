import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import AutoTokenizer
from itertools import product

from utils import ratio2param, get_subgroup_masks, reset_seed


ImageFile.LOAD_TRUNCATED_IMAGES = True

class SubsetDataset:
    SPLITS = {
        'tr': 0,
        'va': 1,
        'te': 2
    }

    def __init__(self, config, root, split, metadata, transform, param):
        reset_seed(self.seed)
        self.split = split
        df = pd.read_csv(metadata)

        if split != 'tr':
            df = df[df["split"] == (self.SPLITS[split])]
            self._get_original_ratios(df)
            self.ratios = self.original_ratios
            self.param = ratio2param(self.ratios)
        else:
            if isinstance(param, list) and len(param) == 4:
                self.new_N = np.array(param)
                self.ratios = self.new_N / self.new_N.sum()
                self.param = ratio2param(self.ratios)
                df = df[df['split'] == (self.SPLITS[split])]
                df = self.get_df(df)
            elif param == "original":
                df = df[(df['split'] == (self.SPLITS[split])) & (df['base'] == 1)]
                masks = get_subgroup_masks(df['y'].to_numpy(), df['a'].to_numpy())
                group_sizes = np.array([m.sum() for m in masks])
                self.ratios = group_sizes / group_sizes.sum()
                self.param = ratio2param(self.ratios)

        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.y = df["y"].tolist()
        if config.data.dataset == 'CivilComments_subset' and split != 'tr':
            self.a = [1 if i == 0 else 0 for i in df["a"].tolist()]
        else:
            self.a = df["a"].tolist()
        self.transform_ = transform
        self._count_groups()

    def get_df(self, df):
        masks = get_subgroup_masks(df["y"], df["a"])
        sampled_dfs = []
        for mask, num in zip(masks, self.new_N):
            num = int(num)
            group_df = df[mask]
            # Separate base==1 and base==0 within this group
            base1_df = group_df[group_df['base'] == 1]
            base0_df = group_df[group_df['base'] == 0]
            n_base1 = len(base1_df)
            n_base0 = len(base0_df)
            
            if num <= n_base1:
                # Target is smaller or equal to base==1: only sample from base==1
                sampled_group = base1_df.sample(n=num, random_state=self.seed)
            elif num <= n_base1 + n_base0:
                # Target is larger than base==1 but within base==1 + base==0
                # Take all base==1, then sample from base==0
                sampled_group = base1_df.copy()
                remaining = num - n_base1
                if remaining > 0 and n_base0 > 0:
                    additional_samples = base0_df.sample(n=remaining, random_state=self.seed)
                    sampled_group = pd.concat([sampled_group, additional_samples], ignore_index=True)
            else:
                # Target is larger than base==1 + base==0
                # Take all base==1 and all base==0, then sample with replacement
                sampled_group = pd.concat([base1_df, base0_df], ignore_index=True)
                remaining = num - n_base1 - n_base0
                if remaining > 0:
                    # Sample with replacement from the entire group
                    additional_samples = group_df.sample(n=remaining, replace=True, random_state=self.seed)
                    sampled_group = pd.concat([sampled_group, additional_samples], ignore_index=True)
            
            # Set base column to 1 for all sampled data
            sampled_group = sampled_group.copy()
            sampled_group['base'] = 1
            sampled_dfs.append(sampled_group)

        sampled_train_df = pd.concat(sampled_dfs, ignore_index=True)
        return sampled_train_df


    def _get_original_ratios(self, df):
        self.unique_y = sorted(df['y'].unique())
        self.unique_a = sorted(df['a'].unique())
        self.group_order = list(product(self.unique_y, self.unique_a))[::-1]
        group_sizes = df.groupby(['y', 'a']).size().to_dict()
        self.total_n = len(df)
        self.original_ratios = [group_sizes.get((y, a), 0) / self.total_n for (y, a) in self.group_order]

    def _count_groups(self):
        self.num_attributes = 2
        self.num_labels = 2
        self.group_sizes = [0] * self.num_attributes * self.num_labels

        for i in self.idx:
            self.group_sizes[self.num_attributes * self.y[i] + self.a[i]] += 1



    def __getitem__(self, index):
        i = self.idx[index]
        x = self.transform(self.x[i])
        y = torch.tensor(self.y[i], dtype=torch.long)
        a = torch.tensor(self.a[i], dtype=torch.long)
        return i, x, y, a

    def __len__(self):
        return len(self.idx)


class CelebA_subset(SubsetDataset):
    def __init__(self, config, split, param, seed):
        data_path = config.data.data_root
        root = os.path.join(data_path, "celeba", "img_align_celeba")
        metadata_name = config.data.get("metadata")
        if metadata_name is None:
            metadata = os.path.join(data_path, "celeba", "metadata_celeba_full.csv")
        else:
            metadata = os.path.join(data_path, "celeba", f"{metadata_name}.csv")

        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.data_type = "images"
        self.seed = seed
        super().__init__(config, root, split, metadata, transform, param)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))



class CivilComments_subset(SubsetDataset):
    def __init__(self, config, split, param, seed,
                 granularity="coarse"):
        data_path = config.data.data_root
        text = pd.read_csv(os.path.join(
            data_path, "civilcomments/civilcomments_{}.csv".format(granularity))
        )
        metadata_name = config.data.get("metadata")
        if metadata_name is None:
            metadata = os.path.join(data_path, "civilcomments", "metadata_civilcomments_coarse_full.csv")
        else:
            metadata = os.path.join(data_path, "civilcomments", f"{metadata_name}.csv")

        self.text_array = text["comment_text"].fillna("").tolist()
        self.seed = seed
        model_name_map = {
            "bert": "bert-base-uncased",
            "roberta": "roberta-base",
            "deberta": "microsoft/deberta-v3-base"
        }
        if config.train.model_type in ["bert", "roberta", "deberta"]:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_map[config.train.model_type])
        else:
            raise ValueError(f'Unsupported model type: {config.train.model_type}')
        self.data_type = "text"
        super().__init__(config, "", split, metadata, self.transform, param)

    def transform(self, i):
        text = self.text_array[int(i)]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",
        )

        if len(tokens) == 3:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"]
                ), dim=2
                ), dim=0
            )
        else:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"]
                ), dim=2
                ), dim=0
            )




