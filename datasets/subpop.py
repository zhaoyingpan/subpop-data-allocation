import os
import pandas as pd
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import AutoTokenizer
from itertools import product

from utils import param2ratio, ratio2param, reset_seed


ImageFile.LOAD_TRUNCATED_IMAGES = True

class SubpopDataset:
    SPLITS = {
        'tr': 0,
        'va': 1,
        'te': 2
    }

    def __init__(self, config, root, split, metadata, transform, param):
        reset_seed(self.seed)
        df = pd.read_csv(metadata)
        df = df[df["split"] == (self.SPLITS[split])]
        self._get_original_ratios(df)

        if isinstance(param, list):
            if len(param) == 3:
                self.param = param
                self.ratios = param2ratio(self.param)
            elif len(param) == 1:
                self.param = config.data.params_list[param[0]]
                self.ratios = param2ratio(self.param)
            elif len(param) == len(self.unique_y) * len(self.unique_a):
                self.ratios = param
                self.param = None

        else:
            if param == "original":
                self.ratios = self.original_ratios
                self.param = ratio2param(self.ratios)

        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.y = df["y"].tolist()
        if config.data.dataset == 'CivilComments' and split != 'tr':
            self.a = [1 if i == 0 else 0 for i in df["a"].tolist()]
        else:
            self.a = df["a"].tolist()
        self.transform_ = transform
        self._count_groups()
    
    def _get_original_ratios(self, df):
        self.unique_y = sorted(df['y'].unique())
        self.unique_a = sorted(df['a'].unique())
        self.group_order = list(product(self.unique_y, self.unique_a))[::-1]
        group_sizes = df.groupby(['y', 'a']).size().to_dict()
        self.total_n = len(df)
        self.original_ratios = [group_sizes.get((y, a), 0) / self.total_n for (y, a) in self.group_order]

    def _count_groups(self):
        self.num_attributes = len(set(self.a))
        self.num_labels = len(set(self.y))
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



class Waterbirds(SubpopDataset):
    def __init__(self, config, split, param, seed):
        data_path = config.data.data_root
        self.seed = seed
        root = os.path.join(data_path, "waterbirds", "waterbird_complete95_forest2water2")
        metadata = os.path.join(data_path, "waterbirds", "metadata_waterbirds.csv")
        transform = transforms.Compose([
            transforms.Resize((int(224 * (256 / 224)), int(224 * (256 / 224)),)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.data_type = "images"
        super().__init__(config, root, split, metadata, transform, param)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))



class CelebA(SubpopDataset):
    def __init__(self, config, split, param, seed):
        data_path = config.data.data_root
        self.seed = seed
        root = os.path.join(data_path, "celeba", "img_align_celeba")
        if isinstance(param, list) and len(param) == 1:
            metadata = config.data.metadata_list[param[0]]
            transform = transforms.Compose([
                transforms.CenterCrop(178),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            metadata = os.path.join(data_path, "celeba", "metadata_celeba.csv")
            transform = transforms.Compose([
                transforms.CenterCrop(178),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        self.data_type = "images"
        super().__init__(config, root, split, metadata, transform, param)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))




class CivilComments(SubpopDataset):
    def __init__(self, config, split, param, seed,
                 granularity="coarse"):
        data_path = config.data.data_root
        self.seed = seed
        text = pd.read_csv(os.path.join(
            data_path, "civilcomments/civilcomments_{}.csv".format(granularity))
        )

        if isinstance(param, list) and len(param) == 1:
            metadata = config.data.metadata_list[param[0]]
        else:
            metadata = os.path.join(data_path, "civilcomments", "metadata_civilcomments_{}.csv".format(granularity))

        self.text_array = text["comment_text"].fillna("").tolist()
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





