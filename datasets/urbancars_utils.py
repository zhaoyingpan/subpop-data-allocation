
import os
import json
import pathlib
import torch

from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset

from typing import Callable, Optional, Any, Tuple


def crop_mask_and_img(mask: torch.Tensor, image: torch.Tensor):
    """
    Given a mask predicted by a saliency model,
    return the enclosed mask, i.e., cropping the original mask
    by based on the bounding box of the mask.

    Args:
        mask (torch.Tensor): shape: [1, H, W]
        image (torch.Tensor): shape: [3, H, W]

    Returns:
        cropped_mask: shape: [1, h, w], where h <= H, w <= W
        cropped_img: shape: [3, h, w], where h <= H, w <= W
    """
    assert len(mask.shape) == 3
    assert len(image.shape) == 3
    assert mask.shape[1:] == image.shape[1:]

    height, width = mask.shape[1:]
    mask = (mask > 0.5).long()  # to binary mask
    indices = mask[0].nonzero()
    height_indices = indices[:, 0]
    width_indices = indices[:, 1]
    top = height_indices.min()
    bottom = height_indices.max()
    left = width_indices.min()
    right = width_indices.max()

    cropped_mask = mask[:, top : min(bottom + 1, height), left : min(right + 1, width)]
    cropped_img = image[:, top : min(bottom + 1, height), left : min(right + 1, width)]
    return cropped_mask, cropped_img



def rescale_cropped_mask_and_img(
    cropped_mask: torch.Tensor,
    cropped_img: torch.Tensor,
    scale: float,
    target_side_size: int = 224,
    object_center_pos=None,
):
    """
    _summary_

    Args:
        cropped_mask (torch.Tensor): cropped mask
        cropped_img (torch.Tensor): cropped image
        scale (float): ranging from 0 to 1
        target_side_size (int): output side size (S) of the rescaled mask (S, S)
    """
    assert len(cropped_mask.shape) == 3
    assert len(cropped_img.shape) == 3
    assert cropped_mask.shape[1:] == cropped_img.shape[1:]
    assert 0 < scale <= 1

    if object_center_pos is None:
        object_center_pos = (target_side_size // 2, target_side_size // 2)

    longer_side = max(cropped_mask.shape[1:])
    scale_factor = scale * target_side_size / longer_side
    cropped_mask = cropped_mask.unsqueeze(0).float()
    rescaled_mask = F.interpolate(
        cropped_mask, scale_factor=scale_factor, mode="nearest-exact"
    ).squeeze(0)

    cropped_img = cropped_img.unsqueeze(0)
    rescaled_img = F.interpolate(
        cropped_img,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)

    assert rescaled_mask.shape[1:] == rescaled_img.shape[1:]

    rescaled_height, rescaled_width = rescaled_mask.shape[1:]

    target_mask = torch.zeros(
        (1, target_side_size, target_side_size),
        dtype=torch.long,
        device=cropped_mask.device,
    )

    top = max(0, object_center_pos[0] - rescaled_height // 2)
    left = max(0, object_center_pos[1] - rescaled_width // 2)

    target_mask[
        :,
        top : top + rescaled_height,
        left : left + rescaled_width,
    ] = rescaled_mask

    target_img = torch.zeros(
        (3, target_side_size, target_side_size),
        dtype=torch.float,
        device=cropped_img.device,
    )

    target_img[
        :,
        top : top + rescaled_height,
        left : left + rescaled_width,
    ] = rescaled_img

    return target_mask, target_img


class Places(Dataset):
    base_folder = "places"
    filter_objs_metadata_folder = "./data/Whac-A-Mole/create_datasets/urbancars/metadata/places/filter_objs"

    def __init__(self, root, category_name, transform=None):
        super().__init__()
        self.root = root

        rel_fpath_list = []
        for split in ["train", "val"]:
            split_dir = os.path.join(root, self.base_folder, split, category_name)
            assert os.path.exists(split_dir)
            metadata_fpath = os.path.join(
                self.filter_objs_metadata_folder, split, f"{category_name}.txt"
            )
            assert os.path.exists(metadata_fpath), f"{metadata_fpath} does not exist"
            with open(metadata_fpath, "r") as f:
                filename_list = [fname.strip() for fname in f.readlines()]
            filename_list.sort()
            split_rel_fpath_list = [
                os.path.join(split, category_name, filename)
                for filename in filename_list
            ]
            rel_fpath_list += split_rel_fpath_list
        self.rel_fpath_list = rel_fpath_list
        self.transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                      std = [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.rel_fpath_list)

    def __getitem__(self, index):
        rel_fpath = self.rel_fpath_list[index]
        img_path = os.path.join(self.root, self.base_folder, rel_fpath)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img



class StanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        """

    mask_base_folder = "stanford_cars/mask"
    filter_objs_metadata_base_folder = (
        "./data/Whac-A-Mole/create_datasets/urbancars/metadata/stanford_cars"
    )

    def __init__(
        self,
        root: str,
        urban_or_country: int,
        split: str = "train",
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        return_bbox: bool = False,
        return_image_path: bool = False,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError(
                "Scipy is not found. This dataset needs to have scipy"
                " installed: pip install scipy"
            )

        super().__init__(
            root,
            transform=transform,
        )
        assert urban_or_country in [0, 1]
        self.mask_transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                ])
        
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
            ])
        self._split = split
        self._base_folder = pathlib.Path("./data/Whac-A-Mole/data/stanford_cars")

        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = (
                self._base_folder / "cars_test_annos_withlabels.mat"
            )
            self._images_base_path = self._base_folder / "cars_test"

        self._mask_split_path = os.path.join(
            root, self.mask_base_folder, f"cars_{split}"
        )

        with open(
            os.path.join(self.filter_objs_metadata_base_folder, f"{split}.txt")
        ) as f:
            filtered_filename_list = f.read().splitlines()

        filtered_filename_set = set(filtered_filename_list)

        with open(
            os.path.join(
                self.filter_objs_metadata_base_folder, "body_type.json"
            )
        ) as f:
            urban_or_country_list = json.load(f)

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"]
                - 1,  # Original target mapping  starts from 1, hence -1
                [
                    annotation["bbox_x1"],
                    annotation["bbox_y1"],
                    annotation["bbox_x2"],
                    annotation["bbox_y2"],
                ],
            )
            for annotation in sio.loadmat(
                self._annotations_mat_path, squeeze_me=True
            )["annotations"]
            if annotation["fname"] in filtered_filename_set
            and urban_or_country_list[annotation["class"] - 1]
            == urban_or_country
        ]

        self.classes = sio.loadmat(
            str(devkit / "cars_meta.mat"), squeeze_me=True
        )["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.return_bbox = return_bbox
        self.return_image_path = return_image_path

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target, bbox = self._samples[idx]

        mask_fname = (
            os.path.splitext(os.path.basename(image_path))[0] + "_mask.png"
        )
        mask_path = os.path.join(self._mask_split_path, mask_fname)

        pil_image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = mask.resize(pil_image.size)

        if self.transform is not None:
            pil_image = self.transform(pil_image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        data_dict = {
            "image": pil_image,
            "target": target,
            "mask": mask,
        }

        if self.return_bbox:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            new_bbox = [x1, y1, w, h]
            new_bbox = torch.tensor(new_bbox, dtype=torch.long)
            data_dict["bbox"] = new_bbox

        if self.return_image_path:
            data_dict["image_path"] = image_path

        return data_dict
