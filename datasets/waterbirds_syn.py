import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

from utils import param2ratio, reset_seed

class water_land_bird(Dataset):
    def __init__(self, data_root, seed=42, tensor_image = False, image_size = 224, split = "tr"):
        self.data_root = data_root
        self.tensor_image = tensor_image
        self.image_size = image_size
        self.max_object_size = 0.15
        self.max_rotation_angle = 30
        self.split = split
        self.seed = seed
        
        if self.tensor_image:
            self.image_transform = transforms.Compose(
                [transforms.Resize((self.image_size, self.image_size)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                      std = [0.229, 0.224, 0.225])
                ])
            self.mask_transform = transforms.Compose(
                [transforms.Resize((self.image_size, self.image_size)),
                 transforms.ToTensor(),
                ])
        else:
            self.image_transform = None
            
        # train test split: a binary list of the is_trainset of 11788 images
        with open(os.path.join(self.data_root, './CUB_200_2011/train_test_split.txt'), 'r')  as f:
            self.train_test_split = f.readlines()
        self.train_test_split = np.array([int(att.split()[1]) for att in self.train_test_split])
        
        if self.split == "tr":
            self.indices = np.where(self.train_test_split == 1)[0]
        else:
            indices = np.where(self.train_test_split == 0)[0]
            if self.split == "va":
                self.indices = indices[: indices.shape[0] // 2]
            elif self.split == "te":
                self.indices = indices[indices.shape[0] // 2 :]
        
        # image list: a list of 11788 images
        with open(os.path.join(self.data_root, './CUB_200_2011/images.txt'), 'r')  as f:
            self.image_list = f.readlines()

        self.image_list = np.array([att.split()[1] for att in self.image_list])
        self.image_list = self.image_list[self.indices]

        # class list: a list of 200 classes
        with open(os.path.join(self.data_root, './CUB_200_2011/classes.txt'), 'r')  as f:
            self.class_list = f.readlines()
        self.class_list = [att.split()[1] for att in self.class_list]
        
        # image to label: a list of the classes of the 11788 images
        with open(os.path.join(self.data_root, './CUB_200_2011/image_class_labels.txt'), 'r')  as f:
            self.image2label = f.readlines()
        self.image2label = np.array([int(att.split()[1])-1 for att in self.image2label])
        self.image2label = self.image2label[self.indices]
               
        self.water_birds_list = [
            'Albatross', # Seabirds
            'Auklet',
            'Cormorant',
            'Frigatebird',
            'Fulmar',
            'Gull',
            'Jaeger',
            'Kittiwake',
            'Pelican',
            'Puffin',
            'Tern',
            'Gadwall', # Waterfowl
            'Grebe',
            'Mallard',
            'Merganser',
            'Guillemot',
            'Pacific_Loon'
        ]
        
        # get the label for the water/land birds
        self.unresampled_label = []
        for i in range(len(self.image_list)):
            is_waterbird = False
            for waterbird in self.water_birds_list:
                if waterbird in self.image_list[i]:
                    is_waterbird = True
                    break
            if is_waterbird:
                self.unresampled_label.append(1)
            else:
                self.unresampled_label.append(0)

        self.unresampled_label = np.array(self.unresampled_label)
        
        self.waterbird_resampled_indices = np.random.default_rng(seed=self.seed).choice(
            np.where(self.unresampled_label == 1)[0],
            size = ((self.unresampled_label == 0).sum() - (self.unresampled_label == 1).sum()))
        self.waterbird_indices = np.concatenate(
            [np.where((self.unresampled_label == 1))[0], self.waterbird_resampled_indices])  
        
        self.landbird_indices = np.where((self.unresampled_label == 0))[0]
        
        self.total_indices = np.concatenate([
            self.waterbird_indices, self.landbird_indices
        ])
        
    
    def resize(self, x, mask, ratio):
        if torch.is_tensor(ratio):
            ratio = ratio.item()
        pad_size = x.shape[-1] - int(x.shape[-1]*np.sqrt(ratio))
        new_x = F.interpolate(x[None], scale_factor = np.sqrt(ratio), mode = 'bilinear', align_corners = False)
        x = F.pad(new_x, (pad_size//2, pad_size-pad_size//2)*2,
                  mode = 'constant', value = 0).squeeze()
    
        new_mask = F.interpolate(mask[None], scale_factor = np.sqrt(ratio), mode = 'bilinear', align_corners = False)
        mask = F.pad(new_mask, (pad_size//2, pad_size-pad_size//2)*2,
                  mode = 'constant', value = 0).squeeze()
        return x, mask

    def rotate(self, x, mask):
        angle = (np.random.rand()*2-1)*self.max_rotation_angle
        x_rot = torchvision.transforms.functional.rotate(x[None], angle).squeeze()
        mask_rot = torchvision.transforms.functional.rotate(mask[None], angle).view(1,self.image_size,self.image_size)
        return x_rot, mask_rot
        
    def __len__(self):
        return len(self.total_indices)

    def __getitem__(self, idx):
        
        idx = self.total_indices[idx]
        image = Image.open(os.path.join(self.data_root, './CUB_200_2011/images', self.image_list[idx]))
        mask = Image.open(os.path.join(self.data_root, 'segmentations', 
                                       self.image_list[idx].split('.jpg')[0]+'.png'))
        image = image.convert('RGB')
        label = self.unresampled_label[idx]
        
        if self.tensor_image:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)[0][None]
            masked = image*mask
            mask_size = (mask > 0).float().mean()
            if mask_size > self.max_object_size:
                masked, mask = self.resize(masked, mask, self.max_object_size/mask_size)
            masked, mask = self.rotate(masked, mask)
            return masked, mask, label
        else:
            return image, mask, label


class places(Dataset):
    def __init__(self, data_root, image_size = 224, tensor_image = True, split = "tr"):
        self.data_root = data_root
        self.tensor_image = tensor_image
        self.split = split
        self.image_size = image_size
        
        if self.tensor_image:
            self.image_transform = transforms.Compose(
                [transforms.Resize((self.image_size, self.image_size)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                      std = [0.229, 0.224, 0.225])
                ])
        else:
            self.image_transform = None
            
        self.land = ['b/bamboo_forest', 'f/forest/broadleaf']
        self.water = ['l/lake/natural', 'o/ocean']
        
        self.image_list = []
        self.label = []
        
        # training images are the first 2500 of all four categories, totally 10000
        # testing images are the last 2500 of all four catetories, totally 10000
        if self.split == "tr":
            for land_class in self.land:
                self.image_list += [os.path.join(self.data_root, land_class, 
                                                 '{:0>8}.jpg'.format(i+1)) for i in range(2500)]
            for water_class in self.water:
                self.image_list += [os.path.join(self.data_root, water_class, 
                                                 '{:0>8}.jpg'.format(i+1)) for i in range(2500)]
        elif self.split == "va":
            for land_class in self.land:
                self.image_list += [os.path.join(self.data_root, land_class, 
                                                 '{:0>8}.jpg'.format(i+1)) for i in range(2500, 3750)]
            for water_class in self.water:
                self.image_list += [os.path.join(self.data_root, water_class, 
                                                 '{:0>8}.jpg'.format(i+1)) for i in range(2500, 3750)]
        elif self.split == "te":
            for land_class in self.land:
                self.image_list += [os.path.join(self.data_root, land_class, 
                                                 '{:0>8}.jpg'.format(i+1)) for i in range(3750, 5000)]
            for water_class in self.water:
                self.image_list += [os.path.join(self.data_root, water_class, 
                                                 '{:0>8}.jpg'.format(i+1)) for i in range(3750, 5000)]
        
        self.image_list = np.array(self.image_list)
        self.label = np.array([0]*(len(self.image_list) // 2)+[1]*(len(self.image_list) // 2))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
        image = Image.open(self.image_list[idx])
        
        image = image.convert('RGB')
        label = self.label[idx]
        return self.image_transform(image), label


class WaterBirds_biased(Dataset):
    def __init__(self, param, data_root, split, seed, num_samples=None, initial_setup=None):

        reset_seed(seed)
        self.split = split
        self.param = param
        self.seed = seed
        self.CUB_root=os.path.join(data_root, 'waterbirds_zeta')
        self.place_root=os.path.join(data_root, 'waterbirds_zeta/selected_places')
        self.image_size=224
        self.num_labels = 2
        self.num_attributes = 2
        self.initial_setup = initial_setup
        if isinstance(param, list):
            self.param = param
        else:
            if param == "original":
                self.param = [0.9, 0.9, 0.5]

        self.set_bird=water_land_bird(self.CUB_root,seed=self.seed,tensor_image=True,image_size=self.image_size,split=self.split)
        self.set_place=places(self.place_root,tensor_image=True,image_size=self.image_size,split=self.split)
        
        if split == 'train':
            self.N = len(self.set_bird) if num_samples is None else num_samples
        else:
            self.N = len(self.set_bird)

        self.generate_grps()
        self.group_sizes = self.grp_sizes[::-1]

        self.a = [1] * self.grp_sizes[0] + [0] * self.grp_sizes[1] + [1] * self.grp_sizes[2] + [0] * self.grp_sizes[3]
        self.a = torch.tensor(self.a).to(torch.long)

        self.y = [1] * self.grp_sizes[0] + [1] * self.grp_sizes[1] + [0] * self.grp_sizes[2] + [0] * self.grp_sizes[3]
        self.y = torch.tensor(self.y).to(torch.long)

        if self.initial_setup is not None:
            self.generate_indices2()
        else:
            self.generate_indices()

    def generate_grps(self):
        if self.split == 'tr':
            ratios = param2ratio(self.param)
        else:
            ratios = [0.25, 0.25, 0.25, 0.25]

        group_0_size, group_1_size, group_2_size = round(ratios[0] * self.N), round(ratios[1] * self.N), round(ratios[2] * self.N)
        group_3_size = self.N - (group_0_size + group_1_size + group_2_size)
        self.grp_sizes = [
            group_0_size,
            group_1_size,
            group_2_size,
            group_3_size,
        ]

    def resample_indices_from_initial(
        self,
        indices,
        target_size_initial,
        target_size_new,
    ):
        rng = np.random.default_rng(self.seed)
        indices = np.asarray(indices)

        # ---- Step 1: initial sample ----
        assert len(indices) >= target_size_initial
        perm = rng.permutation(indices)
        initial = perm[:target_size_initial]
        remaining = perm[target_size_initial:]

        # ---- Step 2: adjust to new target size ----
        if target_size_new == target_size_initial:
            return initial

        # Case A: shrink
        if target_size_new < target_size_initial:
            return rng.choice(initial, target_size_new, replace=False)

        # Case B: grow
        gap = target_size_new - target_size_initial
        if len(remaining) >= gap:
            extra = rng.choice(remaining, gap, replace=False)
        else:
            shortfall = gap - len(remaining)
            supplement = rng.choice(remaining, shortfall, replace=True)
            extra = np.concatenate([remaining, supplement])

        return np.concatenate([initial, extra])


    def generate_indices2(self):
        # (1,1) (1,0) (0,1) (0,0)
        initial_param = self.initial_setup[:3]
        initial_N = self.initial_setup[-1]
        initial_grp_sizes = np.round(initial_N * np.array(param2ratio(initial_param))).astype(int)
        waterbirds_indices = self.resample_indices_from_initial(np.arange(len(self.set_bird) // 2), target_size_initial=initial_grp_sizes[0]+initial_grp_sizes[1], target_size_new=self.grp_sizes[0]+self.grp_sizes[1])
        landbirds_indices = self.resample_indices_from_initial(np.arange(len(self.set_bird) // 2, len(self.set_bird)), target_size_initial=initial_grp_sizes[2]+initial_grp_sizes[3], target_size_new=self.grp_sizes[2]+self.grp_sizes[3])

        water_indices = self.resample_indices_from_initial(np.arange(len(self.set_place) // 2, len(self.set_place)), target_size_initial=initial_grp_sizes[0]+initial_grp_sizes[2], target_size_new=self.grp_sizes[0]+self.grp_sizes[2])
        land_indices = self.resample_indices_from_initial(np.arange(len(self.set_place) // 2), target_size_initial=initial_grp_sizes[1]+initial_grp_sizes[3], target_size_new=self.grp_sizes[1]+self.grp_sizes[3])

        self.bird_indices = np.concatenate([
            waterbirds_indices[:self.grp_sizes[0]],
            waterbirds_indices[self.grp_sizes[0]:],
            landbirds_indices[:self.grp_sizes[2]],
            landbirds_indices[self.grp_sizes[2]:],
        ])
        
        self.place_indices = np.concatenate([
            water_indices[:self.grp_sizes[0]],
            land_indices[:self.grp_sizes[1]],
            water_indices[self.grp_sizes[0]:],
            land_indices[self.grp_sizes[1]:],
        ])

        self.x = []
        for idx in tqdm(range(self.N)):
            bird, mask, _ = self.set_bird[self.bird_indices[idx]]
            place, _ = self.set_place[self.place_indices[idx]]
            self.x.append(place*(1-mask)+bird)

    def resample_indices(self, indices, target_size):
        if len(indices) >= target_size:
            return np.random.default_rng(self.seed).choice(indices, target_size, replace=False)
        else:
            return np.concatenate([indices, np.random.default_rng(self.seed).choice(indices, target_size - len(indices), replace=True)])

    def generate_indices(self):
        # (1,1) (1,0) (0,1) (0,0)
        waterbirds_indices = self.resample_indices(np.arange(self.N // 2), target_size=self.grp_sizes[0]+self.grp_sizes[1])
        landbirds_indices = self.resample_indices(np.arange(self.N // 2, self.N), target_size=self.grp_sizes[2]+self.grp_sizes[3])

        water_indices = self.resample_indices(np.arange(len(self.set_place) // 2, len(self.set_place)), target_size=self.grp_sizes[0]+self.grp_sizes[2])
        land_indices = self.resample_indices(np.arange(len(self.set_place) // 2), target_size=self.grp_sizes[1]+self.grp_sizes[3])

        self.bird_indices = np.concatenate([
            waterbirds_indices[:self.grp_sizes[0]],
            waterbirds_indices[self.grp_sizes[0]:],
            landbirds_indices[:self.grp_sizes[2]],
            landbirds_indices[self.grp_sizes[2]:],
        ])
        
        self.place_indices = np.concatenate([
            water_indices[:self.grp_sizes[0]],
            land_indices[:self.grp_sizes[1]],
            water_indices[self.grp_sizes[0]:],
            land_indices[self.grp_sizes[1]:],
        ])

        self.x = []
        for idx in tqdm(range(self.N)):
            bird, mask, _ = self.set_bird[self.bird_indices[idx]]
            place, _ = self.set_place[self.place_indices[idx]]
            self.x.append(place*(1-mask)+bird)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return idx, self.x[idx], self.y[idx], self.a[idx]



