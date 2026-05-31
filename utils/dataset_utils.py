import os
import random
import copy
from PIL import Image
import numpy as np
import re

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
from utils.image_utils import random_augmentation, crop_img


class TrainDataset(Dataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.lowlight_ids = []

        self.de_temp = 0
        self.de_type = self.args.de_type

        self.de_dict = {'lowlight': 0, 'derain': 1, 'dehaze': 2}

        self._init_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size)
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'lowlight' in self.de_type:
            self._init_lowlight_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()

        random.shuffle(self.de_type)
        print(f"[INFO] Training with degradations: {self.de_type}")

    def _init_lowlight_ids(self):
        ref_file = os.path.join(self.args.data_file_dir, "lowlight/lowlight_train.txt")
        self.lowlight_ids = [os.path.join(self.args.lowlight_dir, id_.strip()) for id_ in open(ref_file)]
        self.lowlight_counter = 0
        self.num_lowlight = len(self.lowlight_ids)
        print(f"[INFO] Lowlight training pairs loaded from TXT: {self.num_lowlight}")

    def _init_rs_ids(self):
        ref_file = os.path.join(self.args.data_file_dir, "rainy/rainTrain.txt")
        self.rs_ids = [os.path.join(self.args.derain_dir, id_.strip()) for id_ in open(ref_file)]
        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print(f"[INFO] Derain training images loaded from TXT: {self.num_rl}")

    def _init_hazy_ids(self):
        ref_file = os.path.join(self.args.data_file_dir, "hazy/hazy_outside.txt")
        self.hazy_ids = [os.path.join(self.args.dehaze_dir, id_.strip()) for id_ in open(ref_file)]
        self.hazy_counter = 0
        self.num_hazy = len(self.hazy_ids)
        print(f"[INFO] Hazy training images loaded from TXT: {self.num_hazy}")

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, max(0, H - self.args.patch_size))
        ind_W = random.randint(0, max(0, W - self.args.patch_size))

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_normal_name(self, low_name):
        parent_dir = os.path.dirname(os.path.dirname(low_name))
        target_dir = os.path.join(parent_dir, 'Normal')
        base_name = os.path.basename(low_name)

        candidates = [
            os.path.join(target_dir, base_name.replace('low', 'normal')),
            os.path.join(target_dir, base_name.replace('low', 'high')),
            os.path.join(target_dir, base_name)
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"[ERROR] Could not find ground truth for LowLight image: {low_name}")

    def _get_gt_name(self, rainy_name):
        dir_name = os.path.dirname(rainy_name).replace('input', 'target')
        file_name = os.path.basename(rainy_name).replace('rain-', 'norain-')
        candidate = os.path.join(dir_name, file_name)

        if os.path.exists(candidate):
            return candidate

        raise FileNotFoundError(f"[ERROR] Could not find ground truth for Rainy image: {rainy_name}")

    def _get_nonhazy_name(self, hazy_name):
        dir_name = os.path.dirname(hazy_name).replace('input', 'target')
        base_name = os.path.basename(hazy_name).split('_')[0]
        extension = os.path.splitext(hazy_name)[1]
        candidate = os.path.join(dir_name, base_name + extension)

        if os.path.exists(candidate):
            return candidate

        raise FileNotFoundError(f"[ERROR] Could not find ground truth for Hazy image: {hazy_name}")

    def __getitem__(self, _):
        current_de_type = self.de_type[self.de_temp]
        de_id = self.de_dict[current_de_type]

        if de_id == 0:
            degrad_path = self.lowlight_ids[self.lowlight_counter]
            clean_path = self._get_normal_name(degrad_path)
            self.lowlight_counter = (self.lowlight_counter + 1) % self.num_lowlight
            if self.lowlight_counter == 0:
                random.shuffle(self.lowlight_ids)

        elif de_id == 1:
            degrad_path = self.rs_ids[self.rl_counter]
            clean_path = self._get_gt_name(degrad_path)
            self.rl_counter = (self.rl_counter + 1) % self.num_rl
            if self.rl_counter == 0:
                random.shuffle(self.rs_ids)

        elif de_id == 2:
            degrad_path = self.hazy_ids[self.hazy_counter]
            clean_path = self._get_nonhazy_name(degrad_path)
            self.hazy_counter = (self.hazy_counter + 1) % self.num_hazy
            if self.hazy_counter == 0:
                random.shuffle(self.hazy_ids)

        # Load images
        degrad_img = crop_img(np.array(Image.open(degrad_path).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)
        clean_name = os.path.basename(clean_path).split('.')[0]

        # Apply data augmentation
        degrad_patch_1, clean_patch_1 = random_augmentation(*self._crop_patch(degrad_img, clean_img))
        degrad_patch_2, clean_patch_2 = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        # To Tensor
        clean_patch_1, clean_patch_2 = self.toTensor(clean_patch_1), self.toTensor(clean_patch_2)
        degrad_patch_1, degrad_patch_2 = self.toTensor(degrad_patch_1), self.toTensor(degrad_patch_2)

        # Advance to next degradation task
        self.de_temp = (self.de_temp + 1) % len(self.de_type)
        if self.de_temp == 0:
            random.shuffle(self.de_type)

        return [clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2

    def __len__(self):
        return 400 * len(self.args.de_type)


class LowLightTestDataset(Dataset):
    def __init__(self, args):
        super(LowLightTestDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_ids()
        self.toTensor = ToTensor()

    def _init_ids(self):
        low_dir = os.path.join(self.args.lowlight_dir, 'Low')
        self.degraded_ids = [os.path.join(low_dir, f) for f in sorted(os.listdir(low_dir)) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.num_img = len(self.degraded_ids)

    def _get_normal_name(self, low_name):
        parent_dir = os.path.dirname(os.path.dirname(low_name))
        target_dir = os.path.join(parent_dir, 'Normal')
        base_name = os.path.basename(low_name)

        candidates = [
            os.path.join(target_dir, base_name.replace('low', 'normal')),
            os.path.join(target_dir, base_name.replace('low', 'high')),
            os.path.join(target_dir, base_name)
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"[ERROR] Could not find test GT for {low_name}")

    def __getitem__(self, idx):
        degraded_path = self.degraded_ids[idx]
        clean_path = self._get_normal_name(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        name = os.path.basename(degraded_path)[:-4]
        return [name], self.toTensor(degraded_img), self.toTensor(clean_img)

    def __len__(self):
        return self.num_img


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain"):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()
        self.set_dataset(task)

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(os.path.join(self.args.derain_path, 'input'))
            self.ids += [os.path.join(self.args.derain_path, 'input', id_) for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(os.path.join(self.args.dehaze_path, 'input'))
            self.ids += [os.path.join(self.args.dehaze_path, 'input', id_) for id_ in name_list]
        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            # Derain pathing
            dir_name = os.path.dirname(degraded_name).replace('input', 'target')
            file_name = os.path.basename(degraded_name)

            candidates = [
                os.path.join(dir_name, file_name.replace('rain-', 'norain-')),
                os.path.join(dir_name, file_name.replace('rain-', '')),
                os.path.join(dir_name, file_name)
            ]

            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate

            existing = os.listdir(dir_name)[:5] if os.path.exists(dir_name) else "Target dir missing!"
            raise FileNotFoundError(f"[ERROR] Could not find test GT for {file_name}. Folder contains: {existing}")

        elif self.task_idx == 1:
            # Dehaze pathing
            dir_name = os.path.dirname(degraded_name).replace('input', 'target')
            base_name = os.path.basename(degraded_name).split('_')[0]
            full_name = os.path.basename(degraded_name)

            candidates = [
                os.path.join(dir_name, base_name + '.png'),
                os.path.join(dir_name, base_name + '.jpg'),
                os.path.join(dir_name, full_name),
                os.path.join(dir_name, os.path.splitext(full_name)[0] + '.png')
            ]

            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate

            existing = os.listdir(dir_name)[:5] if os.path.exists(dir_name) else "Target dir missing!"
            raise FileNotFoundError(f"[ERROR] Could not find test GT for {degraded_name}. Folder contains: {existing}")

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)
        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        name_list = os.listdir(root)
        self.degraded_ids += [os.path.join(root, id_) for id_ in name_list]
        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]
        degraded_img = self.toTensor(degraded_img)
        return [name], degraded_img

    def __len__(self):
        return self.num_img