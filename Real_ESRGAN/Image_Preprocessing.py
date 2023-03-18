import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from utils import util_option
from utils.real_esrgan_config import opt
from PIL import Image
import cv2
import random
from basicsr.data.degradations import circular_lowpass_kernel
from basicsr.utils import img2tensor


class RealEsrganDataset(Dataset):
    def __init__(self, file_dir):
        super().__init__()
        self.img_file = glob.glob(os.path.join(file_dir, '*.png'))
        self.pulse_tensor = torch.zeros(21, 21).float() #sinc kernel size 21 x 21
        self.pulse_tensor[10, 10] = 1
        self.opt = opt

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, idx):
        img_gt = Image.open(self.img_file[idx])
        img_gt = np.array(img_gt)
        h, w = img_gt.shape[:2]
        crop_pad_size = 1024

        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        kernel_list = ['gaussian_blur', 'generalized_gaussian_blur', 'plateau_shaped']
        kernel_sizes = self.opt["blur_kernel_size"]
        kernel_size = random.choice(kernel_sizes)
        pad_size = (21 - kernel_size) // 2

        ################# First Degradation ######################
        if np.random.uniform() < self.opt["sinc_prob"]:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)

            kernel1 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel1 = util_option.random_gaussian_kernel(kernel_list, kernel_size, self.opt["first_deg"],self.opt["noise_kernel_prob"], self.opt["beta_g"], self.opt["beta_p"])

        kernel1 = np.pad(kernel1, ((pad_size, pad_size), (pad_size, pad_size)))
        ##########################################################

        ################# Second Degradation ######################
        if np.random.uniform() < self.opt["sinc_prob"]:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)

            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = util_option.random_gaussian_kernel(kernel_list, kernel_size, self.opt["second_deg"],
                                               self.opt["noise_kernel_prob"], self.opt["beta_g"], self.opt["beta_p"])

        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        #################Final sinc kernel#############
        if np.random.uniform() < self.opt["final_sinc_prob"]:
            kernel_size = random.choice(kernel_sizes)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        kernel1 = torch.FloatTensor(kernel1)
        kernel2 = torch.FloatTensor(kernel2)

        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0]

        dataset = {'img': img_gt, 'kernel1': kernel1, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}

        return dataset




