import torch
import torch.nn.functional as F
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from torch_ema import ExponentialMovingAverage
from utils.real_esrgan_config import opt
import random
import numpy as np


class RealESRGANModel:
    def __init__(self, Generator, Discriminator, dataloader, optim_g, optim_d, l1_loss, percep_loss, gan_loss):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)  # JPERG 압축
        self.usm_sharper = USMSharp().to(self.device)  # USM 이미지로 변경
        self.opt = opt
        self.netG = Generator.to(self.device)  # 생성자
        self.netD = Discriminator.to(self.device)  # 판별자
        self.optim_g = optim_g  # 생성자용 옵티마이저
        self.optim_d = optim_d  # 판별자용 옵티마이저
        self.dataloader = dataloader  # 데이터 로더

        self.l1_loss = l1_loss
        self.percep_loss = percep_loss
        self.gan_loss = gan_loss

        self.g_ema = ExponentialMovingAverage(self.netG.parameters(), decay=0.9999)  # exponential moving average
        self.d_ema = ExponentialMovingAverage(self.netD.parameters(), decay=0.9999)  # exponential moving average

    @torch.no_grad()
    def feed_data(self, data):
        img_gt = data['img'].to(self.device)
        gt_usm = self.usm_sharper(img_gt)

        kernel1 = data['kernel1'].to(self.device)
        kernel2 = data['kernel2'].to(self.device)
        sinc_kernel = data['sinc_kernel'].to(self.device)

        h, w = img_gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(gt_usm, kernel1)

        # random resize
        updown_type = ['up', 'down', 'keep']
        random_type = random.choices(updown_type, self.opt["first_deg"]["resize_prob"])

        if random_type == "up":
            scale_factor = np.random.uniform(1, 1.5)
        elif random_type == "down":
            scale_factor = np.random.uniform(0.5, 1)
        else:
            scale_factor = 1

        choice_mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale_factor, mode=choice_mode)

        # Noise
        if np.random.uniform() < 0.5:
            out = random_add_gaussian_noise_pt(out, sigma_range=self.opt["first_deg"]["noise_sigma_range"], gray_prob=self.opt["gray_prob"], clip=True, rounds=False)
        else:
            out = random_add_poisson_noise_pt(out, scale_range=self.opt["first_deg"]["poisson_noise_scale"], gray_prob=self.opt["gray_prob"], clip=True, rounds=False)

        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['quality_factor'])
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.opt["second_blur_deg_skip"]:
            out = filter2D(out, kernel2)

        # random resize
        updown_type = ['up', 'down', 'keep']
        random_type = random.choices(updown_type, self.opt["second_deg"]["resize_prob"])
        if random_type == "up":
            scale_factor = np.random.uniform(1, 1.5)
        elif random_type == "down":
            scale_factor = np.random.uniform(0.5, 1)
        else:
            scale_factor = 1

        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(int(h / 4 * scale_factor), int(w / 4 * scale_factor)), mode=mode)

        # add noise
        if np.random.uniform() < 0.5:
            out = random_add_gaussian_noise_pt(out, sigma_range=self.opt["second_deg"]["noise_sigma_range"],
                                               gray_prob=self.opt["gray_prob"], clip=True, rounds=False)
        else:
            out = random_add_poisson_noise_pt(out, scale_range=self.opt["second_deg"]["poisson_noise_scale"],
                                              gray_prob=self.opt["gray_prob"], clip=True, rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(h // self.opt['scale'], w // self.opt['scale']), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['quality_factor'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['quality_factor'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(h // self.opt['scale'], w // self.opt['scale']), mode=mode)
            out = filter2D(out, sinc_kernel)

        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255

        # Random Crop
        (gt, gt_usm), lq = paired_random_crop([img_gt, gt_usm], lq, 224, self.opt['scale'])
        #gt_usm = self.usm_sharper(gt)
        lq = lq.contiguous()
        gt_usm = self.usm_sharper(gt)
        return lq, gt_usm

    def training(self, epoch, index, data):
        self.netG.train()
        self.netD.train()

        lq, gt = self.feed_data(data)
        ################### Generator ################
        for p in self.netD.parameters():
            p.requites_grad = False
        self.optim_g.zero_grad()

        fake_img = self.netG(lq)

        fake_out = self.netD(fake_img)

        l1_loss = self.l1_loss(fake_img, gt)

        percep_loss, style_loss = self.percep_loss(fake_img, gt)
        gan_loss = self.gan_loss(fake_out, True, is_disc=False)
        total_loss = l1_loss + percep_loss + style_loss + gan_loss

        total_loss.backward()
        self.optim_g.step()

        self.g_ema.update()

        ################### Discriminator ################

        for p in self.netD.parameters():
            p.requites_grad = True
        self.optim_d.zero_grad()

        real_d_pred = self.netD(gt)
        l_d_real = self.gan_loss(real_d_pred, True, is_disc=True)

        fake_d_pred = self.netD(fake_img.detach().clone())
        l_d_fake = self.gan_loss(fake_d_pred, False, is_disc=True)

        l_d_loss = l_d_real + l_d_fake
        l_d_loss.backward()
        self.optim_d.step()
        self.d_ema.update()

        print(f"RealESRGAN_Model Training => [Epoch : {epoch}], [Batch {index}/{len(self.dataloader)}], [g loss: {total_loss}], [l loss: {l_d_loss}] l1:{l1_loss}, percep:{percep_loss},style:{style_loss}, gan:{gan_loss}")