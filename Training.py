import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Image_Preprocessing import RealEsrganDataset
from Real_ESRNet import RealESRGANNet
from Real_ESRModel import RealESRGANModel
from net_D import Discriminator
from net_G import Generator
from loss.loss_storage import L1Loss, Perceptual_Loss, GANLoss
from torch_ema import ExponentialMovingAverage


class real_esrgan_training:
    def __init__(self):
        dataset = RealEsrganDataset("../dataset/DIV2K_train_HR")
        self.dataloader = DataLoader(dataset, batch_size=36, shuffle=True)

        self.netG = Generator(in_chan=3, out_chan=3, num_feat=64, num_block=23, num_grow_ch=32, scale_factor=1)
        self.netD = Discriminator()

        self.criterion_loss = L1Loss()
        self.criterion_percep = Perceptual_Loss()
        self.criterion_gan = GANLoss()

        self.model_optim_D = torch.optim.Adam(self.netD.parameters(), lr=0.0001)
        self.model_optim_G = torch.optim.Adam(self.netG.parameters(), lr=0.0001)

        self.realmodel = RealESRGANModel(self.netG, self.netD, self.dataloader, self.model_optim_G, self.model_optim_D, self.criterion_loss, self.criterion_percep, self.criterion_gan)

        # ESRGAN Finetune
        self.netG.load_state_dict(torch.load("../pretrained/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth"))

        self.net_optim_D = torch.optim.Adam(self.netD.parameters(), lr=0.0002)
        self.net_optim_G = torch.optim.Adam(self.netG.parameters(), lr=0.0002)

        self.realnet = RealESRGANNet()

        self.iteration = 1000
        self.g_ema = ExponentialMovingAverage(self.netG.parameters(), decay=0.9998)
        self.d_ema = ExponentialMovingAverage(self.netD.parameters(), decay=0.9998)

    def main(self):
        for epoch in range(self.iteration):
            for index, data in enumerate(self.dataloader):
                if epoch < 400:
                  self.realmodel.training(epoch, index, data)

                lq, gt = self.realnet.feed_data(data)

                ################### Generator ################
                for p in self.netD.parameters():
                    p.requires_grad = False
                self.net_optim_G.zero_grad()

                fake_img = self.netG(lq)
                fake_out = self.netD(fake_img)

                l1_loss1 = self.l1_loss(fake_img, gt)

                l1_loss1.backward()
                self.net_optim_G.step()

                self.g_ema.update()

                ################### Discriminator ################

                for p in self.netD.parameters():
                    p.requires_grad = True
                self.net_optim_D.zero_grad()

                real_d_pred = self.netD(gt)
                l_d_real = self.gan_loss(real_d_pred, True, is_disc=True)
                l_d_real.backward()

                fake_d_pred = self.netD(fake_img.detach().clone())
                l_d_fake = self.gan_loss(fake_d_pred, False, is_disc=True)
                l_d_fake.backward()

                l_d_loss = l_d_real + l_d_fake
                self.net_optim_D.step()

                self.d_ema.update()


if __name__ == "__main__":
    training = real_esrgan_training()
    training.main()