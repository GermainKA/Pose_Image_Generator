

import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 



class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
    



class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        self.netG = GenNNSkeToImage()
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/Dance/DanceGenGAN.pth'
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename)


    def train(self, n_epochs=20):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG.to(device)
        self.netD.to(device)

        # Initialisation des poids
        self.netG.apply(init_weights)
        self.netD.apply(init_weights)

        # Optimiseurs
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
        # Critère de perte
        criterion = nn.BCELoss()

        for epoch in range(n_epochs):
            for i, (skeletons, real_images) in enumerate(self.dataloader):
                batch_size = skeletons.size(0)
                real_images = real_images.to(device)
                skeletons = skeletons.to(device)

                # --- Entraîner le discriminateur ---
                # Réelles
                self.netD.zero_grad()
                labels = torch.full((batch_size,), self.real_label, dtype=torch.float, device=device)
                output = self.netD(real_images).view(-1)
                lossD_real = criterion(output, labels)
                lossD_real.backward()

                # Générées
                fake_images = self.netG(skeletons)
                labels.fill_(self.fake_label)
                output = self.netD(fake_images.detach()).view(-1)
                lossD_fake = criterion(output, labels)
                lossD_fake.backward()

                lossD = lossD_real + lossD_fake
                optimizerD.step()

                # --- Entraîner le générateur ---
                self.netG.zero_grad()
                labels.fill_(self.real_label)  # On veut que le discriminateur classe les fausses images comme réelles
                output = self.netD(fake_images).view(-1)
                lossG = criterion(output, labels)
                lossG.backward()
                optimizerG.step()

                # Affichage périodique
                if i % 10 == 0:
                    print(f"[{epoch+1}/{n_epochs}][{i}/{len(self.dataloader)}] "
                        f"Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")

            # Sauvegarde a chaque epoch
            torch.save(self.netG, self.filename)
            print(f"Modèle sauvegardé à l'époque {epoch+1}")




    def generate(self, ske): 
        """ generator of image from skeleton """
        ske_t = torch.from_numpy( ske.__array__(reduced=True).flatten() )
        ske_t = ske_t.to(torch.float32)
        ske_t = ske_t.reshape(1,Skeleton.reduced_dim,1,1) # ske.reshape(1,Skeleton.full_dim,1,1)
        normalized_output = self.netG(ske_t)
        res = self.dataset.tensor2image(normalized_output[0])
        return res




if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    #if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(20) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

