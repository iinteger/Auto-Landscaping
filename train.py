# contributed by eriklindernoren github

import argparse
import os
from math import log10
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.utils import save_image
from models import *
from dataset import My_Dataset
from utils import is_image_file, load_img, save_img
from torch.nn import DataParallel
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import albumentations as A
import albumentations.pytorch
plt.axis("off")


# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=500, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='cosine', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50,
                    help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)


img_size = 256
print('===> Loading datasets')
root_path = "dataset/"


train_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.HorizontalFlip(),
    albumentations.pytorch.transforms.ToTensorV2()
])

# test_transform = A.Compose([
#     A.Resize(img_size, img_size),
#     albumentations.pytorch.transforms.ToTensorV2()
# ])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

train_set = My_Dataset(train_transform, train=True)
test_set = My_Dataset(test_transform, train=False)


training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=opt.test_batch_size,
                                 shuffle=False)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
print("cuda :", cuda)

print('===> Building models')
net_g = define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, norm="instance")
net_d = define_D(input_nc=opt.input_nc + opt.output_nc)
net_g = DataParallel(net_g)
net_d = DataParallel(net_d)

net_g.apply(weights_init_normal)
net_d.apply(weights_init_normal)

criterion_GAN = nn.MSELoss()
criterion_pixelwise = nn.L1Loss()

# loss weight if l1 loss
lambda_pixel = 50

# Calculate output of image discriminator (PatchGAN)
patch = (1, img_size // 2 ** 4, img_size // 2 ** 4)

if cuda:
    generator = net_g.cuda()
    discriminator = net_d.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

start = time.time()
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    for iteration, batch in enumerate(training_data_loader, 1):

        real_A = Variable(batch["A"].type(Tensor))  # edge
        real_B = Variable(batch["B"].type(Tensor))  # color

        # adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # generator train 1 start
        optimizer_g.zero_grad()

        # gan loss
        fake_B = net_g(real_A)  # 엣지를 받아서 가짜 컬러 이미지를 생성
        pred_fake = net_d(fake_B, real_A)  # 가짜 컬러 이미지와 진짜 엣지 이미지(condition)을 넣어서(concat) inference
        loss_GAN = criterion_GAN(pred_fake, valid)
        #pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)  # 생성한 가짜 컬러 이미지와 진짜 컬러 이미지의 l1 loss 계산

        # total loss
        loss_G = loss_GAN + loss_pixel * lambda_pixel  # 논문에서는 가중값을 100 사용했는데, 이후 10이 적당하다고 함. 실험 해봐야 할듯!

        loss_G.backward()
        optimizer_g.step()
        # generator train finish

        # discriminator train start
        optimizer_d.zero_grad()

        # real loss
        pred_real = net_d(real_B, real_A)  # 실제 컬러 이미지와 실제 엣지 이미지(condition)을 넣어서(concat) inference
        loss_real = criterion_GAN(pred_real, valid)  # 진짜를 추론한 결과와 True의 mse. 1(True)로 추론할 수록 error가 낮아짐

        # fake loss
        pred_fake = net_d(fake_B.detach(), real_A)  # 가짜 컬러 이미지와 실제 엣지 이미지를 넣어서 inference. fake_B는 G로부터 나온 Tensor이기 때문에 detach를 하지 않으면 역전파가 흘러들어감. 추가적인 공부 필요
        loss_fake = criterion_GAN(pred_fake, fake)  # 가짜를 추론한 결과와 False의 mse. 0(False)로 추론할 수록 error가 낮아짐

        # total loss
        loss_D = 0.5*(loss_real + loss_fake)

        loss_D.backward()
        optimizer_d.step()
        # discriminator train finish


        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_D.item(), loss_G.item()))

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)
    # test
    avg_psnr = 0
    for batch in testing_data_loader:
        real_A_test = Variable(batch["A"].type(Tensor))  # edge
        real_B_test = Variable(batch["B"].type(Tensor))  # edge

        prediction = net_g(real_A_test)
        mse = criterion_GAN(prediction, real_B_test)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    print("epoch "+str(epoch)+": "+str(time.time()-start))

    f = open("PSNR_list.txt", 'a')
    f.write(str(epoch)+ " : "+str(avg_psnr / len(testing_data_loader))+ "\n")
    f.close()

    # checkpoint
    dataset = "landscape"
    if epoch % 1 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", dataset)):
            os.mkdir(os.path.join("checkpoint", dataset))
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(dataset, epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(dataset, epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + dataset))


        # test
        with torch.no_grad():
            for i, batch_test in enumerate (testing_data_loader):

                # test
                real_A_test = Variable(batch_test["A"].type(Tensor))  # edge

                fake_B_test = net_g(real_A_test)
                out_img = fake_B_test.detach().squeeze(0).cpu()

                fig, axes = plt.subplots(1, 3, figsize=(18,6))
                axes[0].imshow(real_A_test.cpu().squeeze(0).permute(1,2,0), cmap="gray")
                axes[0].axis("off")

                axes[1].imshow(out_img.permute(1,2,0))
                axes[1].axis("off")

                axes[2].imshow(batch_test["B"].squeeze(0).permute(1, 2, 0))
                axes[2].axis("off")
                plt.savefig("per_epoch_result/{}_{}.jpg".format(epoch, i))

                if i == 1:
                    break