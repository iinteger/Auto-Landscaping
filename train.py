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
from models import define_G, define_D, update_learning_rate
from dataset import My_Dataset
from torch.nn import DataParallel
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import lr_scheduler
plt.axis("off")


# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument("--epoch", type=int, default=183, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='cosine', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=0)
opt = parser.parse_args()

print(opt)
dataset = "landscape"

os.makedirs("images/%s" % dataset, exist_ok=True)
os.makedirs("saved_models/%s" % dataset, exist_ok=True)


cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)


img_size = 256
print('===> Loading datasets')
root_path = "dataset/"
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5), (0.5))
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
generator = define_G(input_nc=1, output_nc=3, norm="instance")
discriminator = define_D(input_nc=4)
generator = DataParallel(generator)
discriminator = DataParallel(discriminator)

criterion_GAN = nn.MSELoss()
criterion_pixelwise = nn.L1Loss()

# loss weight if l1 loss
lambda_pixel = 1

# Calculate output of image discriminator (PatchGAN)
patch = (1, img_size // 2 ** 4, img_size // 2 ** 4)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (dataset, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (dataset, opt.epoch)))

# setup optimizer
optimizer_g = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

generator_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=opt.n_epochs, eta_min=0)
discriminator_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=opt.n_epochs, eta_min=0)

start = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for iteration, batch in enumerate(training_data_loader, 1):

        real_A = Variable(batch["A"].type(Tensor))  # edge
        real_B = Variable(batch["B"].type(Tensor))  # color

        # adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # generator train start
        optimizer_g.zero_grad()

        # gan loss
        fake_B = generator(real_A)  # 엣지를 받아서 가짜 컬러 이미지를 생성
        pred_fake = discriminator(fake_B, real_A)  # 가짜 컬러 이미지와 진짜 엣지 이미지(condition)을 넣어서(concat) inference
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
        pred_real = discriminator(real_B, real_A)  # 실제 컬러 이미지와 실제 엣지 이미지(condition)을 넣어서(concat) inference
        loss_real = criterion_GAN(pred_real, valid)  # 진짜를 추론한 결과와 True의 mse. 1(True)로 추론할 수록 error가 낮아짐

        # fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)  # 가짜 컬러 이미지와 실제 엣지 이미지를 넣어서 inference. fake_B는 G로부터 나온 Tensor이기 때문에 detach를 하지 않으면 역전파가 흘러들어감
        loss_fake = criterion_GAN(pred_fake, fake)  # 가짜를 추론한 결과와 False의 mse. 0(False)로 추론할 수록 error가 낮아짐

        # total loss
        loss_D = 0.25*(loss_real + loss_fake)

        loss_D.backward()
        optimizer_d.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_D.item(), loss_G.item()))

    update_learning_rate(generator_scheduler, optimizer_g)
    update_learning_rate(discriminator_scheduler, optimizer_d)
    # test
    avg_psnr = 0
    for batch in testing_data_loader:
        real_A_test = Variable(batch["A"].type(Tensor))  # edge
        real_B_test = Variable(batch["B"].type(Tensor))  # edge

        prediction = generator(real_A_test)
        mse = criterion_GAN(prediction, real_B_test)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    print("epoch "+str(epoch)+": "+str(time.time()-start))

    f = open("PSNR_list.txt", 'a')
    f.write(str(epoch)+ " : "+str(avg_psnr / len(testing_data_loader))+ "\n")
    f.close()

    # checkpoint
    if epoch % 1 == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (dataset, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (dataset, epoch))

        # test
        with torch.no_grad():
            for i, batch_test in enumerate (testing_data_loader):

                # test
                real_A_test = Variable(batch_test["A"].type(Tensor))  # edge

                fake_B_test = generator(real_A_test)
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