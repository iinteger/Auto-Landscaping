# contributed by eriklindernoren github

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02):
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, norm='batch', init_type='normal'):
    init_gain = 0.2
    net = ResnetGenerator(input_nc, output_nc, norm=norm)

    return init_net(net, init_type, init_gain)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, norm="batch"):
        super(ResnetGenerator, self).__init__()

        self.down1 = UNetDown(input_nc, 64)
        self.down2 = UNetDown(64, 128, norm_layer=norm)
        self.down3 = UNetDown(128, 256, norm_layer=norm)
        self.down4 = UNetDown(256, 512, norm_layer=norm)
        self.down5 = UNetDown(512, 512, norm_layer=norm)
        self.down6 = UNetDown(512, 512, norm_layer=norm)
        self.down7 = UNetDown(512, 512, norm_layer=norm)
        self.down8 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, output_nc, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        print("1",u1.shape)
        u2 = self.up2(u1, d6)
        print("2",u2.shape)
        u3 = self.up3(u2, d5)
        print("3",u3.shape)
        u4 = self.up4(u3, d4)
        print("4",u4.shape)
        u5 = self.up5(u4, d3)
        print("5",u5.shape)
        u6 = self.up6(u5, d2)
        print("6",u6.shape)
        u7 = self.up7(u6, d1)
        print("7",u7.shape)
        return self.final(u7)


class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=False, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)]

        if norm_layer == "instance":  # image gen에는 instance가 더 좋다고 함
            layers.append(nn.InstanceNorm2d(out_ch))
        elif norm_layer == "batch":
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_nc, out_nc, dropout=0.0):
        super(UNetUp, self).__init__()


        layers = [nn.ConvTranspose2d(in_nc, out_nc, kernel_size=4, stride=2, padding=1),
                  nn.BatchNorm2d(out_nc),
                  nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


def define_D(input_nc, init_type='normal', init_gain=0.02):
    net = Discriminator(input_nc)
    return init_net(net, init_type, init_gain)


class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize="batch"):
            if normalize == "batch":
                norm_layer = nn.BatchNorm2d(out_filters)
            elif normalize == "instance":
                norm_layer = nn.InstanceNorm2d(out_filters)
            else:
                return nn.Sequential(
                    nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)
                )

            return nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                norm_layer,
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.layer1 = discriminator_block(4, 64, normalize=False)
        self.layer2 = discriminator_block(64, 128)
        self.layer3 = discriminator_block(128, 256)
        self.layer4 = discriminator_block(256, 512)
        self.layer5 = discriminator_block(512, 1, normalize=False)



    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        x = self.layer1(img_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return torch.sigmoid(x)