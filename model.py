import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.utils.spectral_norm as spectral_norm

class ConvBlock(nn.Module):
    def __init__(self, in_chan = 3, out_chan = 64, ks = 3, stride = 2, padding = 1) -> None:
        super(ConvBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module):
    def __init__(self, in_chan = 3) -> None:
        super(ResBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(in_chan),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(in_chan),
        )
        self.act = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        return self.act(self.main(x) + x)


class Content_Encoder(nn.Module):
    def __init__(self) -> None:
        super(Content_Encoder, self).__init__()

        '''
            3   x  256  x  192 - input
            32  x  128  x  96  - layer1
            64  x  64   x  48  - layer2
            128 x  32   x  24  - layer3
            256 x  16   x  12  - layer4
            256 x  8    x  6   - layer5
            512 x  4    x  3   - layer6
        '''
        self.layer1 = nn.Sequential(
            ConvBlock(3, 32),
            ResBlock(32)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(32, 64),
            ResBlock(64)
        )       
        self.layer3 = nn.Sequential(
            ConvBlock(64, 128),
            ResBlock(128)
        )       
        self.layer4 = nn.Sequential(
            ConvBlock(128, 256),
            ResBlock(256)
        )     
        self.layer5= nn.Sequential(            
            ConvBlock(256, 256),
            ResBlock(256),
        )
        self.layer6 = nn.Sequential(            
            ConvBlock(256, 512),
            ResBlock(512),
        )     
    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        return layer6, layer5, layer4, layer3, layer2, layer1

class Specific_Encoder(nn.Module):
    def __init__(self):
        super(Specific_Encoder, self).__init__()
        '''
            3   x  256  x  192 - input
            32  x  128  x  96  - layer1
            64  x  64   x  48  - layer2
            128 x  32   x  24  - layer3
            256 x  16   x  12  - layer4
            512 x  8    x  6   - layer5
            512 x  4    x  3   - layer6
            512 x  1    x  1   - layer7
        '''
        self.layer1 = nn.Sequential(
            ConvBlock(3, 32),
            ResBlock(32)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(32, 64),
            ResBlock(64)
        )       
        self.layer3 = nn.Sequential(
            ConvBlock(64, 128),
            ResBlock(128)
        )       
        self.layer4 = nn.Sequential(
            ConvBlock(128, 256),
            ResBlock(256)
        )     
        self.layer5 = nn.Sequential(            
            ConvBlock(256, 512),
            ResBlock(512),
        ) 
        self.layer6 = nn.Sequential(            
            ConvBlock(512, 512),
            ResBlock(512),
        ) 
        self.layer7 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        layer1 = self.layer1(x) # 32  x  128  x  96
        layer2 = self.layer2(layer1) # 64  x  64   x  48
        layer3 = self.layer3(layer2) # 128 x  32   x  24
        layer4 = self.layer4(layer3) # 256 x  16   x  12
        layer5 = self.layer5(layer4) # 256 x  8    x  6

        layer6 = self.layer6(layer5) # 512 x  4    x  3
        layer7 = self.layer7(layer5) # 512 x  1    x  1

        layer7 = layer7.reshape(-1, 512, 1, 1)
        layer6 = torch.mul(layer6, layer7)
        return layer6, layer5, layer4, layer3, layer2, layer1
    

class DeConvBlock(nn.Module):
    def __init__(self, in_chan = 256, out_chan = 128, last = False) -> None:
        super(DeConvBlock, self).__init__()

        if last:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_chan, out_chan, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_chan),
                nn.Tanh(),
                # ConvBlock(out_chan, out_chan)
            )  
        else:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_chan, out_chan, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_chan),
                nn.LeakyReLU(inplace=True),
                # ConvBlock(out_chan, out_chan)
            )
    
    def forward(self, x):
        return self.main(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        '''
        Specific:
            torch.Size([1, 512, 4, 3])      layer6
            torch.Size([1, 512, 8, 6])      layer5
            torch.Size([1, 256, 16, 12])    layer4
            torch.Size([1, 128, 32, 24])    layer3
            torch.Size([1, 64, 64, 48])     layer2
            torch.Size([1, 32, 128, 96])    layer1
        Content:
            torch.Size([1, 512, 4, 3])      layer6
            torch.Size([1, 256, 8, 6])      layer5
            torch.Size([1, 256, 16, 12])    layer4
            torch.Size([1, 128, 32, 24])    layer3
            torch.Size([1, 64, 64, 48])     layer2
            torch.Size([1, 32, 128, 96])    layer1
        '''
        self.up1 = nn.Sequential(
            ResBlock(1024),
            DeConvBlock(1024, 768),
            ResBlock(768),
        )
        self.up2 = nn.Sequential(
            ResBlock(768*2),
            DeConvBlock(768*2, 512),
            ResBlock(512)
        )
        self.up3 = nn.Sequential(
            ResBlock(512*2),
            DeConvBlock(512*2, 256),
            ResBlock(256)
        )
        self.up4 = nn.Sequential(
            ResBlock(256*2),
            DeConvBlock(256*2, 128),
            ResBlock(128)
        )
        self.up5 = nn.Sequential(
            ResBlock(128*2),
            DeConvBlock(128*2, 64),
            ResBlock(64)
        )
        self.up6 = nn.Sequential(
            ResBlock(64*2),
            DeConvBlock(64*2, 32),
            ResBlock(32),
            nn.Conv2d(32, 3, kernel_size=1, stride = 1, padding = 0, bias = False),
            nn.Tanh()
        )
    def forward(self, c, s): # S -> (B, 512, 1, 1) c -> (B, 512, 8, 6)

        s6, s5, s4, s3, s2, s1 = s[0], s[1], s[2], s[3], s[4], s[5]
        c6, c5, c4, c3, c2, c1 = c[0], c[1], c[2], c[3], c[4], c[5]

        x6 = torch.cat((c6, s6), dim = 1)
        x6 = self.up1(x6)

        x5 = torch.cat((x6, c5, s5), dim = 1)
        x5 = self.up2(x5)

        x4 = torch.cat((x5, c4, s4), dim = 1)
        x4 = self.up3(x4)

        x3 = torch.cat((x4, c3, s3), dim = 1)
        x3 = self.up4(x3)

        x2 = torch.cat((x3, c2, s2), dim = 1)
        x2 = self.up5(x2)

        x1 = torch.cat((x2, c1, s1), dim = 1)
        x1 = self.up6(x1)

        return x1

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()

        self.content_enc = Content_Encoder()
        self.style_enc = Specific_Encoder()
        self.decoder = Decoder()

    def encode(self, img):
        ce = self.content_enc(img)
        se = self.style_enc(img)
        return ce, se

    def decode(self, c, s):
        return self.decoder(c, s)

    def forward(self, c_img, s_img):
        # Encoder
        content = self.content_enc(c_img)
        style = self.style_enc(s_img)
        # Decoder
        img = self.decoder(content, style)
        return img  

# class Spatial_Transform(nn.Module):
#     def __init__(self):
#         super(Spatial_Transform, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=3*256*192,
#                       out_features=20),
#             nn.BatchNorm1d(20),
#             nn.LeakyReLU(),
#             nn.Linear(in_features=20, out_features=6),
#             nn.LeakyReLU(),
#         )
#         bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))

#     def forward(self, g_img, p_img):

#         batch_size = p_img.size(0)

#         theta = self.fc(p_img.view(batch_size, -1)).view(batch_size, 2, 3)

#         grid = F.affine_grid(theta, torch.Size((batch_size, 3,256, 192)))
#         img_transform = F.grid_sample(g_img, grid)
#         return img_transform


# def l2normalize(v, eps = 1e-12):
#     return v / (v.norm() + eps)

# class SpectralNorm(nn.Module):
#     def __init__(self, module, name = 'weight', power_iterations = 1):
#         super(SpectralNorm, self).__init__()
#         self.module = module
#         self.name = name
#         self.power_iterations = power_iterations
#         if not self._made_params():
#             self._make_params()

#     def _update_u_v(self):
#         u = getattr(self.module, self.name + "_u")
#         v = getattr(self.module, self.name + "_v")
#         w = getattr(self.module, self.name + "_bar")

#         height = w.data.shape[0]
#         for _ in range(self.power_iterations):
#             v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
#             u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

#         # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
#         sigma = u.dot(w.view(height, -1).mv(v))
#         setattr(self.module, self.name, w / sigma.expand_as(w))

#     def _made_params(self):
#         try:
#             u = getattr(self.module, self.name + "_u")
#             v = getattr(self.module, self.name + "_v")
#             w = getattr(self.module, self.name + "_bar")
#             return True
#         except AttributeError:
#             return False

#     def _make_params(self):
#         w = getattr(self.module, self.name)

#         height = w.data.shape[0]
#         width = w.view(height, -1).data.shape[1]

#         u = Parameter(w.data.new(height).normal_(0, 1), requires_grad = False)
#         v = Parameter(w.data.new(width).normal_(0, 1), requires_grad = False)
#         u.data = l2normalize(u.data)
#         v.data = l2normalize(v.data)
#         w_bar = Parameter(w.data)

#         del self.module._parameters[self.name]

#         self.module.register_parameter(self.name + "_u", u)
#         self.module.register_parameter(self.name + "_v", v)
#         self.module.register_parameter(self.name + "_bar", w_bar)

#     def forward(self, *args):
#         self._update_u_v()
#         return self.module.forward(*args)


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        self.block1 = nn.Sequential(
            ConvBlock(3, 64),
            ResBlock(64),
            ResBlock(64)
        )
        self.block2 = nn.Sequential(
            ConvBlock(64, 128),
            ResBlock(128),
            ResBlock(128)
        )
        self.block3 = nn.Sequential(
            ConvBlock(128, 256),
            ResBlock(256),
            ResBlock(256)
        )
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = nn.Sequential(
            ConvBlock(256, 512, ks = 1, stride=1, padding=0),
            ResBlock(512),
            ResBlock(512)
        )
        self.final2 = ConvBlock(512, 1, ks = 1, stride = 1, padding = 0)

    def forward(self, x):
        # Concatenate image and condition image by channels to produce input
        x = self.block1(x)                                      # out: batch * 64 * 128 * 96
        x = self.block2(x)                                      # out: batch * 128 * 64 * 48
        x = self.block3(x)                                      # out: batch * 256 * 32 * 24
        x = self.final1(x)                                      # out: batch * 512 * 32 * 24
        x = self.final2(x)                                      # out: batch * 1 * 32 * 24
        return x


# torch.Size([1, 512, 4, 3])      layer6
# torch.Size([1, 256, 8, 6])      layer5
# torch.Size([1, 256, 16, 12])    layer4
# torch.Size([1, 128, 32, 24])    layer3
# torch.Size([1, 64, 64, 48])     layer2
# torch.Size([1, 32, 128, 96])    layer1
class Critic(nn.Module):
    def __init__(self, inchan = 32, times = 1) -> None:
        super(Critic, self).__init__()


        if times != 0:
            layers = []
            outchan = inchan*2
            for i in range(times):
                layers.append(spectral_norm(nn.Conv2d(inchan, outchan, kernel_size=3, stride = 2, padding = 1)))
                layers.append(nn.LeakyReLU(inplace=True))
                layers.append(ResBlock(outchan))
                inchan = outchan
                outchan *= 2
                if i == (times - 2):
                    outchan = 1
            self.main = nn.Sequential(*layers)
        else:
            self.main = nn.Sequential(
                nn.Conv2d(inchan, 1, kernel_size=1, stride=1, padding=0),
            )
    def forward(self, x):
        return self.main(x)

if __name__== "__main__":
    from torchinfo import summary
    from lib import *
    from dataloader import Loader
    import time
    # model = Specific_Encoder().to('cuda:1')
    # x = torch.rand(1, 3, 256, 192).to('cuda:1')

    # model = Generator().to('cuda:1')
    # x = torch.rand(1, 3, 256, 192).to('cuda:1')

    # c, s = model.encode(x)
    # y = model.decode(c, s)
    # print(y.size())\
    model = Critic(32, times = 3)
    summary(model, (32, 128, 96), device='cpu')
    
