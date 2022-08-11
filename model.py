from lib import *

'''
    Ec
    Esi
    Gi
    Gg
    D
'''
class AdaIN(nn.Module):
    '''
    Module normalize S-C encoder instead Concat
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        eps = 1e-5	
        mean_x = torch.mean(x, dim=[2,3])
        mean_y = torch.mean(y, dim=[2,3])

        std_x = torch.std(x, dim=[2,3])
        std_y = torch.std(y, dim=[2,3])

        mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
        mean_y = mean_y.unsqueeze(-1).unsqueeze(-1)

        std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
        std_y = std_y.unsqueeze(-1).unsqueeze(-1) + eps

        out = (x - mean_x)/ std_x * std_y + mean_y
        return out

class PoolBlock(nn.Module):
    def __init__(self, in_chan, out_chan, size = (1, 1)) -> None:
        super(PoolBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=1),
            nn.AdaptiveAvgPool2d(size)
        )
    def forward(self, x):
        return self.main(x)

class Res_block(nn.Module):
    def __init__(self, chan, down = False, groups = 1) -> None:
        super(Res_block, self).__init__()
        '''
        IF down: feature map size 1/2
        '''
        self.down = down
        if not down:
            self.layers = nn.Sequential(
                nn.Conv2d(chan, chan, kernel_size = 3, stride = 1, padding = 1, padding_mode='reflect', groups=groups),
                nn.BatchNorm2d(chan),
                nn.LeakyReLU(0.2, inplace = False),
                nn.Conv2d(chan, chan, kernel_size = 3, stride = 1, padding = 1, padding_mode='reflect', groups=groups),
            )
        elif down:
            self.layers = nn.Sequential(
                nn.Conv2d(chan, chan*2, kernel_size = 3, stride = 2, padding = 1, padding_mode='reflect', groups=groups),
                nn.BatchNorm2d(chan*2),
                nn.LeakyReLU(0.2, inplace = False),
                nn.Conv2d(chan*2, chan*2, kernel_size = 3, stride = 1, padding = 1, padding_mode='reflect', groups=groups),
                nn.BatchNorm2d(chan*2),
            )
            self.res = nn.Sequential(
                nn.Conv2d(chan, chan*2, kernel_size = 3, stride = 2, padding = 1, padding_mode='reflect', groups=groups),
                nn.BatchNorm2d(chan*2),
            )
        self.act = nn.LeakyReLU(0.2, inplace = False)
    def forward(self, x):
        if self.down:
            residual = self.res(x)
        else:
            residual = x

        x = self.layers(x)
        return self.act(x + residual)

class C_encoder(nn.Module):
    def __init__(self, chan = 32) -> None:
        super(C_encoder, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, chan, kernel_size=3, stride = 2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(chan),
            Res_block(chan, down = True, groups=chan),
            Res_block(chan*2, down = True, groups=chan*2),
        )
        self.layer1 = nn.Sequential(
            Res_block(chan*4, down = False),
            Res_block(chan*4, down = True),
        )
        self.layer2= nn.Sequential(
            Res_block(chan*8, down = False),
            Res_block(chan*8, down = True),
        )
        self.layer3 = nn.Sequential(
            Res_block(chan*16, down = False),
            Res_block(chan*16, down = True),
        )
        # self.layer2 = Res_block(chan*8, down = True)
        # self.layer3 = Res_block(chan*16, down = True)

    

        # self.down1 = nn.Sequential(
        #     nn.Conv2d(chan*8, chan*16, kernel_size=3, stride = 2, padding = 1, padding_mode="reflect"),
        #     nn.BatchNorm2d(chan*16),
        #     nn.Conv2d(chan*16, chan*32, kernel_size=3, stride = 2, padding = 1, padding_mode="reflect"),
        #     nn.BatchNorm2d(chan*32),
        #     # nn.LeakyReLU(0.2),
        # )
        # self.down2 = nn.Sequential(
        #     nn.Conv2d(chan*16, chan*32, kernel_size=3, stride = 2, padding = 1, padding_mode="reflect"),
        #     nn.BatchNorm2d(chan*32),
        #     # nn.LeakyReLU(0.2),
        # )
        # self.down3 = nn.Sequential(
        #     nn.Conv2d(chan*32, chan*32, kernel_size=3, stride = 1, padding = 1, padding_mode="reflect"),
        #     nn.BatchNorm2d(chan*32),
        #     # nn.LeakyReLU(0.2),
        # )
        # self.last_conv = nn.Sequential(
        #     nn.Conv2d(3072, 1024, kernel_size=1),
        #     nn.BatchNorm2d(1024),
        #     nn.LeakyReLU(0.2)
        # )
    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
    
        return layer1, layer2, layer3


class S_encoder(nn.Module):
    def __init__(self, chan = 32) -> None:
        super(S_encoder, self).__init__()

        self.conv = nn.Conv2d(3, chan, kernel_size=3, stride = 2, padding=1, padding_mode='reflect')
        self.res_layer1 = Res_block(chan, down = True) # 64x48x64
        self.res_layer2 = Res_block(chan*2, down = True) # 128x24x32
        self.res_layer3 = Res_block(chan*4, down = True) # 256x12x16
        self.res_layer4 = Res_block(chan*8, down = True) # 512x6x8
        self.res_layer5 = Res_block(chan*16, down = True)


    def forward(self, x):

        x = self.conv(x)
        x1 = self.res_layer1(x)
        x2 = self.res_layer2(x1)
        x3 = self.res_layer3(x2)
        x4 = self.res_layer4(x3)
        x5 = self.res_layer5(x4)
        return x3, x4, x5



class Up(nn.Module):
    def __init__(self, chan, last = False) -> None:
        super(Up, self).__init__()
        self.last = last
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(chan, chan//2, kernel_size=3, stride = 2, padding = 1, output_padding=1),
            nn.BatchNorm2d(chan//2),
            nn.LeakyReLU(0.2, inplace = False),
        )
        self.last_conv = nn.Conv2d(chan//2, 3, kernel_size=1, stride = 1)
    def forward(self, x):
        if self.last:
            return self.last_conv(self.main(x))
        return self.main(x)


class I_Generator(nn.Module):
    def __init__(self, chan = 2048) -> None:
        super(I_Generator, self).__init__()

        self.norm = AdaIN()
        self.s_encoder = S_encoder()

        self.layer1 = Up(chan) # 512x12x16
        self.layer2 = Up(chan//2)  # 256x24x32
        self.layer3 = Up(chan//4)  # 128x48x64
        self.layer4 = Up(chan//8)  # 64x96x128
        self.layer5 = Up(chan//16)  # 3x192x256
        self.layer6 = Up(chan//32, last=True)  # 3x192x256
        self.act = nn.Tanh()
    def forward(self, x, c1, c2, c3):
        '''
        c1 256x16x12
        c2 512x8x6
        c3 1024x4x3
        '''
        # S --> G
        x1, x2, x3 = self.s_encoder(x)

        f1 = torch.cat((x1, c1), dim = 1) # 512x16x12 
        f2 = torch.cat((x2, c2), dim = 1) # 1024x8x6
        f3 = torch.cat((x3, c3), dim = 1) # 2048x4x3

        layer1 = self.layer1(f3) + f2
        layer2 = self.layer2(layer1) + f1
        layer3 = self.layer3(layer2) 
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        return self.act(self.layer6(layer5))


class G_Generator(nn.Module):
    def __init__(self, chan = 1024) -> None:
        super(G_Generator, self).__init__()

        # self.c_encoder = C_encoder()
        # self.conv_init = nn.ConvTranspose2d(chan, chan, kernel_size=(4, 3), stride=1)
        self.layer1 = Up(chan) # 512x12x16
        self.layer2 = Up(chan//2)  # 256x24x32
        self.layer3 = Up(chan//4)  # 128x48x64
        self.layer4 = Up(chan//8)  # 64x96x128
        self.layer5 = Up(chan//16)  # 3x192x256
        self.layer6 = Up(chan//32, last=True)  # 3x192x256


        self.act = nn.Tanh()
    def forward(self, c1, c2, c3):
        '''
        C----> G-->        
        '''
        layer1 = self.layer1(c3) + c2
        layer2 = self.layer2(layer1) + c1
        layer3 = self.layer3(layer2) 
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        return self.act(layer6)



class D_Block(nn.Module):
    def __init__(self, chan = 64, first = False) -> None:
        super(D_Block, self).__init__()

        self.first = first
        self.init_conv = nn.Conv2d(3, chan, kernel_size=1)

        self.layer = nn.Sequential(
            spectral_norm(nn.Conv2d(chan, chan, kernel_size=3, stride = 1, padding = 1, padding_mode='reflect')),
            nn.LeakyReLU(0.2, inplace = False),
            spectral_norm(nn.Conv2d(chan, chan*2, kernel_size=3, stride = 2, padding = 1, padding_mode='reflect')),
        )
        self.conv = nn.Conv2d(chan, chan*2, kernel_size=3, stride = 2, padding = 1, padding_mode='reflect')
        self.act = nn.LeakyReLU(0.2, inplace = False)

    def forward(self, x):
        if self.first:
            x = self.init_conv(x)
        res = self.conv(x)
        x = self.layer(x)
        return self.act(x + res)

class Discriminator(nn.Module):
    def __init__(self, chan = 64) -> None:
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            D_Block(chan, first=True),
            D_Block(chan*2),
            D_Block(chan*4),
            D_Block(chan*8),
            D_Block(chan*16)
        )

    def forward(self, x):
        return self.main(x)
if __name__ == "__main__":
    model = I_Generator()
    c = C_encoder()

    x = torch.rand(1, 3, 256, 192)

    c1, c2, c3 = c(x)

    y = model(x, c1, c2, c3)
    print(y.size())

