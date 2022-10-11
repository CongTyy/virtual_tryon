from lib import *
from dataloader import Loader
from utils import *
from model import Generator, Critic
from flow import AFWM
from config import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

train_data = Loader(g_path="VITON_traindata/train_color", p_path="VITON_traindata/train_img")
train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, shuffle=True, drop_last=True)
writer = SummaryWriter("tensorboard_2")

warp_model = AFWM(3).to(device)
load_checkpoint(warp_model, "./ckp/non_aug/PFAFN_warp_epoch_101.pth")

G_A = Generator().to(device)
G_A.load_state_dict(torch.load("weights_stage1/G_A_stage1_3.pth"))
# G_A = nn.DataParallel(G_A, device_ids=[0, 1])
G_B = Generator().to(device)
G_A.load_state_dict(torch.load("weights_stage1/G_B_stage1_3.pth"))
# G_B = nn.DataParallel(G_B, device_ids=[0, 1])

Critic_32 = Critic(32, 4).to(device)
# Critic_32 = nn.DataParallel(Critic_32, device_ids=[0, 1])
Critic_64 = Critic(64, 3).to(device)
# Critic_64 = nn.DataParallel(Critic_64, device_ids=[0, 1])
Critic_128 = Critic(128, 2).to(device)
# Critic_128 = nn.DataParallel(Critic_128, device_ids=[0, 1])
Critic_256 = Critic(256, 1).to(device)
# Critic_256 = nn.DataParallel(Critic_256, device_ids=[0, 1])
# Critic_512 = Critic(512, 0).to(device)
# Critic_512 = nn.DataParallel(Critic_512, device_ids=[0, 1])

GA_optim = optim.Adam(G_A.parameters(), lr = lr, betas=betas, weight_decay=weight_decay)
GB_optim = optim.Adam(G_B.parameters(), lr = lr/10, betas=betas, weight_decay=weight_decay)
GA_lr_sche = lr_scheduler.StepLR(GA_optim, step_size= 1, gamma=0.5)
GB_lr_sche = lr_scheduler.StepLR(GB_optim, step_size= 1, gamma=0.5)

critic_param = list(Critic_32.parameters()) + list(Critic_64.parameters()) + list(Critic_128.parameters()) + list(Critic_256.parameters()) #+ list(Critic_512.parameters()) 
critic_optim = optim.Adam([p for p in critic_param if p.requires_grad], lr = lr, betas=betas, weight_decay=weight_decay)
critic_lr_sche = lr_scheduler.StepLR(critic_optim, step_size= EPOCH//10, gamma=0.5)

l1 = nn.L1Loss().to(device)

def save_model(e):
    torch.save(G_A.state_dict(), f'weights_stage2/G_A_stage1_{e}.pth')
    torch.save(G_B.state_dict(), f'weights_stage2/G_B_stage1_{e}.pth')
    torch.save(Critic_32.state_dict(), f'weights_stage2/Critic_32_stage1_{e}.pth')
    torch.save(Critic_64.state_dict(), f'weights_stage2/Critic_64_stage1_{e}.pth')
    torch.save(Critic_128.state_dict(), f'weights_stage2/Critic_128_stage1_{e}.pth')
    torch.save(Critic_256.state_dict(), f'weights_stage2/Critic_256_stage1_{e}.pth')
    # torch.save(Critic_512.state_dict(), f'weights_stage2/Critic_512_stage1_{e}.pth')

def model_train():
    G_A.train()
    G_B.train()
    Critic_32.train()
    Critic_64.train()
    Critic_128.train()
    Critic_256.train()
    # Critic_512.train()

global_step = 0
_i_loss = []
_g_loss = []



for e in range(EPOCH):
    model_train()

    for i, (xA, xB) in enumerate(tqdm(train_loader)):
        xA, xB = xA.to(device), xB.to(device)

        # critic update
        dxA = copy.deepcopy(xA)
        dxB = copy.deepcopy(xB)

        c_a, s_a = G_A.encode(dxA)

        dxB, _ = warp_model(dxA, dxB)
        c_b, s_b = G_B.encode(dxB.detach())

        d_32A = Critic_32(c_a[5])
        d_32B = Critic_32(c_b[5])

        d_64A = Critic_64(c_a[4])
        d_64B = Critic_64(c_b[4])

        d_128A = Critic_128(c_a[3])
        d_128B = Critic_128(c_b[3])

        d_256A = Critic_256(c_a[2])
        d_256B = Critic_256(c_b[2])

        d_32 = gradient_penalty(Critic_32, c_b[5], c_a[5])
        d_64 = gradient_penalty(Critic_64, c_b[4], c_a[4])
        d_128 = gradient_penalty(Critic_128, c_b[3], c_a[3])
        d_256 = gradient_penalty(Critic_256, c_b[2], c_a[2])

        d32_loss = -(d_32B.mean() - d_32A.mean()) + GP*d_32 # -( real - fake ) + gp
        d64_loss = -(d_64B.mean() - d_64A.mean()) + GP*d_64 # -( real - fake ) + gp
        d128_loss = -(d_128B.mean() - d_128A.mean()) + GP*d_128 # -( real - fake ) + gp
        d256_loss = -(d_256B.mean() - d_256A.mean()) + GP*d_256 # -( real - fake ) + gp
        d_loss = d32_loss + d64_loss + d128_loss + d256_loss
        critic_optim.zero_grad()
        d_loss.backward()
        critic_optim.step()

        
        # # Image
        
        c_a, s_a = G_A.encode(xA)
        x_AA = G_A.decode(c_a, s_a) # AE
        i_loss = l1(x_AA, xA)
        GA_optim.zero_grad()
        i_loss.backward()
        GA_optim.step()
        _i_loss.append(i_loss.item())
        # Garment
        c_b, s_b = G_B.encode(dxB.detach())
        x_BB = G_B.decode(c_b, s_b)
        g_loss = l1(x_BB, dxB)
        GB_optim.zero_grad()
        g_loss.backward()
        GB_optim.step()
        _g_loss.append(g_loss.item())
        # # Plot
        global_step += 1
        # Test
        if global_step % 100 == 0:
            print(f'Epoch: {e}\ni_loss: { np.mean(_i_loss)}\ng_loss: {np.mean(_g_loss)}')
            save_img([xA, x_AA, dxB, x_BB], f"outputs_stage2/{global_step}.png")
            writer.add_scalar('Epoch', e, global_step)
            writer.add_scalar('i_loss', np.mean(_i_loss), global_step)
            writer.add_scalar('g_loss', np.mean(_g_loss), global_step)
            
            _i_loss = []
            _g_loss = []

    # Save model
    save_model(e)

    # Update learning rate
    GA_lr_sche.step()
    GB_lr_sche.step()
    writer.add_scalar('G_lr', GA_lr_sche.get_last_lr()[0], global_step)
    
class Trainer:
    def __init__(self) -> None:
        self.train_data = Loader(g_path="VITON_traindata/train_color", p_path="VITON_traindata/train_img")
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size= 1, shuffle=True, drop_last=True)
        self.writer = SummaryWriter("tensorboard_2")

        self.warp_model = AFWM(3).to(device)
        load_checkpoint(self.warp_model, "./ckp/non_aug/PFAFN_warp_epoch_101.pth")
        self.G_A = Generator().to(device)
        self.G_A.load_state_dict(torch.load("weights_stage1/G_A_stage1_3.pth"))
        self.G_B = Generator().to(device)
        self.G_B.load_state_dict(torch.load("weights_stage1/G_B_stage1_3.pth"))
        # G_B = nn.DataParallel(G_B, device_ids=[0, 1])

        self.Critic_32 = Critic(32, 4).to(device)
        self.Critic_64 = Critic(64, 3).to(device)
        self.Critic_128 = Critic(128, 2).to(device)
        self.Critic_256 = Critic(256, 1).to(device)

        self.GA_optim = optim.Adam(self.G_A.parameters(), lr = lr, betas=betas, weight_decay=weight_decay)
        self.GB_optim = optim.Adam(self.G_B.parameters(), lr = lr/10, betas=betas, weight_decay=weight_decay)
        self.GA_lr_sche = lr_scheduler.StepLR(GA_optim, step_size= 1, gamma=0.5)
        self.GB_lr_sche = lr_scheduler.StepLR(GB_optim, step_size= 1, gamma=0.5)

        self.critic_param = list(self.Critic_32.parameters()) + list(self.Critic_64.parameters()) + list(self.Critic_128.parameters()) + list(self.Critic_256.parameters()) #+ list(self.Critic_512.parameters()) 
        self.critic_optim = optim.Adam([p for p in self.critic_param if p.requires_grad], lr = lr, betas=betas, weight_decay=weight_decay)
        self.critic_lr_sche = lr_scheduler.StepLR(self.critic_optim, step_size= EPOCH//10, gamma=0.5)

        self.l1 = nn.L1Loss().to(device)

        self.e = 0
    def model_train(self):
        self.G_A.train()
        self.G_B.train()
        self.Critic_32.train()
        self.Critic_64.train()
        self.Critic_128.train()
        self.Critic_256.train()

    def save_model(self):
        torch.save(self.G_A.state_dict(), f'weights_stage2/G_A_stage1_{self.e}.pth')
        torch.save(self.G_B.state_dict(), f'weights_stage2/G_B_stage1_{self.e}.pth')
        torch.save(self.Critic_32.state_dict(), f'weights_stage2/Critic_32_stage1_{self.e}.pth')
        torch.save(self.Critic_64.state_dict(), f'weights_stage2/Critic_64_stage1_{self.e}.pth')
        torch.save(self.Critic_128.state_dict(), f'weights_stage2/Critic_128_stage1_{self.e}.pth')
        torch.save(self.Critic_256.state_dict(), f'weights_stage2/Critic_256_stage1_{self.e}.pth')

    def warp(self, xA, xB):
        warp, _ = self.warp_model(xA, xB)
        return warp.detach()

    def train_critic(self, xA, xB, x_warp):
        dxA = copy.deepcopy(xA)
        dxB = copy.deepcopy(xB)

        c_a, s_a = self.G_A.encode(dxA)

        # dxB = self.warp(dxA, dxB)
        c_b, s_b = self.G_B.encode(x_warp)

        d_32A = self.Critic_32(c_a[5])
        d_32B = self.Critic_32(c_b[5])

        d_64A = self.Critic_64(c_a[4])
        d_64B = self.Critic_64(c_b[4])

        d_128A = self.Critic_128(c_a[3])
        d_128B = self.Critic_128(c_b[3])

        d_256A = self.Critic_256(c_a[2])
        d_256B = self.Critic_256(c_b[2])

        d_32 = gradient_penalty(Critic_32, c_b[5], c_a[5])
        d_64 = gradient_penalty(Critic_64, c_b[4], c_a[4])
        d_128 = gradient_penalty(Critic_128, c_b[3], c_a[3])
        d_256 = gradient_penalty(Critic_256, c_b[2], c_a[2])

        d32_loss = -(d_32B.mean() - d_32A.mean()) + GP*d_32 # -( real - fake ) + gp
        d64_loss = -(d_64B.mean() - d_64A.mean()) + GP*d_64 # -( real - fake ) + gp
        d128_loss = -(d_128B.mean() - d_128A.mean()) + GP*d_128 # -( real - fake ) + gp
        d256_loss = -(d_256B.mean() - d_256A.mean()) + GP*d_256 # -( real - fake ) + gp
        d_loss = d32_loss + d64_loss + d128_loss + d256_loss
        self.critic_optim.zero_grad()
        d_loss.backward()
        self.critic_optim.step()

    def train_gen(self, xA, xB, x_warp):

        c_a, s_a = G_A.encode(xA)
        x_AA = G_A.decode(c_a, s_a) # AE
        loss_ae_a = self.l1(x_AA, xA)
        self.GA_optim.zero_grad()
        loss_ae_a.backward()
        self.GA_optim.step()

        c_b, s_b = G_B.encode(x_warp)
        x_BB = G_B.decode(c_b, s_b)
        loss_ae_b = self.l1(x_BB, x_warp)
        self.GB_optim.zero_grad()
        loss_ae_b.backward()
        self.GB_optim.step()

    def plot(self):
        return
    
    def trainer(self):
        self.model_train()

        for i, (xA, xB) in enumerate(tqdm(self.train_loader)):
            xA, xB = xA.to(device), xB.to(device)

            x_warp = self.warp(xA, xB)
            self.train_critic(xA, xB, x_warp)
            self.train_gen(xA, xB, x_warp)