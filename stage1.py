from lib import *
from dataloader import Loader
from utils import *
from model import Generator, PatchDiscriminator, Critic
from flow import AFWM
from config import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

train_data = Loader(g_path="VITON_traindata/train_color", p_path="VITON_traindata/train_img")
train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, shuffle=True, drop_last=True)
writer = SummaryWriter("tensorboard")

warp_model = AFWM(3).to(device)
load_checkpoint(warp_model, "./ckp/non_aug/PFAFN_warp_epoch_101.pth")

G_A = Generator().to(device)
# G_A.load_state_dict(torch.load("weights_stage1/G_A_stage1_0.pth"))
# G_A = nn.DataParallel(G_A, device_ids=[0, 1])
G_B = Generator().to(device)
# G_A.load_state_dict(torch.load("weights_stage1/G_B_stage1_0.pth"))
# G_B = nn.DataParallel(G_B, device_ids=[0, 1])

# Critic_32 = Critic(32, 4).to(device)
# Critic_32 = nn.DataParallel(Critic_32, device_ids=[0, 1])
# Critic_64 = Critic(64, 3).to(device)
# Critic_64 = nn.DataParallel(Critic_64, device_ids=[0, 1])
# Critic_128 = Critic(128, 2).to(device)
# Critic_128 = nn.DataParallel(Critic_128, device_ids=[0, 1])
# Critic_256 = Critic(256, 1).to(device)
# Critic_256 = nn.DataParallel(Critic_256, device_ids=[0, 1])
# Critic_512 = Critic(512, 0).to(device)
# Critic_512 = nn.DataParallel(Critic_512, device_ids=[0, 1])

GA_optim = optim.Adam(G_A.parameters(), lr = lr, betas=betas, weight_decay=weight_decay)
GB_optim = optim.Adam(G_B.parameters(), lr = lr/10, betas=betas, weight_decay=weight_decay)
GA_lr_sche = lr_scheduler.StepLR(GA_optim, step_size= 1, gamma=0.5)
GB_lr_sche = lr_scheduler.StepLR(GB_optim, step_size= 1, gamma=0.5)

# critic_param = list(Critic_32.parameters()) + list(Critic_64.parameters()) + list(Critic_128.parameters()) + list(Critic_256.parameters()) + list(Critic_512.parameters()) 
# critic_optim = optim.Adam([p for p in critic_param if p.requires_grad], lr = lr, betas=betas, weight_decay=weight_decay)
# critic_lr_sche = lr_scheduler.StepLR(critic_optim, step_sssize= EPOCH//10, gamma=0.5)

l1 = nn.L1Loss().to(device)

def save_model(e):
    torch.save(G_A.state_dict(), f'weights_stage1/G_A_stage1_{e}.pth')
    torch.save(G_B.state_dict(), f'weights_stage1/G_B_stage1_{e}.pth')
    # torch.save(Critic_32.state_dict(), f'weights_stage1/Critic_32_stage1_{e}.pth')
    # torch.save(Critic_64.state_dict(), f'weights_stage1/Critic_64_stage1_{e}.pth')
    # torch.save(Critic_128.state_dict(), f'weights_stage1/Critic_128_stage1_{e}.pth')
    # torch.save(Critic_256.state_dict(), f'weights_stage1/Critic_256_stage1_{e}.pth')
    # torch.save(Critic_512.state_dict(), f'weights_stage1/Critic_512_stage1_{e}.pth')

def model_train():
    G_A.train()
    G_B.train()
    # Critic_32.train()
    # Critic_64.train()
    # Critic_128.train()
    # Critic_256.train()
    # Critic_512.train()

global_step = 0
_i_loss = []
_g_loss = []
for e in range(EPOCH):
    model_train()

    for i, (xA, xB) in enumerate(tqdm(train_loader)):
        xA, xB = xA.to(device), xB.to(device)

        # Image
        c_a, s_a = G_A.encode(xA)
        x_AA = G_A.decode(c_a, s_a) # AE
        i_loss = l1(x_AA, xA)
        GA_optim.zero_grad()
        i_loss.backward()
        GA_optim.step()
        _i_loss.append(i_loss.item())
        # Garment
        warp_xA = copy.deepcopy(xA)
        warp_xB = copy.deepcopy(xB)
        flow_out = warp_model(warp_xA, warp_xB)
        warped_cloth, last_flow = flow_out

        c_b, s_b = G_B.encode(warped_cloth.detach())
        x_BB = G_B.decode(c_b, s_b)
        g_loss = l1(x_BB, warped_cloth)
        GB_optim.zero_grad()
        g_loss.backward()
        GB_optim.step()
        _g_loss.append(g_loss.item())
        # Plot
        global_step += 1
        # Test
        if global_step % 100 == 0:
            print(f'Epoch: {e}\ni_loss: { np.mean(_i_loss)}\ng_loss: {np.mean(_g_loss)}')
            save_img([xA, x_AA, warped_cloth, x_BB], f"outputs/{global_step}.png")
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
    
