from lib import *
from dataloader import Loader
from utils import *
from model import Generator, PatchDiscriminator, F_Discriminator
from flow import AFWM
from config import *

train_data = Loader(g_path="VITON_traindata/train_color", p_path="VITON_traindata/train_img")
train_loader = torch.utils.data.DataLoader(train_data, batch_size= 4, shuffle=True, drop_last=True)
writer = SummaryWriter()

GA = Generator().cuda()
GA = nn.DataParallel(GA, device_ids=[0,1])
GA.load_state_dict(torch.load("./weights_stage1/G_A_stage1_24.pth"))

GB = Generator().cuda()
GB = nn.DataParallel(GB, device_ids=[0,1])
GB.load_state_dict(torch.load("./weights_stage1/G_B_stage1_24.pth"))

DA = PatchDiscriminator().cuda()
DA = nn.DataParallel(DA, device_ids=[0,1])
DB = PatchDiscriminator().cuda()
DB = nn.DataParallel(DB, device_ids=[0,1])


# Di = PatchDiscriminator().cuda() # image
# Di = nn.DataParallel(Di, device_ids=[0,1])
# Df1 = F_Discriminator(chan = 256).cuda() # feature
# Df1 = nn.DataParallel(Df1, device_ids=[0,1])
# Df2 = F_Discriminator(chan = 512).cuda() # feature
# Df2 = nn.DataParallel(Df2, device_ids=[0,1])

warp_model = AFWM(3).to('cuda:1')
load_checkpoint(warp_model, "./ckp/non_aug/PFAFN_warp_epoch_101.pth")
# warp_model = nn.DataParallel(warp_model, device_ids=[0,1])
# warp_optim = optim.Adam(warp_model.parameters(), lr = 1e-6, betas=betas, weight_decay=weight_decay)
# warp_lr_sche= lr_scheduler.StepLR(warp_optim, step_size= EPOCH//10, gamma=0.5)

G_params = list(GA.parameters()) + list(GB.parameters()) 
G_optim = optim.Adam([p for p in G_params if p.requires_grad], lr = lr, betas=betas, weight_decay=weight_decay)
G_lr_sche = lr_scheduler.StepLR(G_optim, step_size= EPOCH//10, gamma=0.5)

D_params = list(DA.parameters()) + list(DB.parameters())
D_optim = optim.Adam([p for p in D_params if p.requires_grad], lr = lr, betas=betas, weight_decay=weight_decay)
D_lr_sche = lr_scheduler.StepLR(D_optim, step_size= EPOCH//10, gamma=0.5)

l1 = nn.L1Loss().cuda()

def save_model(e):
    torch.save(GA.state_dict(), f'weights/GA_{e}.pth')
    torch.save(GA.state_dict(), f'weights/GB_{e}.pth')
    torch.save(warp_model.state_dict(), f'weights/warp_model_{e}.pth')
    torch.save(DA.state_dict(), f'weights/Df1_{e}.pth')
    torch.save(DB.state_dict(), f'weights/Df2_{e}.pth')
    # torch.save(Di.state_dict(), f'weights/Di_{e}.pth')

def model_train():
    GA.train()
    GB.train()
    warp_model.train()
    DA.train()
    DB.train()
    # Df2.train()
    

def model_test():
    GA.eval()
    GB.eval()
    warp_model.eval()
    DA.eval()
    DB.eval()
    # Df2.eval()

global_step = 0
for e in range(EPOCH):
    model_train()
    for i, (xA, xB) in enumerate(tqdm(train_loader)):
        xA, xB = xA.cuda(), xB.cuda()

        '''
        Update D
        1. Df1 <-- c_A[1] | c_B[1]
        2. Df2 <-- c_A[2] | c_B[2]
        3. Di <-- (s_A, c_B) | xA

        '''
        
        dxA = copy.deepcopy(xA)
        dxB = copy.deepcopy(xB)

        # Extract
        cA, sA = GA.module.encode(dxA)
        warped_cloth, _ = warp_model(dxA, dxB)
        cB, sB = GB.module.encode(warped_cloth)

        # Transfer
        xAB = GB.module.decode(cA, sB)
        xBA = GA.module.decode(cB, sA)

        # feed D
        dBA = DA(xBA) # fake A
        dA = DA(dxA) # real A

        dAB = DB(xAB) # fake B
        dB = DB(dxB) # real B

        # Gradient Penalty
        dA_penalty = gradient_penalty(DA, dA, dBA) # --     ERROR ------------
        dB_penalty = gradient_penalty(DB, dB, dAB)

        # ---------------- Loss --------------#
        dA_loss = dBA.mean() - dA.mean() + GP*dA_penalty
        dB_loss = dAB.mean() - dB.mean() + GP*dB_penalty

        d_loss = dA_loss + dB_loss
        D_optim.zero_grad()
        d_loss.backward()
        D_optim.step()
        
        '''
        G Update
        '''
        # ---------------- 1'st --------------#
        # Extract
        cA, sA = GA.module.encode(xA)
        warped_cloth, _ = warp_model(xA, xB)
        cB, sB = GB.module.encode(warped_cloth)

        # AutoEncoder
        xAA = GA.module.decode(cA, sA)
        xBB = GB.module.decode(cB, sB)

        # Transfer
        xAB = GB.module.decode(cA, sB)
        xBA = GA.module.decode(cB, sA)

        # ---------------- 2'nd --------------#
        # Extract
        cAB, sAB = GB.module.encode(xAB)
        cBA, sBA = GA.module.encode(xBA)

        # Transfer
        xABA = GA.module.decode(cAB, sBA)
        xBAB = GB.module.decode(cBA, sAB)

        # ---------------- Loss --------------#
        '''
        1. D(xAB) - D(xBA)
        2. L1: (xABA, xA) - (xBAB, xB)
        3. L1: (xAA, xA) - (xBB, xB)
        4. L1: (sAB, sB) - (sBA, sA)
        5. L1: (cAB, cA) - (cBA, cB)
        6. D(cAB[...]) - D(cBA[...]) -- Custom
        '''
        g_loss_fake = torch.mean(-DB(xAB)) + torch.mean(-DA(xBA))
        loss_cycle = l1(xABA, xA) + l1(xBAB, xB)
        loss_ae = l1(xAA, xA) + l1(xBB, xB)
        loss_s = l1(sAB, sB) + l1(sBA, sA)
        loss_c = l1(cAB, cA) + l1(cBA, cB)

        g_loss = GD*g_loss_fake + GCYCLE*loss_cycle + GAE*loss_ae + GS*loss_s + GC*loss_c
        G_optim.zero_grad()
        g_loss.backward()
        G_optim.step()

        # Plot
        writer.add_scalar('Epoch', e, global_step)
        writer.add_scalar('d_loss', d_loss.item(), global_step)
        writer.add_scalar('di_loss', di_loss.item(), global_step)
        writer.add_scalar('di_loss', df1_loss.item(), global_step)
        writer.add_scalar('di_loss', df2_loss.item(), global_step)

        writer.add_scalar('g_loss', g_loss.item(), global_step)
        writer.add_scalar('gi_loss', GI*gi_loss.item(), global_step)
        writer.add_scalar('gf1_loss', GF*gf1_loss.item(), global_step)
        writer.add_scalar('gf2_loss', GF*gf2_loss.item(), global_step)
        writer.add_scalar('ri_loss', RI*ri_loss.item(), global_step)
        writer.add_scalar('rf_loss', RF*rf_loss.item(), global_step)
        
        global_step += 1

        # Test
        if global_step % 100 == 0:
            output = torch.cat((train_data.invtrans(xA), train_data.invtrans(xAA), train_data.invtrans(xB),\
                                train_data.invtrans(warped_cloth), train_data.invtrans(xBB), train_data.invtrans(xAB), train_data.invtrans(xABA)), dim = 2)
            torchvision.utils.save_image(output, f"outputs/{global_step}.png")

    save_model(e)
    # Update learning rate
    D_lr_sche.step()
    G_lr_sche.step()
    # warp_lr_sche.step()
    
