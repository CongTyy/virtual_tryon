from lib import *
from model import G_Generator, T_Generator, Discriminator
from dataloader import Loader
from utils import *

shutil.rmtree('outputs')
os.mkdir('outputs')

writer = SummaryWriter(f"logs")
t_model = T_Generator().to(device)
g_model = G_Generator().to(device)
d_model = Discriminator().to(device)

t_model.load_state_dict(torch.load("tryons/tryon.pth"))
g_model.load_state_dict(torch.load("garments/garment.pth"))

t_optim = optim.Adam(t_model.parameters(), lr = 0.1*lr, betas=betas)
g_optim = optim.Adam(g_model.parameters(), lr = 0.1*lr, betas=betas)
d_optim = optim.Adam(d_model.parameters(), lr = lr, betas=betas)

train_data = Loader()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

# d_scaler = torch.cuda.amp.GradScaler()
# tryon_scaler = torch.cuda.amp.GradScaler()
# garment_scaler = torch.cuda.amp.GradScaler()

l1 = nn.L1Loss()
l2 = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
def update(optim, loss):
    optim.zero_grad()
    loss.backward()
    optim.step()


for e in range(1, EPOCH):
    d_arr = []
    ct_arr = []
    st_arr = []
    cg_arr = []
    rg_arr = []
    rt_arr = []
    g_arr = []
    for i, (p_img, g_img) in enumerate(tqdm(train_loader)):
        p_img = p_img.to(device)
        g_img = g_img.to(device)


        #   Update D
        d_real = d_model(p_img)
        si, x1, x2, x3 = t_model(p_img, s = True)
        cg = g_model(g_img, c=True)

        temp = torch.cat((cg, si), dim = 1)        
        _tryon = t_model(temp, g = True, x1 = x1, x2 = x2, x3 = x3)
        d_tryon = d_model(_tryon.detach())
        gp = gradient_penalty(d_model, p_img, _tryon)
        # Maximize D
        d_loss = - (torch.mean(d_real) - torch.mean(d_tryon)) + gp*lambda_gp
        d_arr.append(d_loss)
        # print(f'D_loss: {d_loss:.4f} {torch.mean(d_real)} {torch.mean(d_tryon)}')
        update(d_optim, d_loss)

        
        ##############################################################################
        #   Update Extractor - Tryonmodel
        t_optim.zero_grad()
        si, x1, x2, x3 = t_model(p_img, s = True)
        cg = g_model(g_img, c=True)
        temp = torch.cat((cg, si), dim = 1)  
        _tryon = t_model(temp, g = True, x1 = x1, x2 = x2, x3 = x3)
        _tryon = _tryon.detach()

        _ci = t_model(_tryon, c = True)
        ci_loss = l1(_ci, cg) * F_BIAS
        ct_arr.append(ci_loss)
        # update(t_optim, ci_loss)
        ci_loss.backward()

        _si, x1, x2, x3 = t_model(_tryon, s = True)
        s_loss = l1(_si, si) * F_BIAS
        st_arr.append(s_loss)
        # update(t_optim, s_loss)
        s_loss.backward()
        t_optim.step()

        # print(f'c_loss: {ci_loss:.4f}\ts_loss: {s_loss:.4f}')
        ##############################################################################
        #   Update Extractor - Garmentmodel     
        ci = t_model(p_img, c = True)
        ci = ci.detach()
        cg = g_model(g_img, c=True)

        c_loss = l1(cg, ci) * F_BIAS
        cg_arr.append(c_loss)
        update(g_optim, c_loss)
        # print(f'C_loss_garment: {c_loss:.4f}')

        # Update Generator - Garment
        cg = g_model(g_img, c=True)
        cg = cg.detach()

        _gar = g_model(cg, g = True)
        gar_loss = l1(_gar, g_img) * F_BIAS
        rg_arr.append(gar_loss)
        update(g_optim, gar_loss)
        # print(f'Garment loss: {gar_loss:.4f}')

        if i % 5 == 0:
            ##############################################################################
            #   Update Tryon Generator
            t_optim.zero_grad()
            si, x1, x2, x3 = t_model(p_img, s = True)
            cg = g_model(g_img, c=True)
            temp = torch.cat((cg, si), dim = 1)  
            _tryon = t_model(temp, g = True, x1 = x1, x2 = x2, x3 = x3)
            _tryon = _tryon.detach()

            _ci = t_model(_tryon, c = True)
            ci_loss = l1(_ci, cg) * F_BIAS
            # ct_arr.append(ci_loss)
            # update(t_optim, ci_loss)
            ci_loss.backward()

            _si, x1, x2, x3 = t_model(_tryon, s = True)
            s_loss = l1(_si, si) * F_BIAS
            # st_arr.append(s_loss)
            # update(t_optim, s_loss)
            s_loss.backward()

            # Update Consistency Fist
            # si, x1, x2, x3 = t_model(p_img, s = True)
            si = si.detach()
            ci = t_model(p_img, c = True)
            ci = ci.detach()
            temp = torch.cat((ci, si), dim = 1)

            _p_img = t_model(temp, g = True, x1 = x1, x2 = x2, x3 = x3)
            consis_loss = l1(_p_img, p_img) * F_BIAS # 1.
            rt_arr.append(consis_loss)
            consis_loss.backward()

            cg = g_model(g_img, c=True)
            cg = cg.detach()

            temp = torch.cat((cg, si), dim = 1)
            _tryon = t_model(temp, g = True, x1 = x1.detach(), x2 = x2.detach(), x3 = x3.detach())
            d_gen = d_model(_tryon)
            gen_loss = -torch.mean(d_gen)
            g_arr.append(gen_loss)
            gen_loss.backward()

            t_optim.step()

        writer.add_scalar('D_loss', d_loss, (e-1)*len(train_data)+i*batch_size)
        writer.add_scalar('CT_loss', ci_loss, (e-1)*len(train_data)+i*batch_size)
        writer.add_scalar('ST_loss', s_loss, (e-1)*len(train_data)+i*batch_size)
        writer.add_scalar('CG_loss', c_loss, (e-1)*len(train_data)+i*batch_size)
        writer.add_scalar('RG_loss', gar_loss, (e-1)*len(train_data)+i*batch_size)
        writer.add_scalar('RT_loss', consis_loss, (e-1)*len(train_data)+i*batch_size)
        writer.add_scalar('G_loss', gen_loss, (e-1)*len(train_data)+i*batch_size)

    torchvision.utils.save_image(torch.cat((train_data.invtrans(_tryon).squeeze(), 
                                                    torch.cat((train_data.invtrans(p_img), train_data.invtrans(g_img)), dim = 2)), dim = 2), f"outputs/image{e}.jpg")
    torchvision.utils.save_image(torch.cat((train_data.invtrans(_gar.squeeze()), train_data.invtrans(g_img)), dim = 2), f"outputs/garment{e}.jpg")
    torchvision.utils.save_image(torch.cat((train_data.invtrans(_p_img).squeeze(), train_data.invtrans(p_img)), dim = 2), f"outputs/person{e}.jpg")


    torch.save(t_model.state_dict(), f"weights/tryon_{e}.pth")
    torch.save(g_model.state_dict(), f"weights/garment_{e}.pth")
    torch.save(d_model.state_dict(), f"weights/discriminator_{e}.pth")





