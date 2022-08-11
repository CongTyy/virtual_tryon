from lib import *
from model import C_encoder, G_Generator, I_Generator
from dataloader import Loader
from utils import *
from metrics.metric_folder import calculate_fid_given_paths

shutil.rmtree('outputs')
shutil.rmtree('garments')
shutil.rmtree('tryons')

os.mkdir('outputs')
os.mkdir('garments')
os.mkdir('tryons')

writer = SummaryWriter(f"logs")
t_model = I_Generator().to(device)
g_model = G_Generator().to(device)
c_model = C_encoder().to(device)
# t_model.load_state_dict(torch.load("weights_reconstruct/tryon.pth"))
# g_model.load_state_dict(torch.load("weights_reconstruct/garment.pth"))


t_optim = optim.Adam(t_model.parameters(), lr = lr, betas=betas)
g_optim = optim.Adam(g_model.parameters(), lr = lr, betas=betas)
c_optim = optim.Adam(c_model.parameters(), lr = lr, betas=betas)

train_data = Loader()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

l1 = nn.L1Loss()
l2 = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
def update(optim, loss):
    optim.zero_grad()
    loss.backward()
    optim.step()


temp_iloss = 100
temp_gloss = 100


for e in range(1, EPOCH):
    t_loss = []
    g_loss = []
    for i, (p_img, g_img, p_pth, g_pth) in enumerate(tqdm(train_loader)):
        p_img = p_img.to(device)
        g_img = g_img.to(device)

        t_optim.zero_grad()
        c_optim.zero_grad()
        g_optim.zero_grad()

        t_optim.zero_grad()
        c_optim.zero_grad()
        c1, c2, c3 = c_model(p_img)
        _p_img = t_model(p_img, c1, c2, c3)
        rimg_loss = l1(_p_img, p_img)
        rimg_loss.backward()
        t_optim.step()
        c_optim.step()

        t_loss.append(rimg_loss.item())

        
        c_optim.zero_grad()
        g_optim.zero_grad()
        c1, c2, c3 = c_model(g_img)
        _g_img = g_model(c1, c2, c3)
        rgar_loss = l1(_g_img, g_img)
        rgar_loss.backward()
        g_optim.step()
        c_optim.step()

        g_loss.append(rgar_loss.item())


        writer.add_scalar('Tryon', rimg_loss, (e-1)*len(train_data)+i*batch_size)
        writer.add_scalar('Garment', rgar_loss, (e-1)*len(train_data)+i*batch_size)

    
    print(f'Ri_loss: {np.mean(t_loss):.4f}\nRg_loss: {np.mean(g_loss):.4f}')
    torchvision.utils.save_image(torch.cat((train_data.invtrans(_g_img).squeeze(), train_data.invtrans(g_img)), dim = 2), f"outputs/garment{e}.jpg")
    torchvision.utils.save_image(torch.cat((train_data.invtrans(_p_img).squeeze(), train_data.invtrans(p_img)), dim = 2), f"outputs/person{e}.jpg")

    shutil.rmtree('FID')

    os.mkdir('FID')
    os.mkdir('FID/g_fake')
    os.mkdir('FID/g_real')
    os.mkdir('FID/p_fake')
    os.mkdir('FID/p_real')

    for k in range(batch_size):
        torchvision.utils.save_image(train_data.invtrans(_g_img[k]), f"FID/g_fake/garment{k}.jpg")
        torchvision.utils.save_image(train_data.invtrans(g_img[k]), f"FID/g_real/garment{k}.jpg")
        torchvision.utils.save_image(train_data.invtrans(_p_img[k]), f"FID/p_fake/person{k}.jpg")
        torchvision.utils.save_image(train_data.invtrans(p_img[k]), f"FID/p_real/person{k}.jpg")

    g_fid = calculate_fid_given_paths(paths = ['FID/g_fake', 'FID/g_real'], batch_size=batch_size, device='cuda', num_workers=8)
    p_fid = calculate_fid_given_paths(paths = ['FID/p_fake', 'FID/p_real'], batch_size=batch_size, device='cuda', num_workers=8)
    writer.add_scalar('p_fid', p_fid, e)
    writer.add_scalar('g_fid', g_fid, e)

    torch.save(t_model.state_dict(), f"tryons/tryon_{e}_{p_fid:.4f}.pth")
    torch.save(g_model.state_dict(), f"garments/garment_{e}_{g_fid:4f}.pth")
