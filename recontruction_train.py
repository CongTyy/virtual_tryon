from lib import *
from model import  Discriminator, Extractor, Generator, Reconstructor
from dataloader import Loader
from utils import *

device = 'cuda'
batch_size = 64


r_model = Reconstructor().to(device)
r_model.train()
d_model = Discriminator().to(device)
d_model.train()

train_data = Loader()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

consistancy_loss = nn.L1Loss()
bce_loss = nn.BCELoss()
 
lr = 1e-4
betas = (0.5, 0.909)

r_optimizer = optim.Adam(r_model.parameters(), lr = lr, betas=betas)
d_optimizer = optim.Adam(d_model.parameters(), lr = lr, betas=betas)
gen_loss =[]
dis_loss =[]

critic_iter = 2
one = torch.tensor(1, dtype=torch.float).to(device)
mone = (one * -1).to(device)
losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}


for e in range(EPOCH):
    genloss_epoch = []
    disloss_epoch = []
    for idx, data in enumerate(tqdm(train_loader)):
        # Requires grad, Generator requires_grad = False
        # for p in d_model.parameters():
        #     p.requires_grad = True

        # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update

        p_img = Variable(data).to(device)

        # Train discriminator
        # WGAN - Training discriminator more iterations than generator

        # Train with real images
        d_real = d_model(p_img)

        # Train with fake images
        fake_images = r_model(p_img)
        d_fake = d_model(fake_images)

        # Train with gradient penalty
        gradient_penalty = calculate_gradient_penalty(d_model, p_img.data, fake_images.data, losses)
        losses['GP'].append(gradient_penalty.data)

        d_model.zero_grad()
        d_loss = d_fake.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        d_optimizer.step()
        losses['D'].append(d_loss.data)

        # Generator update
        if idx % 5 == 0:
            # for p in d_model.parameters():
                # p.requires_grad = False  # to avoid computation
            r_model.zero_grad()
            fake_images = r_model(p_img)
            d_fake = d_model(fake_images)
            c_loss = consistancy_loss(fake_images, p_img)
            g_loss = d_fake.mean() + c_loss.mean()
            g_loss.backward()
            r_optimizer.step()
            losses['G'].append(g_loss.data)

        if idx % 50 == 0:
            print("Iteration {}".format(idx + 1))
            print("D: {}".format(losses['D'][-1]))
            print("GP: {}".format(losses['GP'][-1]))
            print("Gradient norm: {}".format(losses['gradient_norm'][-1]))
            if idx > 5:
                print("G: {}".format(losses['G'][-1]))
            torchvision.utils.save_image(torch.cat((fake_images.squeeze(), p_img), dim = 2), f"outputs/image.jpg")

            
    print("-----------------")
    # gen_loss.append(np.mean(genloss_epoch))
    # dis_loss.append(np.mean(disloss_epoch))
    # print(f'{e}/{EPOCH} - GLOSS:{np.mean(genloss_epoch)}\tDLOSS:{np.mean(disloss_epoch)}')
    if e % 50 == 0:
        torch.save(r_model.state_dict(), f"weights/r_model_{e}.pth")

plt.plot(gen_loss)
plt.plot(dis_loss)
plt.savefig("loss.jpg")