from lib import *
from dataloader import Loader
invtrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                    transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],std = [ 1., 1., 1. ]),])

def gradient_penalty(critic, real, fake, device="cuda:0"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]

    model.load_state_dict(checkpoint_new)
def load(model, checkpoint_path):
    checkppoint = torch.load(checkpoint_path)

def visualize_feature_map(feature, e = 0, n = 'g'):
    feature = np.sum(feature, 1)
    # for f in feature[0]:
    plt.imshow(feature[0])
    plt.savefig(f'visualize/{n}_{e}.png')
    plt.clf()
    # plt.show

# def save_img(print_list, name):
#     nrow = len(print_list)
#     img = torch.cat(print_list, dim=3)
#     img = img.permute(1,0,2,3).contiguous()
#     vutils.save_image(img.view(1,img.size(0),-1,img.size(3)).data, name, nrow=nrow, padding=0, normalize=True)



        
def save_img(imgs, name):
    list_img = []
    for im in imgs:
        list_img.append(invtrans(im))
    img = torch.cat(list_img, dim = 2)
    torchvision.utils.save_image(img, name)
