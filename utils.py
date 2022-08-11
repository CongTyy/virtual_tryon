from lib import *


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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'ConvBlock':
        torch.nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.2)

    # elif classname.find("InstanceNorm2d") != -1:
    #     torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     torch.nn.init.constant_(m.bias.data, 0.0)


def save_model(a, b, c, op, path):
    if c is not None:
        torch.save({
                'epc_state_dict': a.state_dict(),
                'eps_state_dict': b.state_dict(),
                'gp_state_dict': c.state_dict(),
                'recons_img_optim_state_dict': op.state_dict(),
                }, path)
    # else:
