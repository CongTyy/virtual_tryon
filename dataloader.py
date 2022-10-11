from lib import *

class Loader(Dataset):
    def __init__(self, p_path = "VITON_traindata/train_img", g_path = "VITON_traindata/train_color",\
        image_size = (256, 192), trasforms = True):
        super(Loader, self).__init__()
        self.image_size = image_size

        self.p_imgs = glob(f'{p_path}/*')
        self.g_imgs = glob(f'{g_path}/*')

        self._transform = self.transform()

        self.invtrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                            transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],std = [ 1., 1., 1. ]),])
    


    def __len__(self):
        return len(self.p_imgs)

    def __getitem__(self, idx):
        # print(self.g_imgs[idx])
        # print(self.g_imgs[idx])
        p_img = Image.open(self.p_imgs[idx])
        g_img = Image.open(self.g_imgs[idx])
        p_img = self._transform(p_img)
        g_img = self._transform(g_img)
        return p_img, g_img


    def transform(self):
        # options = []
        # options.append(transforms.ToTensor())
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        ])

        return transform

    def inv_Trasform(self, tensor):
        return self.invtrans(tensor)

if __name__ == "__main__":
    train_data = Loader()
    batch_size = 4
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # functions to show an image


    def imshow(img):
        # img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(train_loader)
    p, g = dataiter.next()
    torchvision.utils.save_image(g.squeeze(), "image.jpg")


    invTrans  = transforms.Compose([    transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                        transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    
    # inv = 
    torchvision.utils.save_image(train_data.invtrans(g).squeeze(), "invimage.jpg")

    
    # print(images.size())
    # cv2.imwrite("image.jpg", images.squeeze().cpu().numpy())
    # imshow(torchvision.utils.make_grid(images))
