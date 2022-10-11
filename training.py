import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms

class My_data(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.inp = torch.rand(20, 1)
        self.label = torch.rand(20, 1)
        # self.transform =  transforms.Compose([
        #     transforms.ToTensor(),
            # transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        # ])
    def __getitem__(self, index):

        return self.inp[index], self.label[index]
        
    def __len__(self):
        return len(self.inp)

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(1, 1),
            nn.Linear(1, 1))
        
    def forward(self, x):
        return self.layer(x)

dataset = My_data()
dataloader = torch.utils.data.DataLoader(dataset, batch_size= 5, shuffle=True, drop_last=True)
model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

loss_func = nn.L1Loss()
Epoch = 100
loss_total = []

for e in range(Epoch):
    for i, (x, y) in enumerate(dataloader):

        output = model(x)
        loss = loss_func(output, y)
        
        loss_total.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(np.mean(loss_total))
    loss_total = []
    
# for name, param in model.named_parameters():
#     if name == 'linear':
#         name = 'conv'
# model.layer[1] = nn.Conv2d(3, 16, kernel_size=3, stride=1)
# print(model)