import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import torch.nn.functional as F
import visdom
import numpy as np
import matplotlib.pyplot as plt
import random

# 设置超参数
h_dim = 400
batch_size = 512
viz = visdom.Visdom()

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.net = nn.Sequential(nn.Linear(2,h_dim),
                                 nn.ReLU(True),
                                 nn.Linear(h_dim,h_dim),
                                 nn.ReLU(True),
                                 nn.Linear(h_dim,h_dim),
                                 nn.ReLU(True),
                                 nn.Linear(h_dim,2))
    def forward(self, z):
        output = self.net(z)
        return output

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.net = nn.Sequential(nn.Linear(2,h_dim),
                                 nn.ReLU(True),
                                 nn.Linear(h_dim,h_dim),
                                 nn.ReLU(True),
                                 nn.Linear(h_dim,h_dim),
                                 nn.ReLU(True),
                                 nn.Linear(h_dim,1),
                                 nn.Sigmoid())

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)

def data_generator():
    scale = 2.
    centers = [(1,0),
               (-1,0),
               (0,1),
               (0,-1),
               (1./np.sqrt(2),1./np.sqrt(2)),
               (1./np.sqrt(2),-1./np.sqrt(2)),
               (-1./np.sqrt(2),1./np.sqrt(2)),
               (-1./np.sqrt(2),-1./np.sqrt(2)),]

    centers = [(scale*x,scale*y) for x,y in centers]

    while True:
        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2)*0.02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)

        dataset = np.array(dataset,dtype='float32')
        dataset /= 1.414
        yield dataset

def generate_image(D,G,xr,epoch):
    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS,N_POINTS,2),dtype='float32')
    points[:,:,0] = np.linspace(-RANGE,RANGE,N_POINTS)[:,None]
    points[:,:,1] = np.linspace(-RANGE,RANGE,N_POINTS)[None,:]
    points = points.reshape((-1,2))

    with torch.no_grad():
        points = torch.Tensor(points).cuda()
        disc_map = D(points).cpu().numpy()

    x = y = np.linspace(-RANGE,RANGE,N_POINTS)
    cs = plt.contour(x,y,disc_map.reshape((len(x),len(y))).transpose())
    plt.clabel(cs,inline=1,fontsize=10)

    with torch.no_grad():
        z = torch.randn(batch_size,2).cuda()
        samples = G(z).cpu().numpy()

    plt.scatter(xr[:,0],xr[:,1],c='red',marker='.')
    plt.scatter(samples[:,0],samples[:,1],c='green',marker='+')

    viz.matplot(plt,win='contour',opts=dict(title='p(x):%d'%epoch))


def weights_init(m):
    if isinstance(m,nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def gradient_penalty(D,xr,xf):

    LAMBDA = 0.3

    xf = xf.detach()
    xr = xr.detach()

    alpha = torch.rand(batch_size,1).cuda()
    alpha = alpha.expend_as(xr)

    interpolates = alpha * xr + ((1-alpha) * xf)
    interpolates.requires_grad()

    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates,
                              inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    gp = ((gradients.norm(2,dim=1) -1)**2).mean()*LAMBDA

    return gp

def main():
    torch.manual_seed(23)
    np.random.seed(23)

    G = Generator().cuda()
    D = Discriminator().cuda()
    G.apply(weights_init)
    D.apply(weights_init)

    optim_G = optim.Adam(G.parameters(),lr=1e-3,betas=(0.5,0.9))
    optim_D = optim.Adam(D.parameters(),lr=1e-3,betas=(0.5,0.9))

    data_iter = data_generator()
    print('batch:',next(data_iter).shape)

    viz.line([[0,0]],[0],win='loss',opts=dict(title='loss',legend=['D','G']))

    for epoch in range(50000):
        for _ in range(5):
            x = next(data_iter)
            xr = torch.from_numpy(x).cuda()

            predr = (D(xr))

            lossr = -(predr.mean())

            z = torch.randn(batch_size,2).cuda()
            xf = G(z).detach()

            predf = (D(xf))

            lossf = (predf.mean())

            gp = gradient_penalty(D,xr,xf)

            loss_D = lossr + lossf + gp
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        z = torch.randn(batch_size,2).cuda()
        xf = G(z)
        predf = (D(xf))

        loss_G = - (predf.mean())
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            viz.line([[loss_D.item(),loss_G.item()]],[epoch],win='loss',update='append')

            generate_image(D,G,xr,epoch)

            print(loss_D.item(),loss_G.item())


if __name__ == '__main__':
    main()



