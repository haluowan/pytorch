import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# 设置超参数
epochs = 10
batch_size = 50
learning_rate = 1e-3

# 加载数据
# 设置随机数种子
torch.manual_seed(1)

train_data = datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True)

test_data = datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor())

train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

test_x = torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)[:2000]/255
# print('test_x:',test_x)

test_y = test_data.targets[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(    # input size (1,28,28)
            nn.Conv2d(
                in_channels=1, # input height 类似RGB
                out_channels=16, # n_filters
                kernel_size=5,   # filter size
                stride=1,    # filter movement/step
                padding=2,   # if want same width and length of this image after conv2d,padding = (kernel_size -1)/2 if stride=1
            ),
            nn.ReLU(),       # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2X2 area,output shape(16,14,14)
        )
        self.conv2 = nn.Sequential(       # input size(1,28,28)
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2), # output size(32,14,14)
            nn.ReLU(),                    # activation
            nn.MaxPool2d(kernel_size=2),  # output shape(32,7,7)
        )
        self.out = nn.Linear(32*7*7,10)   # fully connected layer,output 10 classes

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)         # flatten the output of conv2 to (batch_size,32*7*7)
        output = self.out(x)
        return output,x                  # return x

# 网络实例化
cnn = CNN()
# print(cnn)
optimizer = optim.Adam(cnn.parameters(),lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

# 模型训练
for epoch in range(epochs):
    for step,(data,target) in enumerate(train_loader):   # gives batch data,normalize x when iterate train_loader
    # data = data.view(-1,28*28)
        # print('data:',data[0])
        # print('target:',target[0])
        output = cnn(data)[0]
    # print('output:',output)
        # output = cnn(x)[0] # cnn output
        loss = loss_func(output,target)  # cross entropy loss
    # print('loss:',loss)
        optimizer.zero_grad() # clear gradients for this training step
        loss.backward() # backpropagation,compute gradients
        optimizer.step() # apply gradients
    #
        if step % 100 ==0:
            test_output,last_layer = cnn(test_x)
            pred_y = torch.max(test_output,1)[1].data.numpy()
            acc = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch:{} |train loss:{:.4f} |test acc:{:.2f}'.format(epoch,loss.item(),acc))
            # print('Train Epoch:{} [{}/{} ({:.0f}%)] \tLoss:{:.6f}'.format(
            #     epoch,step * len(data),len(train_loader.dataset),
            #     100.* step / len(train_loader),loss.item()))
    # test_loss = 0
    # correct = 0
    # for data,target in test_loader:
    #     data = data.view(-1,28*28)
    #     logits = cnn(data)
    #     test_loss += loss_func(logits,target).item()
    #
    #     pred = logits.data.max(1)[1]
    #     correct += pred.eq(target.data).sum()
    #
    # test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
# 10 prediction from test data
test_output,_ = cnn(test_x[:10])
pred_y = torch.max(test_output,1)[1].numpy().squeeze()
print('prediction number:',pred_y)
print('real number:',test_y[:10].numpy())





