import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# 1.定义超参数
epochs = 10         # train the training data n times
batch_size = 64
time_step = 28      # rnn:time step -->image height
input_size = 28     # rnn:input size -->image width
LR = 1e-3

# 2.加载数据
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
# test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)
test_x = test_data.data.type(torch.FloatTensor)[:2000]/255  # shape(2000,28,28),value in (0,1)
test_y = test_data.targets.numpy()[:2000]  # convert into numpy array
# print('test_y:',test_y)

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        self.rnn = nn.LSTM(             # if use nn.RNN(),it hardly learn
            input_size = input_size,    # iamge width
            hidden_size = 64,
            num_layers = 2,             # number of rnn layer
            batch_first=True,           # input & output will has batch size as first dimension e.g.(batch,time_step,input_size)
        )

        self.out = nn.Linear(64,10)

    def forward(self,x):
        # x shape (batch,time_step,input_size)
        # r_out shape(batch,time_step,output_size)
        # h_n shape(n_layers,batch,hidden_size)
        # h_c shape(n_layers,batch,hidden_size)
        r_out,(h_n,h_c) = self.rnn(x,None) # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:,-1,:])
        return out

rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)
criterion = nn.CrossEntropyLoss()

# training and testing
for epoch in range(epochs):
    for step,(data,target) in enumerate(train_loader):
        data = data.view(-1,28,28)

        output = rnn(data)
        loss = criterion(output,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            test_output = rnn(test_x)   # (samples,time_step,input_size)
            pred_y = torch.max(test_output,1)[1].data.numpy()
            acc = float((pred_y == test_y.data).astype(int).sum()) / float(test_y.size)
            print('Epoch {} |Train loss {:.5f} | Test acc{:.2f}'.format(epoch,loss.data.numpy(),acc))

test_output = rnn(test_x[:10].view(-1,28,28))
pred_y = torch.max(test_output,1)[1].data.numpy()
print('prediction:',pred_y)
print('real number:',test_y[:10])

