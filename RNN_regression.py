import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# 1.设置超参数
time_step = 10  # rnn time step
input_size = 1 # rnn input size
LR = 1e-3

# 2.生成数据
# steps = np.linspace(0,np.pi*2,100,dtype=np.float32)
# x_np = np.sin(steps)
# y_np = np.cos(steps)
# plt.plot(steps,y_np,'r-',label='target(cos)')
# plt.plot(steps,x_np,'b-',label='input(sin)')
# plt.legend(loc='best')
# plt.show()


class Rnn(nn.Module):
    def __init__(self):
        super(Rnn,self).__init__()

        self.rnn = nn.RNN(
            input_size = input_size,
            hidden_size= 32,
            num_layers= 2,
            batch_first=True,
        )

        self.out = nn.Linear(32,1)

    def forward(self,x,h_state):
        # x shape (batch,time_step,input_size)
        # h_state (n_layers,batch,hidden_size)
        # r_out (batch,time_step,hidden_size)
        r_out,h_state = self.rnn(x,h_state)

        outs = []
        for time_step in range(r_out.size(1)): # calculate output for each time step
            outs.append(self.out(r_out[:,time_step,:]))
        return torch.stack(outs,dim=1),h_state

rnn = Rnn()
optimizer = optim.Adam(rnn.parameters(),lr=LR)
criterion = nn.MSELoss()

h_state = None
plt.figure(1,figsize=(12,5))
plt.ion()
for step in range(100):
    start,end = step*np.pi,(step+1)*np.pi   # time range
    # use sin predict cos
    steps = np.linspace(start,end,time_step,dtype=np.float32,endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis,:,np.newaxis])  # shape(batch,time_step,input_size)
    y = torch.from_numpy(y_np[np.newaxis,:,np.newaxis])

    prediction,h_state = rnn(x,h_state)

    h_state = h_state.data # repack the hidden state,break the connection from last iteration

    loss = criterion(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ploting
    plt.plot(steps,y_np.flatten(),'r-')
    plt.plot(steps,prediction.data.numpy().flatten(),'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()

