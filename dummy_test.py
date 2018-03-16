import torch
from torch.autograd import Variable
import torch.nn as nn
from net.CNN import CNN
import torch.optim as optim

train_x = Variable(torch.randn(100, 1, 32, 32))
train_y = Variable(torch.randn(100, 10))

cnn = CNN()
cnn.zero_grad()

output = cnn(train_x)
criterion = nn.MSELoss()
loss = criterion(output, train_y)
print(loss)

optimizer = optim.SGD(cnn.parameters(), lr=0.01)


for epoch in range(1000):
    optimizer.zero_grad()
    output = cnn(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print("%s MSE Loss: %s", epoch, loss)


