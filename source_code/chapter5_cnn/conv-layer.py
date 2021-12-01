import torch
from torch import nn
from d2l import torch as d2l

# 自定义卷积运算
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
Y = corr2d(X, K)
print(Y)


# 自定义卷积操作
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

X = torch.ones((6, 8))
print(X)
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1.0, -1.0]])
print(K)
Y = corr2d(X, K)
print(Y)

# 学习卷积核
net = nn.Conv2d(1, 1, kernel_size=(1,2), bias=False)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(30):
    Y_hat = net(X)
    l = (Y_hat - Y)**2    # l shape: (1, 1, 6, 7)
    net.zero_grad()
    l.sum().backward()    # 因此需要执行sum()操作
    net.weight.data[:] -= 3e-2 * net.weight.grad
    if (i+1)%2 == 0:
        print(f'batch {i+1}, loss {l.sum():.3f}')



