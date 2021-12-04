
import torch
from torch import nn

# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    print(X.shape)
    X = X.reshape((1, 1) + X.shape)   # 将维度从[8, 8]  --> [1, 1, 8, 8]
    print(X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    shape_tmp = Y.shape[2:]
    return Y.reshape(shape_tmp)     # 将维度从[1, 1, 8, 8]  --> [8, 8]

# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)


# 步幅相同，结论：当kernel_size - padding = 1的条件下，当stride为2时，feature map减为原来的一半，注意：此处的padding为参数padding的2倍，因为上下左右都有padding
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

#步幅不同
conv2d = nn.Conv2d(1, 1, kernel_size=(3,5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)

'''
结论：
    1、计算公式：
        floor[(h - k + p + s)/s] * floor[(w - k + p + s)/s]
    2、当p=k-1时：
        floor[(h + s - 1)/s] * floor[(w + s - 1)/s]
        更进一步，当s = 1时：
            [h] * [w]
    3、如果输入的高度和宽度可以被垂直和水平步幅整除：
        [h/s] * [w/s]
        达到步幅了衰减s和s倍的目的
'''

