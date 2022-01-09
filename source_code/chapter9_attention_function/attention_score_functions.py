# 注意力评分函数
import math
import torch
from torch import nn
from d2l import torch as d2l

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上遮蔽元素来执行 softmax 操作"""
    # `X`: 3D张量, `valid_lens`: 1D或2D 张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 在最后的轴上，被遮蔽的元素使用一个非常大的负值替换，从而其 softmax (指数)输出为 0
        tmp = X.reshape(-1, shape[-1])
        X = d2l.sequence_mask(tmp, valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

tmp = torch.rand(2, 2, 4)
print(tmp)
ret = masked_softmax(tmp, torch.tensor([2, 3]))
print(ret)
ret = masked_softmax(tmp, torch.tensor([[1, 3], [2, 4]]))
print(ret)

# 加性注意力
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)      # 2x8
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)    # 20x8  
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)             # 8x1
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):      # valid_lens表示queries需要关注的keys-values对的个数
        queries = self.W_q(queries)    # 2x1x8
        keys = self.W_k(keys)          # 2x10x8
        # 在维度扩展后，
        # `queries` 的形状：(`batch_size`, 查询的个数, 1, `num_hidden`)
        # `key` 的形状：(`batch_size`, 1, “键－值”对的个数, `num_hiddens`)
        # 使用广播方式进行求和，形式为：(`batch_size`,  查询的个数, “键－值”对的个数, `num_hiddens`)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)    # queries 2x1x1x8    keys 2x1x10x8  ->  features 2x1x10x8
        features = torch.tanh(features)
        # `self.w_v` 仅有一个输出，因此从形状中移除最后那个维度。
        # `scores` 的形状：(`batch_size`, 查询的个数, “键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)    # 2x1x10x8 -> 2x1x10x1 -> 2x1x10
        self.attention_weights = masked_softmax(scores, valid_lens)    # 将会无用的部分设置为0
        # `values` 的形状：(`batch_size`, “键－值”对的个数, 值的维度)
        dout = self.dropout(self.attention_weights)
        res = torch.bmm(dout, values)    # 2x1x10 2x10x4   -> 2x1x4
        return  res

queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# `values` 的小批量数据集中，两个值矩阵是相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
res = attention(queries, keys, values, valid_lens)
print("\nres:\n", res)

d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')


# 缩放点积的注意力
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # `queries` 的形状：(`batch_size`, 查询的个数, `d`)
    # `keys` 的形状：(`batch_size`, “键－值”对的个数, `d`)
    # `values` 的形状：(`batch_size`, “键－值”对的个数, 值的维度)
    # `valid_lens` 的形状: (`batch_size`,) 或者 (`batch_size`, 查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置 `transpose_b=True` 为了交换 `keys` 的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens)

d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')