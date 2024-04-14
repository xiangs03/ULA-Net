import torch
import torch.nn as nn
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current Device: {DEVICE}. Torch Version: {torch.__version__}')


class Swish(nn.Module):
    """
    Swish activation function.
    f(h) = h * sigmoid(beta*h)
    """
    def __init__(self, beta=1.):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class GatedLinearUnit(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.gate = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))


class UnidirectionalAttention(nn.Module):
    """
    单向的多头注意力机制网络，设头为M个，
    首先将输入展平成(B, L, K×K)，
    然后每个vector全连接得到一个key，key∈(B, Dk, K×K),
    用Channel Mask后全连接得到value，val∈(B, Dv, K×K)
    取中间的目标像元的key作为query,得que∈(B, Dk, 1),
    采用多头，每个小key就被分割为skey∈(B, Dk//M, K×K),
    sque同理，为(B, Dk//M, 1),
    接着用sque去对skey做atten, K^T*Q/sqrt(Dk) 再softmax 得到 (B, K×K, 1)=atten,
    sval属于(B, Dv//M, K×K)，sval*atten∈(B, Dv//M, 1)，
    每个头注意力一下拼接就得到了(B, Dv, 1)，作为提取的空间特征
    注意：这时光谱信息的value还没有被加进去，因为做attention时故意遮盖了光谱信息，
    所以空间特征要加上一个key的中间值(B, Dv, 1)，才能成为空-谱信息。
    """
    def __init__(self, L, K, n_head=4, val_dim=128, key_dim=128):
        super().__init__()
        self.L, self.K = L, K
        assert n_head == 0 or (val_dim % n_head == 0 and key_dim % n_head == 0)
        self.head_num = n_head
        n_head = 1 if n_head == 0 else n_head
        self.val_split = val_dim // n_head
        self.key_split = key_dim // n_head
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.seq_len = K * K
        self.seq_center = self.seq_len // 2
        self.central_mask = torch.zeros(1, 1, self.seq_len).to(DEVICE)
        self.central_mask[..., self.seq_center] = -1e6
        self.softmax = nn.Softmax(dim=2)

        self.w_key = nn.Linear(L, key_dim)
        self.w_val = nn.Linear(L, val_dim)
        self.w_que = nn.Linear(L, key_dim)

        self.nearby_gate = nn.Sequential(nn.Linear(val_dim, 1), nn.Sigmoid())
        self.nearby = torch.tensor(0.).to(DEVICE)
        self._init_weight()

    def _init_weight(self):
        for name, param in self.named_parameters():
            if len(param.shape) == 2:
                stdv = 1.0 / param.shape[1] ** 0.5
                torch.nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x):
        """
        :param x: x的shape必须是(B, L, K, K), 是一个patch
        """
        x = x.view(x.shape[0], x.shape[1], self.K**2).transpose(1, 2)  # (B, KK, L)
        keys = self.w_key(x)  # (B, KK, Dk)
        vals = self.w_val(x)  # (B, KK, Dv)
        query = self.w_que(x[:, self.seq_center:self.seq_center+1])  # (B, 1, Dk)

        attended_val = []
        for head_idx in range(self.head_num):
            small_key = keys[..., head_idx * self.key_split: (head_idx+1) * self.key_split]
            small_query = query[..., head_idx * self.key_split: (head_idx+1) * self.key_split]

            small_atten_weight = small_query @ small_key.transpose(1, 2) / np.sqrt(self.key_dim)
            small_atten = self.softmax(small_atten_weight+self.central_mask)

            small_val = vals[..., head_idx * self.val_split: (head_idx+1) * self.val_split]
            attended_val.append(small_atten @ small_val)

        central_val = vals[:, self.seq_center:self.seq_center+1]

        if self.head_num == 0:
            return central_val

        surround_val = torch.cat(attended_val, dim=2)
        self.nearby = self.nearby_gate(central_val - surround_val)

        return central_val + self.nearby * surround_val


class RuaAE(nn.Module):
    def __init__(self, L, P, K, edm, model_dim=128, n_head=4, scale_range=0.2):
        super().__init__()
        self.model_dim = model_dim
        self.L, self.P, self.K = L, P, K
        self.scale_range = scale_range
        self.endmember = nn.Parameter(edm)

        self.uda = UnidirectionalAttention(L, K,
                                           n_head=n_head,
                                           key_dim=model_dim,
                                           val_dim=model_dim,
                                           )

        self.ffn = nn.Sequential(     # feed forward network
            Swish(),
            GatedLinearUnit(model_dim, 64),
            nn.BatchNorm1d(64),
            Swish(),
            GatedLinearUnit(64, P),
            nn.Softmax(dim=1),
        )
        self.scalar = nn.Sequential(  # for scaling endmembers
            Swish(),
            GatedLinearUnit(model_dim, 64),
            nn.BatchNorm1d(64),
            Swish(),
            GatedLinearUnit(64, P),
            nn.Tanh(),
        )

        self._init_weight()

    def _init_weight(self):
        for name, param in self.named_parameters():
            if len(param.shape) == 2:  # 如果是Linear
                stdv = 1.0 / param.shape[1] ** 0.5
                torch.nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x, return_abd=False):

        ss_feat = self.uda(x).squeeze(1)

        abd = self.ffn(ss_feat).unsqueeze(1)

        edm_scale = self.scalar(ss_feat)
        scale = 1 + self.scale_range * edm_scale.unsqueeze(-1)
        edm = scale * self.endmember
        # edm = self.endmember

        x = abd @ edm
        x = x.squeeze(1)

        return x if return_abd else x, abd.squeeze(1), edm_scale
