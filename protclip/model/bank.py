import torch
from torch import nn
import torch.nn.functional as F
import pdb

class AttrProtoBank(nn.Module):
    def __init__(self, proto_dim, proto_num, decay=0.99, eps=1e-5, temp=0.9):
        super().__init__()
        self.proto_dim = proto_dim
        self.proto_num = proto_num
        self.decay = decay
        self.eps = eps
        self.prototype = nn.Embedding(proto_num, proto_dim)
        self.temp = temp
        self.curr_temp = temp

    def forward(self, subtexts):
        """Params:
            subtexts: [bsz, subtext_num, dim]
        """
        flatten = subtexts.reshape(-1, self.proto_dim)
        dist = flatten @ self.prototype.weight.T
        soft_one_hot = F.gumbel_softmax(dist, tau=self.curr_temp, dim=1, hard=False)
        output = soft_one_hot @ self.prototype.weight #[bsz*subtext_num, dim]
        proto_idx = soft_one_hot.argmax(1) #[bsz*subtext_num]
        proto_loss = (output - flatten).abs().mean()
        self.dist = dist
        return output, proto_loss, proto_idx