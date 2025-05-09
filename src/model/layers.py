import torch
from torch import nn
import torch.nn.functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class CLIPLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (F.cross_entropy(logits_per_image.float(), labels) +
                      F.cross_entropy(logits_per_text.float(), labels)) / 2
        total_loss = total_loss.to(dtype=image_features.dtype)
        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CrossAttentionBlock(nn.Module):

    def __init__(self, hidden_dim, num_heads):
        super(CrossAttentionBlock, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_heads))
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _alpha_from_logits(self, logits, mask_row, mask_col):
        N, L1, L2, H = logits.shape
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col)

        logits = torch.where(mask_pair.bool(), logits, -float('inf') * torch.ones_like(logits))
        alpha = torch.softmax(logits, dim=2)
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row.bool(), alpha, torch.zeros_like(alpha))
        return alpha

    def _heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def forward(self, input1, mask1, input2, mask2):
        query1 = self._heads(self.query1(input1), self.num_heads, self.head_size)
        key1 = self._heads(self.key1(input1), self.num_heads, self.head_size)
        query2 = self._heads(self.query2(input2), self.num_heads, self.head_size)
        key2 = self._heads(self.key2(input2), self.num_heads, self.head_size)
        logits11 = torch.einsum('blhd, bkhd->blkh', query1, key1)
        logits12 = torch.einsum('blhd, bkhd->blkh', query1, key2)
        logits21 = torch.einsum('blhd, bkhd->blkh', query2, key1)
        logits22 = torch.einsum('blhd, bkhd->blkh', query2, key2)

        alpha11 = self._alpha_from_logits(logits11, mask1, mask1)
        alpha12 = self._alpha_from_logits(logits12, mask1, mask2)
        alpha21 = self._alpha_from_logits(logits21, mask2, mask1)
        alpha22 = self._alpha_from_logits(logits22, mask2, mask2)

        value1 = self._heads(self.value1(input1), self.num_heads, self.head_size)
        value2 = self._heads(self.value2(input2), self.num_heads, self.head_size)
        output1 = (torch.einsum('blkh, bkhd->blhd', alpha11, value1).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha12, value2).flatten(-2)) / 2
        output2 = (torch.einsum('blkh, bkhd->blhd', alpha21, value1).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha22, value2).flatten(-2)) / 2

        return output1, output2


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=1, num_heads=8, batch_norm=False, activation="relu"):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(CrossAttentionBlock(hidden_dim, num_heads))

        if batch_norm:
            self.protein_batch_norm_layers = nn.ModuleList()
            self.text_batch_norm_layers = nn.ModuleList()
            for _ in range(self.num_layers):
                self.protein_batch_norm_layers.append(nn.BatchNorm1d(hidden_dim))
                self.text_batch_norm_layers.append(nn.BatchNorm1d(hidden_dim))

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def forward(self, protein_input_ids, protein_attention_mask, text_input_ids, text_attention_mask):

        for i, layer in enumerate(self.layers):
            protein_output, text_output = layer(protein_input_ids, protein_attention_mask,
                                                text_input_ids, text_attention_mask)
            if self.batch_norm:
                protein_output = self.protein_batch_norm_layers[i](protein_output.transpose(1, 2)).transpose(1, 2)
                text_output = self.text_batch_norm_layers[i](text_output.transpose(1, 2)).transpose(1, 2)
            if self.activation:
                protein_output = self.activation(protein_output)
                text_output = self.activation(text_output)

        return {
            'protein_output': protein_output,
            'text_output': text_output,
        }
