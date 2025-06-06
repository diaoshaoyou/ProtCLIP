import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import get_rank, get_world_size
from torch.distributed.nn.functional import all_gather
from torch.nn.functional import cross_entropy
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel

from model.layers import CrossAttention, CLIPLoss
from model.bank import AttrProtoBank
from model.attention import LocalCrossAttention, FragmentDecoder
from model.modeling_esm import MyEsmAutoModel
import pdb

class ProteinTextCLIPConfig(PretrainedConfig):
    model_type = "protein_text_clip"
    is_composition = True

    def __init__(self,
                 protein_model_config,
                 text_model_config,
                 projection_dim,
                 **kwargs):
        super().__init__(**kwargs)
        self.protein_model_config = protein_model_config
        self.text_model_config = text_model_config

        if isinstance(protein_model_config, dict):
            self.protein_model_config = AutoConfig.for_model(**protein_model_config)
        if isinstance(text_model_config, dict):
            self.text_model_config = AutoConfig.for_model(**text_model_config)
        self.projection_dim = projection_dim

        self.hidden_sizes = [self.protein_model_config.hidden_size,
                             self.text_model_config.hidden_size,
                             self.projection_dim]
        self.logit_scale_init = kwargs.pop("logit_scale_init", 0.07)


class ProteinTextCLIPForPretrain(PreTrainedModel):
    config_class = ProteinTextCLIPConfig

    def __init__(self, config):
        super().__init__(config)
        protein_model_config = config.protein_model_config
        text_model_config = config.text_model_config

        self.protein_model = AutoModel.from_pretrained(protein_model_config._name_or_path)
        self.text_model = AutoModel.from_pretrained(text_model_config._name_or_path)

        self.protein_projection = nn.Sequential(
            nn.Linear(protein_model_config.hidden_size, self.config.projection_dim),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_model_config.hidden_size, self.config.projection_dim),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.config.logit_scale_init))

    def forward(self, protein_input_ids, protein_attention_mask, text_input_ids, text_attention_mask):
        protein_embeds = self.protein_model(
            input_ids=protein_input_ids, attention_mask=protein_attention_mask
        ).last_hidden_state.mean(dim=1)
        protein_embeds = self.protein_projection(protein_embeds)

        text_embeds = self.text_model(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        ).last_hidden_state.mean(dim=1)
        text_embeds = self.text_projection(text_embeds)

        # normalize the embeddings
        protein_embeds = protein_embeds / protein_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        cl_loss = CLIPLoss(
            local_loss=False,
            gather_with_grad=True,
            cache_labels=True,
            rank=get_rank(),
            world_size=get_world_size()
        )(protein_embeds, text_embeds, self.logit_scale.exp())

        return {
            "loss": cl_loss,
            "cl_loss": cl_loss,
            "logit_scale": self.logit_scale.exp()
        }


class FSTConfig(PretrainedConfig):
    model_type = "FST"
    is_composition = True

    def __init__(self,
                 protein_model_config,
                 text_model_config,
                 mlp_num_layers,
                 fusion_num_heads,
                 projection_dim,
                 fusion_num_layers,
                 fusion_batch_norm,
                 mmp,
                 local_contrast, 
                 t2p_mlm,
                 proto_dim=512,
                 proto_num=512,
                 local_pool='avg',
                 **kwargs):
        super().__init__(**kwargs)

        self.protein_model_config = protein_model_config
        self.text_model_config = text_model_config

        if isinstance(protein_model_config, dict):
            self.protein_model_config = AutoConfig.for_model(**protein_model_config)
        if isinstance(text_model_config, dict):
            self.text_model_config = AutoConfig.for_model(**text_model_config)
        self.projection_dim = projection_dim

        self.mlp_num_layers = mlp_num_layers
        self.fusion_num_heads = fusion_num_heads
        self.projection_dim = projection_dim
        self.fusion_num_layers = fusion_num_layers
        self.fusion_batch_norm = fusion_batch_norm
        self.hidden_sizes = [self.protein_model_config.hidden_size,
                             self.text_model_config.hidden_size,
                             self.projection_dim]
        self.logit_scale_init = kwargs.pop("logit_scale_init", 0.07)
        self.protein_mask_probability = kwargs.pop("protein_mask_probability", 0.15)
        self.text_mask_probability = kwargs.pop("text_mask_probability", 0.15)

        self.mmp = mmp
        self.local_contrast = local_contrast
        self.t2p_mlm = t2p_mlm
        self.proto_dim = proto_dim
        self.proto_num = proto_num
        self.local_pool = local_pool
        self.local_logit_scale_init = 0.1

class FSTForPretrain(PreTrainedModel):
    # enable gradient_checkpointing
    _keys_to_ignore_on_load_missing = [r"position_ids", "lm_head.decoder.weight"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    supports_gradient_checkpointing = True

    config_class = FSTConfig

    def __init__(self, config):
        super().__init__(config)
        protein_model_config = config.protein_model_config
        text_model_config = config.text_model_config
        self.mmp = self.config.mmp
        self.local_contrast = self.config.local_contrast
        self.t2p_mlm = self.config.t2p_mlm

        # enable gradient_checkpointing
        self.protein_model = MyEsmAutoModel.from_pretrained(protein_model_config._name_or_path)
        
        self.text_model = AutoModel.from_pretrained(text_model_config._name_or_path)
        self.protein_projection = nn.Sequential(
            nn.Linear(protein_model_config.hidden_size, self.config.projection_dim),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_model_config.hidden_size, self.config.projection_dim),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.config.logit_scale_init))

        self.mlm_head = nn.Sequential(
            nn.Linear(protein_model_config.hidden_size, self.config.projection_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.projection_dim),
            nn.Linear(self.config.projection_dim, protein_model_config.vocab_size),
        )
        if self.mmp == 1:
            self.fusion_model = CrossAttention(
                hidden_dim=self.config.projection_dim,
                num_layers=self.config.fusion_num_layers,
                num_heads=self.config.fusion_num_heads,
                batch_norm=self.config.fusion_batch_norm)

            self.mmp_protein_head = nn.Sequential(
                nn.Linear(self.config.projection_dim, self.config.projection_dim),
                nn.GELU(),
                nn.LayerNorm(self.config.projection_dim),
                nn.Linear(self.config.projection_dim, protein_model_config.vocab_size),
            )
            self.mmp_text_head = nn.Sequential(
                nn.Linear(self.config.projection_dim, self.config.projection_dim),
                nn.GELU(),
                nn.LayerNorm(self.config.projection_dim),
                nn.Linear(self.config.projection_dim, text_model_config.vocab_size),
            )
        if self.config.local_pool == 'avg':
            self.local_pool = lambda x: torch.mean(x, dim=1, keepdim=False)
        elif self.config.local_pool == 'max':
            self.local_pool = lambda x: torch.max(x, dim=1, keepdim=False)
        # prototype_bank setting
        self.proto_bank = AttrProtoBank(self.config.proto_dim, self.config.proto_num)   
        
        if self.t2p_mlm == 1:
            self.t2p_mlm_head = nn.Sequential(
                nn.Linear(self.config.projection_dim, self.config.projection_dim),
                nn.GELU(),
                nn.LayerNorm(self.config.projection_dim),
                nn.Linear(self.config.projection_dim, protein_model_config.vocab_size),
            )
            self.decoder = FragmentDecoder(self.config.projection_dim)
        if self.local_contrast == 1:
            self.local_cross_attention =  LocalCrossAttention(self.config.proto_dim)  
            self.local_logit_scale = nn.Parameter(torch.ones([2]) * np.log(1 / self.config.local_logit_scale_init))
            self.predictor = nn.Sequential(nn.Linear(self.config.projection_dim, self.config.projection_dim // 2),
                                        nn.ReLU(inplace=True), # hidden layer 
                                        nn.Linear(self.config.projection_dim // 2, self.config.projection_dim)) # output layer # used for simsiam loss


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self,
                current_epoch,
                protein_input_ids,
                protein_attention_mask,
                text_input_ids,
                text_attention_mask,

                subtext_input_ids,
                subtext_attention_mask,
                fragment_positions,
                fragment_masked_input_ids,
                fragment_masked_labels,

                protein_masked_input_ids,
                protein_masked_labels,
                text_masked_input_ids,
                text_masked_labels
                ):

        text_outputs = self.text_model(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        ).last_hidden_state
        text_embeds = text_outputs.mean(dim=1) #[bsz, dim]
        text_embeds = self.text_projection(text_embeds)

    
        protein_outputs = self.protein_model(
            input_ids=protein_input_ids, attention_mask=protein_attention_mask
        ).last_hidden_state
        protein_embeds = protein_outputs.mean(dim=1) #[bsz, dim]
        protein_embeds = self.protein_projection(protein_embeds)

        # normalize the embeddings
        protein_embeds = protein_embeds / protein_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        loss_dict={}
        cl_loss = CLIPLoss(
            local_loss=False,
            gather_with_grad=True,
            cache_labels=True,
            rank=get_rank(),
            world_size=get_world_size()
        )(protein_embeds, text_embeds, self.logit_scale.exp())
        loss_dict['cl_loss'] = cl_loss
        loss_dict['loss'] = cl_loss

        del protein_embeds

        # compute outputs
        protein_outputs1 = self.protein_model(input_ids=protein_masked_input_ids,
                                             attention_mask=protein_attention_mask).last_hidden_state
        # compute the mlm loss
        protein_p2p_mlm_logits = self.mlm_head(protein_outputs1)
        protein_p2p_mlm_loss = cross_entropy(protein_p2p_mlm_logits.view(-1, protein_p2p_mlm_logits.shape[-1]),
                                         protein_masked_labels.view(-1))
        loss_dict['protein_p2p_mlm_loss'] = protein_p2p_mlm_loss
        loss_dict['loss'] += protein_p2p_mlm_loss*0.3

        # compute the mmp loss
        if self.mmp == 1:
            protein_outputs1 = self.protein_projection(protein_outputs1)
            text_outputs = self.text_model(input_ids=text_masked_input_ids,
                                        attention_mask=text_attention_mask).last_hidden_state
            text_outputs = self.text_projection(text_outputs)
            fusion_outputs = self.fusion_model(protein_outputs1, protein_attention_mask, text_outputs, text_attention_mask)

            protein_mmp_logits = self.mmp_protein_head(fusion_outputs['protein_output'])
            protein_mmp_loss = cross_entropy(protein_mmp_logits.view(-1, protein_mmp_logits.shape[-1]),
                                            protein_masked_labels.view(-1))
            text_mmp_logits = self.mmp_text_head(fusion_outputs['text_output'])
            text_mmp_loss = cross_entropy(text_mmp_logits.view(-1, text_mmp_logits.shape[-1]),
                                        text_masked_labels.view(-1))
            loss_dict['mmp_loss'] = text_mmp_loss + protein_mmp_loss
            loss_dict['loss'] += text_mmp_loss + protein_mmp_loss
        del protein_outputs1

        if self.local_contrast == 1:
            with torch.no_grad():
                subtext_num = len(subtext_input_ids[0])
                subtext_embeds = []
                for i in range(subtext_num):
                    out = self.text_model(input_ids=subtext_input_ids[:,i],
                                                    attention_mask=subtext_attention_mask[:,i]).last_hidden_state #[bsz, token_len, dim]
                    embed = self.local_pool(out) #[bsz, dim]
                    embed = self.text_projection(embed)
                    subtext_embeds.append(embed) #[subtext_num, bsz, dim=512]
                
                subtext_embeds = torch.stack(subtext_embeds, dim=0).transpose(0, 1) #[bsz, subtext_num, dim=512]

            # get prototypes
            subtext_proto_embeds, proto_loss, proto_idx = self.proto_bank(subtext_embeds)
            subtext_proto_embeds = subtext_proto_embeds.reshape(subtext_embeds.shape[0], subtext_embeds.shape[1], -1) #[bsz, subtext_num, dim=512]
            loss_dict['proto_loss'] = proto_loss
            # loss_dict['loss'] += proto_loss

        
        if self.t2p_mlm == 1:
            protein_outputs2 = self.protein_model(input_ids=fragment_masked_input_ids,
                                             attention_mask=protein_attention_mask).last_hidden_state #[bsz, token_len, dim]
            protein_t2p_mlm_loss = self.t2p_mlm_loss(fragment_masked_labels, 
                                                     text_outputs,
                                                     protein_outputs2)
            loss_dict['protein_t2p_mlm_loss'] = protein_t2p_mlm_loss
            loss_dict['loss'] += protein_t2p_mlm_loss*0.7
            del protein_outputs2
        
        if self.local_contrast == 1:
            local_contrast_loss = self.local_contrastive_loss(subtext_proto_embeds, self.protein_projection(protein_outputs))
            loss_dict['local_contrast_loss'] = local_contrast_loss 
            loss_dict['loss'] += local_contrast_loss
        del protein_outputs
        
        loss_dict['logit_scale'] = self.logit_scale.exp()
        return loss_dict
    

    def local_contrastive_loss(self, text_outputs, protein_outputs):
        threshold = 0.3
        # eps=1e-8
        similarity = torch.einsum('btd,bpd->btp', text_outputs, protein_outputs)
        min_sim = similarity.min(dim=-1, keepdim=True)[0]
        max_sim = similarity.max(dim=-1, keepdim=True)[0]
        similarity = (similarity - min_sim) / (max_sim - min_sim)
        similarity = torch.where(similarity < threshold, torch.tensor(0.0), similarity)
        weights = similarity / similarity.sum(dim=-1, keepdim=True)
        protein2text = torch.einsum('btp,bpd->btd', weights, protein_outputs)
        bsz = protein2text.shape[0]
        local_loss = 0.
        for b in range(bsz):
            local_loss += ((self._local_loss_fn(protein2text[b], text_outputs[b]) + self._local_loss_fn(protein2text[b], text_outputs[b])) / 2)
        local_loss /= bsz
        return local_loss

    def _local_loss_fn(self, A, B, norm=True):
        logit_scale = self.local_logit_scale[0].exp()
        if norm:
            A = F.normalize(A, dim=-1, p=2)
            B = F.normalize(B, dim=-1, p=2)
        logits = A @ B.t() * logit_scale
        labels = torch.eye(A.shape[0]).to(A.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    

    def t2p_mlm_loss(self, 
                     fragment_masked_labels,  
                     text_outputs,
                     protein_outputs2):
        """Params:
            fragment_masked_labels: [bsz, token_len], eg. (-100, -100, 5, 8, -100, ..)
            subtext_embeds: [bsz, subtext_num, dim=512], embeddings of each subtext/prototype
            protein_outputs: [bsz, token_len, dim], all unmasked protein tokens, for calculating weights
            protein_outputs2: [bsz, token_len, dim], contain masked fragment
        """
        with torch.no_grad():
            protein_outputs2 = self.protein_projection(protein_outputs2) #[bsz, token_len, dim=512]
            text_outputs = self.text_projection(text_outputs) #[bsz, token_len, dim=512]
            fused_protein = self.decoder(protein_outputs2, text_outputs) # [bsz, token_len, 512]
        protein_t2p_mlm_logits = self.t2p_mlm_head(fused_protein) #[bsz, token_len, vocab_size]
        protein_t2p_mlm_loss = cross_entropy(protein_t2p_mlm_logits.view(-1, protein_t2p_mlm_logits.shape[-1]), fragment_masked_labels.view(-1))
        return protein_t2p_mlm_loss
