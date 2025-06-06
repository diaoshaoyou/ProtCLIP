import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import pdb

class LocalCrossAttention(nn.Module):
    def __init__(self, embed_dim, drop_rate=0):
        super(LocalCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query1 = nn.Linear(embed_dim, embed_dim)
        self.key1 = nn.Linear(embed_dim, embed_dim)
        self.value1 = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(drop_rate)
        self.query2 = nn.Linear(embed_dim, embed_dim)
        self.key2 = nn.Linear(embed_dim, embed_dim)
        self.value2 = nn.Linear(embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(drop_rate)

    def forward(
        self,
        input_tensor1,
        input_tensor2,
        attention_mask1=None,
        attention_mask2=None      
    ):
        """Params:
            input_tensor1: fragment embeddings, [frag_num, dim=512]
            input_tensor2: subtext prototype embeddings, [subtext_num, dim=512]
        """
        # for protein input [frag_num, dim]
        query_layer1 = self.query1(input_tensor1)
        key_layer1 = self.key1(input_tensor1)
        value_layer1 = self.value1(input_tensor1)

        # for text input [subtext_num, dim]
        query_layer2 = self.query2(input_tensor2)
        key_layer2 = self.key2(input_tensor2)
        value_layer2 =  self.value2(input_tensor2)
        
        attention_scores1  = query_layer2 @ key_layer1.T # [subtext_num, dim] @ [dim, frag_num] = [subtext_num, frag_num]
        attention_scores1 = attention_scores1 / math.sqrt(self.embed_dim)
        if attention_mask1 is not None:
            attention_scores1 = attention_scores1 + attention_mask1

        # Sigmoid is better in this case
        # TODO: pre-normalize vs. post-normalize 
        attention_probs1 = F.sigmoid(attention_scores1)
     
        attention_probs1 = self.dropout1(attention_probs1) #[subtext_num, frag_num]
        context_layer1 =  attention_probs1 @ value_layer1 # [subtext_num, frag_num] @ [frag_num, dim] = [subtext_num, dim]
        attention_scores2 = query_layer1 @ key_layer2.T # [frag_num, dim] @ [dim, subtext_num] = [frag_num, subtext_num]
        attention_scores2 = attention_scores2 / math.sqrt(self.embed_dim)

        if attention_mask2 is not None:
            attention_scores2 = attention_scores2 + attention_mask2
       
        attention_probs2 = F.sigmoid(attention_scores2)

        attention_probs2 = self.dropout2(attention_probs2) #[frag_num, subtext_num]
        context_layer2 = attention_probs2 @ value_layer2 # [frag_num, subtext_num] @ [subtext_num, dim] = [frag_num, dim]
        return context_layer2, attention_probs2, context_layer1, attention_probs1

class FragmentDecoder(nn.Module):
    def __init__(self, embed_dim):
        super(FragmentDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim) # weighted text 
        self.key = nn.Linear(embed_dim, embed_dim) # protein fragment
        self.value = nn.Linear(embed_dim, embed_dim) # protein fragment
    
    def forward(
        self,
        protein,
        weighted_text  
    ):
        """Params:
            protein: [bsz, token_len, dim=512]
            weighted_text: [bsz, 1, dim=512]
        """
        query_layer = self.query(protein)
        key_layer = self.key(weighted_text)
        value_layer =  self.value(weighted_text)

        attention_scores  = query_layer @ key_layer.transpose(-1, -2) # [token_len, dim] @ [dim, 1] = [token_len, 1]
        attention_scores = attention_scores / math.sqrt(self.embed_dim)
        attention_probs = F.softmax(attention_scores, dim=1) # [token_len, 1]
        fused_protein =  attention_probs @ value_layer # [token_len, 1] @ [1, dim] = [token_len, dim]
        return fused_protein