import random
import warnings
import torch
from transformers import BertTokenizer, BertTokenizerFast
from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch
import pdb

def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()


def torch_mask_tokens_with_mask_labels(inputs, tokenizer, mask_labels):
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
            " --mlm flag if you want to use this tokenizer."
        )
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)

    probability_matrix = mask_labels

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = probability_matrix.bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    return inputs, labels


def torch_mask_tokens(inputs, tokenizer, mlm_probability, special_tokens_mask=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 100% MASK.
    """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels

def torch_mask_fragments(inputs, tokenizer, mask_range, special_tokens_mask=None):
    """Params:
    inputs: [bsz, token_len]
    mask_range: <int>, the token range of being masked

    Prepare masked tokens inputs/labels for masked language modeling: fragement-level masking
    """
    labels = inputs.clone() 
    probability_matrix = torch.full(labels.shape, 0.0)
    frag_area = torch.full(labels.shape, 0)
    bsz = len(inputs)
    token_len = len(inputs[0])
    if token_len < mask_range * 1.5:
        mask_range = token_len // 2
    for b in range(bsz):
        while(sum(frag_area[b]) < token_len*0.15):
            mask_range=random.randint(mask_range//2, mask_range)
            mask_start = random.randint(1, token_len - mask_range - 1)
            while(frag_area[b][mask_start]==1):
                mask_start = random.randint(1, token_len - mask_range - 1)
            frag_area[b][mask_start:mask_start+ mask_range + 1] = 1
            probability_matrix[b, mask_start:mask_start + mask_range + 1] = 1.0
    
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens, unmasked token labels=-100

    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token) # assign [MASK] to the masked fragment, [MASK]=32
    
    return inputs, labels

def whole_word_mask(input_tokens, tokenizer, mlm_probability, max_predictions=512):
    """
    Get 0/1 labels for masked tokens with whole word mask proxy
    """
    if not isinstance(tokenizer, (BertTokenizer, BertTokenizerFast)):
        warnings.warn(
            "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
            "Please refer to the documentation for more information."
        )

    cand_indexes = []
    for i, token in enumerate(input_tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue

        if len(cand_indexes) >= 1 and token.startswith("##"):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    random.shuffle(cand_indexes)
    num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_probability))))
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_lms.append(index)

    if len(covered_indexes) != len(masked_lms):
        raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
    mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
    return mask_labels


class DataCollatorForProteinTextCLIPPretrain(DataCollatorMixin):
    def __init__(self, protein_tokenizer, text_tokenizer, mlm_probability=0.0):
        self.return_tensors = "pt"
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.mlm_probability = mlm_probability

    def torch_call(self, examples):
        protein_tokenized = {
            'input_ids': [e["protein_input_ids"] for e in examples],
            'attention_mask': [e["protein_attention_mask"] for e in examples],
        }
        text_tokenized = {
            'input_ids': [e["text_input_ids"] for e in examples],
            'attention_mask': [e["text_attention_mask"] for e in examples],
        }
        protein_tokenized = self.protein_tokenizer.pad(protein_tokenized, return_tensors='pt')
        text_tokenized = self.text_tokenizer.pad(text_tokenized, return_tensors='pt')

        if self.mlm_probability == 0.0:
            return {
                'protein_input_ids': protein_tokenized['input_ids'],
                'protein_attention_mask': protein_tokenized['attention_mask'],
                'text_input_ids': text_tokenized['input_ids'],
                'text_attention_mask': text_tokenized['attention_mask'],
            }

        protein_masked_input_ids, protein_masked_labels = torch_mask_tokens(protein_tokenized["input_ids"].clone(),
                                                                            self.protein_tokenizer,
                                                                            self.mlm_probability)

        text_mask_labels = []
        for e in text_tokenized["input_ids"]:
            ref_tokens = []
            for id in tolist(e):
                token = self.text_tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            text_mask_labels.append(whole_word_mask(ref_tokens, self.text_tokenizer, self.mlm_probability))
        batch_mask = _torch_collate_batch(text_mask_labels, self.text_tokenizer)
        text_masked_input_ids, text_masked_labels = torch_mask_tokens_with_mask_labels(
            text_tokenized["input_ids"].clone(),
            self.text_tokenizer,
            batch_mask)
        return {
            'protein_input_ids': protein_tokenized['input_ids'],
            'protein_attention_mask': protein_tokenized['attention_mask'],
            'text_input_ids': text_tokenized['input_ids'],
            'text_attention_mask': text_tokenized['attention_mask'],
            'protein_masked_input_ids': protein_masked_input_ids,
            'protein_masked_labels': protein_masked_labels,
            'text_masked_input_ids': text_masked_input_ids,
            'text_masked_labels': text_masked_labels,
        }

class DataCollatorForFSTPretrain(DataCollatorMixin):
    def __init__(self, protein_tokenizer, text_tokenizer, mlm_probability=0.0, mask_range=50):
        self.return_tensors = "pt"
        self.protein_tokenizer = protein_tokenizer
        self.text_tokenizer = text_tokenizer
        self.mlm_probability = mlm_probability
        self.mask_range = mask_range

    def torch_call(self, examples):
        protein_tokenized = {
            'input_ids': [e["protein_input_ids"] for e in examples],
            'attention_mask': [e["protein_attention_mask"] for e in examples],
        }
        text_tokenized = {
            'input_ids': [e["text_input_ids"] for e in examples],
            'attention_mask': [e["text_attention_mask"] for e in examples],
        }
        fragment_positions = []
        for e in examples:
            fragment_positions.append(torch.unsqueeze(e['fragment_positions'], dim=0))
        fragment_positions = torch.cat(fragment_positions) #[bsz, frag_num=11, 2]

        bsz = len(examples)
        subtext_num = len(examples[0]['subtext_input_ids'])
        subtext_tokenized={}
        subtext_tokenized['input_ids']=[]
        subtext_tokenized['attention_mask']=[]
        for e in examples:
            subtext_tokenized['input_ids'] += e["subtext_input_ids"]
            subtext_tokenized['attention_mask'] += e['subtext_attention_mask']
        # subtext_tokenized: [2, bsz*subtext_num]

        # pad seqs within a batch to the same length, padding token=1  
        protein_tokenized = self.protein_tokenizer.pad(protein_tokenized, return_tensors='pt')
        text_tokenized = self.text_tokenizer.pad(text_tokenized, return_tensors='pt')
        subtext_tokenized = self.text_tokenizer.pad(subtext_tokenized, return_tensors='pt')
        subtext_tokenized['input_ids'] = subtext_tokenized['input_ids'].reshape(bsz, subtext_num, -1)
        subtext_tokenized['attention_mask'] = subtext_tokenized['attention_mask'].reshape(bsz, subtext_num, -1)
        # subtext_tokenized: [2, bsz, subtext_num]

        if self.mlm_probability == 0.0:
            return {
                'protein_input_ids': protein_tokenized['input_ids'],
                'protein_attention_mask': protein_tokenized['attention_mask'],
                'text_input_ids': text_tokenized['input_ids'],
                'text_attention_mask': text_tokenized['attention_mask'],
                'subtext_input_ids': subtext_tokenized['input_ids'],
                'subtext_attention_mask': subtext_tokenized['attention_mask'],
                'fragment_positions': fragment_positions, 
            }

        protein_masked_input_ids, protein_masked_labels = torch_mask_tokens(protein_tokenized["input_ids"].clone(),
                                                                            self.protein_tokenizer,
                                                                            self.mlm_probability)
        fragment_masked_input_ids, fragment_masked_labels = torch_mask_fragments(protein_tokenized["input_ids"].clone(),
                                                                            self.protein_tokenizer,
                                                                            self.mask_range)
        # pdb.set_trace()
        text_mask_labels = []
        for e in text_tokenized["input_ids"]:
            ref_tokens = []
            for id in tolist(e):
                token = self.text_tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            text_mask_labels.append(whole_word_mask(ref_tokens, self.text_tokenizer, self.mlm_probability))
        batch_mask = _torch_collate_batch(text_mask_labels, self.text_tokenizer)
        text_masked_input_ids, text_masked_labels = torch_mask_tokens_with_mask_labels(
            text_tokenized["input_ids"].clone(),
            self.text_tokenizer,
            batch_mask)
        return {
            'protein_input_ids': protein_tokenized['input_ids'],
            'protein_attention_mask': protein_tokenized['attention_mask'],
            'text_input_ids': text_tokenized['input_ids'],
            'text_attention_mask': text_tokenized['attention_mask'],

            'subtext_input_ids': subtext_tokenized['input_ids'],
            'subtext_attention_mask': subtext_tokenized['attention_mask'],
            'fragment_positions': fragment_positions,
            'fragment_masked_input_ids': fragment_masked_input_ids,
            'fragment_masked_labels': fragment_masked_labels,

            'protein_masked_input_ids': protein_masked_input_ids,
            'protein_masked_labels': protein_masked_labels,
            'text_masked_input_ids': text_masked_input_ids,
            'text_masked_labels': text_masked_labels,
        }
