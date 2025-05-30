import pandas as pd
import pickle
import os
import numpy as np
from tqdm import tqdm
import transformers
from transformers import EarlyStoppingCallback, AutoTokenizer, AutoConfig, AutoModel
import torch
import datasets
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pdb



def inference(model, batch):
    """average pooling of the last layer of the model considering the different lengths of the sequences in the batch
    ref: https://github.com/facebookresearch/esm
    """
    model.eval()
    with torch.no_grad():
        output = model(**batch)

    attention_mask = batch["attention_mask"]
    emb = output.last_hidden_state # (batch_size, seq_length, hidden_size)
    protein_attention_mask = attention_mask.bool()
    protein_embedding = torch.stack([emb[i,protein_attention_mask[i, :]][1:-1].mean(dim=0) for i in range(len(emb))], dim=0)
    return protein_embedding

def preprocess(
    # pretrained encoder model name
    model_name,
    # encode batch size
    batch_size,
    # data path
    data_path = "/root/DATA/datasets/KG/",
    # output path
    output_path = "/root/DATA/downstream/KG",
    # model
    model = None
    ):
    # model device
    device = "cuda:0"
    
    # load model
    if model == None:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(config._name_or_path)
    # model.print_trainable_parameters()
    tokenizer = AutoTokenizer.from_pretrained('/root/DATA/backbones/esm2_t33_650M_UR50D')
    model.to(device)

    # tokenizing data path
    tokenized_data_dir="/root/encoded_protein"

    # load data
    if not os.path.exists(tokenized_data_dir):
        def tokenize_data(datapoint):
            seq = datapoint['sequence']
            tokenized = tokenizer(
                    seq,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                    max_length=1024
                    )
            tokenized.update(datapoint)
            return tokenized
        data = load_dataset(data_path, data_files={"protein.csv"})
        data = data["train"].map(tokenize_data, num_proc=8)
        data.save_to_disk(tokenized_data_dir)
    data = datasets.load_from_disk(tokenized_data_dir)

    # save the data node_index, node_id, and node_name
    outputs = {
        "node_index": data["node_index"],
        "node_id": data["node_id"],
        "node_name": data["node_name"],
    }
    data = data.remove_columns(["node_id","node_index","node_name","node_type","node_source","sequence"])

    # start encoding using protein model
    loader = DataLoader(data, 
                        batch_size=batch_size,
                        collate_fn=transformers.DataCollatorWithPadding(tokenizer, 
                                max_length=tokenizer.model_max_length,
                                pad_to_multiple_of=8,
                                return_tensors="pt",
                                ),
                        )

    embeddings = []
    for batch in tqdm(loader):
        # map batch components to cuda device
        batch = {k:v.to(device) for k,v in batch.items()}
        emb = inference(model, batch)
        emb = emb.cpu().numpy()
        embeddings.append(emb)

    outputs["embedding"] = np.concatenate(embeddings, axis=0)

    # save outputs to disk
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(os.path.join(output_path, "protein.pkl"), "wb") as f:
        pickle.dump(
            outputs,
            f,
        )
    print("Done!")