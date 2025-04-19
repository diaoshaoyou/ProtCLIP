# ProtCLIP: Function-Informed Protein Multi-Modal Learning
ProtCLIP is a protein multi-modality foundation model for protein sequence understanding, aligning protein sequences and biotexts, as introduced in our [AAAI 2025 oral paper](https://ojs.aaai.org/index.php/AAAI/article/view/34456). And we have contained more technical details in the [arXiv version](https://arxiv.org/abs/2412.20014).

<img src="figures/overview.png" alt="framework" width="850" height="300"> 
Figure 1: Overview of ProtCLIP.

Multi-modality pre-training paradigm that aligns protein sequences and biological descriptions has learned general protein representations and achieved promising performance in various downstream applications. However, these works were still unable to replicate the extraordinary success of language-supervised visual foundation models due to the ineffective usage of aligned protein-text paired data and the lack of an effective function-informed pre-training paradigm. To address these issues, this paper curates a large-scale protein-text paired dataset called ProtAnno with a property-driven sampling strategy, and introduces a novel function-informed protein pre-training paradigm. Specifically, the sampling strategy
determines selecting probability based on the sample confidence and property coverage, balancing the data quality and data quantity in face of large-scale noisy data. Furthermore, motivated by significance of the protein specific functional mechanism, the proposed paradigm explicitly model protein static and dynamic functional segments by two segment-wise pre-training objectives, injecting fine-grained information in a function-informed manner. Leveraging all these innovations, we develop ProtCLIP, a multi-modality foundation model that comprehensively represents function-aware protein embeddings. On 22 different protein benchmarks within 5 types, including protein functionality classification, mutation effect prediction, cross-modal transformation, semantic similarity inference and protein-protein interaction prediction, our ProtCLIP consistently achieves SOTA performance, with remarkable improvements of 75% on average in five cross-modal transformation benchmarks, 59.9% in GO-CC and 39.7% in GO-BP protein function prediction. The experimental results verify the extraordinary potential of ProtCLIP serving as the protein multi-modality foundation model.

## Installation


## Multi-Modal Aligned Dataset
The pre-training dataset for ProtCLIP is built on large-scale protein corpus [UniProt](https://www.uniprot.org/). 

Our pre-training data is sourced from SwissProt and trEMBL, containing proteins with textual descriptions. We align protein sequences with meticulously selected properties to curate
ProtAnno, which is available in sparse version (ProtAnno-S) and dense version (ProtAnno-D). ProtAnno-S includes manually reviewed protein-biotext pairs with higher annotation quality, whereas ProtAnno-D comprises mostly computationally analyzed protein-biotext pairs which are less accurate due to the machine-annotated bias. To gain more insights into the dataset, we conduct extensive quantitative analyses, and display the compositional structure of ProtAnno with varying confidence and property coverage.

The compiled dataset can be accessed through the Zenodo repository: [ProtAnno](https://zenodo.org/records/15245588)

<img src="figures/data_distribution_1.png" alt="data_1" width="480" height="200"> 
Figure 2: Data distribution of ProtAnno-S and ProtAnno-D with different property coverage.

<img src="figures/data_distribution_2.png" alt="data_2" width="600" height="120"> 
Table 1: Data distribution of ProtAnno-S and ProtAnno-D with different sample confidence.

Note that the ProtAnno-D dataset has been filtered using the property-driven sampling strategy.

## Pretrained Model Zoo
We hope ProtCLIP could serve as the protein multi-modality foundation model to promote controllable protein discovery and optimization in real-world scenarios.

ProtCLIP: [config](https://github.com/diaoshaoyou/ProtCLIP/blob/main/config/config.json) | [checkpoint](https://zenodo.org/records/15245588/files/model.safetensors?download=1)

## Usage


## License
This codebase is released under the Apache License 2.0 as in the [LICENSE](https://github.com/diaoshaoyou/ProtCLIP/blob/main/LICENSE) file.

## Citation
If you find this research work interesting and helpful, please cite our paper:
```
@inproceedings{zhou2025protclip,
  title={ProtCLIP: Function-Informed Protein Multi-Modal Learning},
  author={Zhou, Hanjing and Yin, Mingze and Wu, Wei and Li, Mingyang and Fu, Kun and Chen, Jintai and Wu, Jian and Wang, Zheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={21},
  pages={22937--22945},
  year={2025}
}
``` 

