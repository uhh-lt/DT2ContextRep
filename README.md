# DT2ContextRep

This is the official implementation of [Using distributional thesaurus to enhance transformer-based contextualized representations for low resource languages](https://dl.acm.org/doi/abs/10.1145/3477314.3507077).

**Abstract**: Transformer-based language models recently gained large popularity in Natural Language Processing (NLP) because of their diverse applicability in various tasks where they reach state-of-the-art performance. Even though for resource-rich languages like English, performance is very high, there is still headroom for improvement for low resource languages. In this paper, we propose a methodology to incorporate Distributional Thesaurus information using a Graph Neural Network on top of pretrained Transformer models to improve the state-of-the-art performance for tasks like semantic textual similarity, sentiment analysis, paraphrasing, and discourse analysis. In this study, we attempt various NLP tasks using our proposed methodology for five languages - English, German, Hindi, Bengali, and Amharic - and show that by using our approach, the performance improvement over transformer models increases as we move from resource-rich (English) to low-resource languages (Hindi, Bengali, and Amharic).

## Environment Setup

1. Move into `src` folder: `cd src`
2. Set the required configuration in `config.py`
3. Execute `python main.py`

If you use this method in your work then please cite the following paper:
```
@inproceedings{10.1145/3477314.3507077,
author = {Venkatesh, Gopalakrishnan and Jana, Abhik and Remus, Steffen and Sevgili, \"{O}zge and Srinivasaraghavan, Gopalakrishnan and Biemann, Chris},
title = {Using Distributional Thesaurus to Enhance Transformer-Based Contextualized Representations for Low Resource Languages},
year = {2022},
isbn = {9781450387132},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477314.3507077},
doi = {10.1145/3477314.3507077},
booktitle = {Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing},
pages = {845–852},
numpages = {8},
keywords = {semantics, distributional thesaurus, low resource language, transformers, graph convolution network},
location = {Virtual Event},
series = {SAC '22}
}
```
