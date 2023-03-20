# Unsupervised Heterophilous Network Embedding via r-Ego Network Discrimination

Implementation of Selene with Pytorch.

### Required packages
The code has been tested running under Python 3.9.6. with the following packages installed (along with their dependencies):

- numpy == 1.16.5
- pandas == 0.25.1
- scikit-learn == 0.21.2
- networkx == 2.3
- pytorch == 1.9.0
- torch_geometric == 2.0.1
- gensim==3.6.0
- fastdtw==0.3.2
- lightning-bolts==0.4.0

### Data requirement
All eight datasets we used in the paper are all public datasets which can be downloaded from the internet.

### Code execution
```
python main.py --data_name Texas --layer_num 2 --epoch_num 501 --lr 0.01 --p_x 0.2 --p_e 0.3 --gnn_encoder GCN --feat_method rw --SEED 1234 --gpu True
```

## Cite

Please cite our paper if it is helpful in your own work:

```bibtex
@article{ZGGP22,
title = {Unsupervised Network Embedding Beyond Homophily},
author = {Zhiqiang Zhong and Guadalupe Gonzalez and Daniele Grattarola and Jun Pang},
journal = {Transactions on Machine Learning Research (TMLR)},
year = {2022},
}
```