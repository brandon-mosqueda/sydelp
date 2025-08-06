# Sybil-resistant Decentralized Learning Protocol (SyDeLP)

This repository contains the code for the experiments of the research paper [_Sybil-Poisoning Resilience in Decentralized Learning With Verifiable Delay Puzzles_](). We provide the implementation of the following:

**Systems**

* SyDeLP (proposal)
* Federated learning [[1]](#1)
* Gossip learning [[5]](#5)
* SybilWall [[3]](#3)
* MAB-FL [[4]](#4)

**Aggregation**

* FedAvg [[1]](#1)
* KRUM, Multi-KRUM [[2]](#2)

    The aggregation functions are reusable as they expect a matrix of model parameters (a Numpy array) and returns the aggregated model as a 1-D Numpy array.

**Attacks**

* Random attack [[2]](#2)
* Sign flipping [[6]](#6)
* Targeted label flipping [[7]](#7)
* Untargeted label flipping [[8]](#8)

**Data partitioning**

* Unbalanced class split (as in [[1]](#1))
* Balanced splits
* Dirichlet split [[9]](#9)

    These functions can be easily reused in other experiments as they receive the matrices X and Y of the datasets and generate the partitions of these datasets.

# Run the experiments

You can use the following command to run any of the provided experiments. The json files have all the parameters required to reproduce the results of the paper. You can also try your own parameters, just be sure you use the expected format as in all the provided examples of `params` folder.

```bash
#
python run.py params/datasets/<dataset>.json params/<attacks>.json <seed>

# For instance
python run.py params/datasets/mnist.json params/attacks/01.fl_random.json 2
```

# Tested dependencies versions

Python 3.9.21 and the following packages were used to run the experiments.

Package | Version
------- | -------
tensorflow | 2.17.0
keras | 3.7.0
numpy | 1.25.0
scikit-learn | 1.5.1
scipy | 1.9.3
pandas | 2.2.1
networkx | 3.2.1

We noted that using a GPU does not offer a better performance due to the constant necessity of moving model parameters to the RAM and GPU memory, so GPU usage is disabled by default. To enable GPU usage please comment the `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` line in `run.py`.

# Citation

```bibtex
@unpublished{ }
```

# References


<a id="1">[1]</a> McMahan et al. (2016). **Communication-Efficient Learning of Deep Networks from Decentralized Data**. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629).


<a id="2">[2]</a> Blanchard et al. (2017). **Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent**. Advances in Neural Information Processing Systems. [Link](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html).

<a id="3">[3]</a> Werthenbach et at. (2023). **Towards Sybil Resilience in Decentralized Learning**. [arXiv:2306.15044](https://arxiv.org/abs/2306.15044).

<a id="4">[4]</a> Wan et al. (2022). **Shielding federated learning: Robust aggregation with adaptive client selection**. [arXiv:2204.13256](https://arxiv.org/abs/2204.13256).

<a id="5">[5]</a> Lian et al. (2017). **Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent**. [arXiv:1705.09056](https://arxiv.org/abs/1705.09056).

<a id="6">[6]</a> Li et al. (2019). **RSA: Byzantine-robust stochastic aggregation methods for distributed learning from heterogeneous datasets**. Proceedings of the AAAI conference on artificial intelligence. [DOI](https://doi.org/10.1609/aaai.v33i01.33011544).

<a id="7">[7]</a> Tolpegin et al. (2020). **Data Poisoning Attacks Against Federated Learning Systems**. Computer Security -- ESORICS 2020. [DOI](https://doi.org/10.1007/978-3-030-58951-6_24).

<a id="8">[8]</a> Fang et al. (2020). **Local Model Poisoning Attacks to Byzantine-Robust Federated Learning**. 29th USENIX Security Symposium. [Link](https://www.usenix.org/conference/usenixsecurity20/presentation/fang).

<a id="9">[9]</a> Li et al. (2022). **Federated learning on non-iid data silos: An experimental study**. 2022 IEEE 38th international conference on data engineering (ICDE). [DOI](https://doi.org/10.1109/ICDE53745.2022.00077).
