# GPM
Official Pytorch implementation for "**Gradient Projection Memory for Continual Learning**", **ICLR 2021 (Oral)**. [Paper Link](https://openreview.net/forum?id=3AOj0RCNC2)

## Abstract 
The ability to learn continually without forgetting the past tasks is a desired attribute for artificial learning systems. Existing approaches to enable such learning in artificial neural networks usually rely on network growth, importance based weight update or replay of old data from the memory. In contrast, we propose a novel approach where a neural network learns new tasks by taking gradient steps in the orthogonal direction to the gradient subspaces deemed important for the past tasks. We find the bases of these subspaces by analyzing network representations (activations) after learning each task with Singular Value Decomposition (SVD) in a single shot manner and store them in the memory as Gradient Projection Memory (GPM). With qualitative and quantitative analyses, we show that such orthogonal gradient descent induces minimum to no interference with the past tasks, thereby mitigates forgetting. We evaluate our algorithm on diverse image classification datasets with short and long sequences of tasks and report better or on-par performance compared to the state-of-the-art approaches.

## Authors 
Gobinda Saha, Isha Garg, Kaushik Roy 

## Citation
```
@inproceedings{
saha2021gradient,
title={Gradient Projection Memory for Continual Learning},
author={Gobinda Saha and Isha Garg and Kaushik Roy},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=3AOj0RCNC2}
}
```
