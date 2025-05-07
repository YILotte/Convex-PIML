# Convex Physics-Informed Machine Learning (Convex-PIML)
 
This repository contains codes and data for the following publication:
* Letian YI, Siyuan YANG, Ying CUI, and Zhilu LAI (2025). [Transforming Physics-Informed Machine Learning to Convex Optimization](https://arxiv.org/abs/2505.01047).

Convex-PIML is a comprehensive framework that transforms PIML to convex optimization to overcome the optimization challenges of PIML. The linear combination of B-splines is utilized to approximate the data, promoting the convexity of the loss function. By replacing the non-convex components of the loss function with convex approximations, the problem is further converted into a sequence of successively refined approximated convex optimization problems. This conversion allows the use of well-established convex optimization algorithms, obtaining solutions effectively and efficiently. Furthermore, an adaptive knot optimization method based on error estimate is introduced to mitigate the spectral bias issue of PIML, further improving the performance. 

## Setup
python>=3.11 is recommended.

## Repository Overview
 * `Adaptive example` - The adaptive example of parameter estimation and equation discovery of K-S equation
 * `data` - simulated data of K-S equation and N-S equation.
 * `AdapParaEst.py` - function library of Convex-PIML.
 * `bsplines.py` - The function library of generating B-splines.
 * `fast_proximal.py` - The function of Fast Iterative Soft-Thresholding Algorithm (FISTA).
 * 
## Citation
Please cite the following paper if you find the work relevant and useful in your research:
```
@article{https://arxiv.org/abs/2505.01047,
author = {Letian YI and Siyuan YANG and Ying CUI and Zhilu LAI},
title ={Transforming Physics-Informed Machine Learning to Convex Optimization},
year = {2025},
doi = {10.48550/arXiv.2505.01047}
}
```
