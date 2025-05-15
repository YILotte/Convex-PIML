# Convex Physics-Informed Machine Learning (Convex-PIML)
 
This repository contains codes and data for the following publication:
* Letian YI, Siyuan YANG, Ying CUI, and Zhilu LAI (2025). [Transforming Physics-Informed Machine Learning to Convex Optimization](https://arxiv.org/abs/2505.01047).

Physics-Informed Machine Learning (PIML) offers a powerful paradigm of integrating data with physical laws to address important scientific problems, such as parameter estimation, inferring hidden physics, equation discovery, and state prediction, etc. However, PIML still faces many serious optimization challenges that significantly restrict its applications. In this study, we transform PIML to convex optimization to overcome all these limitations, referred to as \textbf{Convex-PIML}. The linear combination of B-splines is utilized to approximate the data, promoting the convexity of the loss function. By replacing the non-convex components of the loss function with convex approximations, the problem is further converted into a sequence of successively refined approximated convex optimization problems. This conversion allows the use of well-established convex optimization algorithms, obtaining solutions effectively and efficiently. Furthermore, an adaptive knot optimization method is introduced to mitigate the spectral bias issue of PIML, further improving the performance. The proposed fully adaptive framework by combining the adaptive knot optimization and BSCA is tested in scenarios with distinct types of physical prior. The results indicate that optimization problems are effectively solved in these scenarios, highlighting the potential of the framework for broad applications.

## Setup
python>=3.11 is recommended.

## Repository Overview
 * `Adaptive example` - The adaptive example of parameter estimation and equation discovery of K-S equation
 * `data` - simulated data of K-S equation and N-S equation.
 * `AdapParaEst.py` - function library of Convex-PIML.
 * `bsplines.py` - The function library of generating B-splines.
 * `fast_proximal.py` - The function of Fast Iterative Soft-Thresholding Algorithm (FISTA).
   
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
