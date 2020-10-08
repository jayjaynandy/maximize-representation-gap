# Towards Maximizing the Representation Gap between In-Domain & Out-of-Distribution Examples (ICML UDL 2020, NeurIPS 2020)
Workshop version: ICML Workshop on Uncertainty & Robustness in Deep Learning (UDL) [[Link]](http://www.gatsby.ucl.ac.uk/~balaji/udl2020/accepted-papers/UDL2020-paper-134.pdf)

Full paper: Paper link to be uploaded soon. 

Abstract - Among existing uncertainty estimation approaches, Dirichlet Prior Network (DPN) distinctly models different predictive uncertainty types. 
However, for in-domain examples with high data uncertainties among multiple classes, even a DPN model often produces indistinguishable representations from the out-of-distribution (OOD) examples, compromising their OOD detection performance. 
We address this shortcoming by proposing a novel loss function for DPN to maximize the representation gap between in-domain and OOD examples. 
Experimental results demonstrate that our proposed approach consistently improves OOD detection performance.



## Descriptions of the codes:
In this repository, we provide the training code for our DPN models along with codes for necessary metrics applied in our paper:
`C-10_DPN_training.py` is the training code for our `DPN^-` classifier for C10 classification task.

Training `DPN+`: Please follow the instructions inside the `C-10_DPN_training.py` code to modify the hyper-parameter of our proposed loss function for `DPN+`.

Also, we provide the `ipython demo` for our synthetic experiment in `synthetic_experiment_demo` directory. 

`uncertainty_metric.py` provides the code for calculating the uncertainty metrices including `total uncertainty` measures i.e `Max-probability`, `Entropy` and `distributional uncertainty` measures i.e `mutual information`, `precision (or inverse-EPKL)` and the code for `differential entropy` for Dirichlet distributions. The functions provided in the code takes the `logit values` of the network as their inputs.

`klDiv_gaussians.py` provides the code to compute the KL-divergence between two Gaussian distributions given their mean and co-variance matrix. This measure is used in Table-3 of our paper.
