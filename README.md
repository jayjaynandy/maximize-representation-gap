# Maximizing the Representation Gap between In-Domain and Out-of-Distribution Examples
Existing OOD detection models lead to produce similar representation for in-domain examples with high degree of data uncertainties and OOD examples. This often leads to compromise their OOD detection performance. In this work, we propose a novel loss function for DPN classification frameworks that maximizes the representational gap between in-domain and OOD examples to address this shortcoming.

## Descriptions of the codes:
In this repository, we provide the training code for our DPN models along with codes for necessary metrics applied in our paper:
`C-10_DPN_training.py` is the training code for our `DPN^-` classifier for C10 classification task.

Training `DPN+`: Please follow the instructions inside the `C-10_DPN_training.py` code to modify the hyper-parameter of our proposed loss function for `DPN+`.

Also, we provide the `ipython demo` for our synthetic experiment in `synthetic_experiment_demo` directory. 

`uncertainty_metric.py` provides the code for calculating the uncertainty metrices including `total uncertainty` measures i.e `Max-probability`, `Entropy` and `distributional uncertainty` measures i.e `mutual information`, `precision (or inverse-EPKL)` and the code for `differential entropy` for Dirichlet distributions. The functions provided in the code takes the `logit values` of the network as their inputs.

`klDiv_gaussians.py` provides the code to compute the KL-divergence between two Gaussian distributions given their mean and co-variance matrix. This measure is used in Table-3 of our paper.
