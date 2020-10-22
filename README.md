# Towards Maximizing the Representation Gap between In-Domain & Out-of-Distribution Examples
Workshop version: [ICML 2020 Workshop on Uncertainty & Robustness in Deep Learning (UDL)](http://www.gatsby.ucl.ac.uk/~balaji/udl2020/accepted-papers/UDL2020-paper-134.pdf)

Full paper:  [NeurIPS 2020 [arXiv link]](https://arxiv.org/abs/2010.10474)

Abstract - Among existing uncertainty estimation approaches, Dirichlet Prior Network (DPN) distinctly models different predictive uncertainty types. 
However, for in-domain examples with high data uncertainties among multiple classes, even a DPN model often produces indistinguishable representations from the out-of-distribution (OOD) examples, compromising their OOD detection performance. 
In this paper, we address this shortcoming by proposing a novel loss function for DPN to maximize the representation gap between in-domain and OOD examples. 
Experimental results demonstrate that our proposed approach consistently improves OOD detection performance.

<p align="center">
  <img width="750" height="250" src="https://github.com/jayjaynandy/maximize-representation-gap/blob/master/DPN_uncertainties.png" alt="Desired representation of predictive uncertainties.">
</p>


We show that for in-domain examples with high data uncertainties, their loss function distributes the target precision values among the overlapping classes, leading to much flatter distributions. 
Hence, it often produces indistinguishable representations for those in-domain misclassified examples from the OOD examples, compromising the OOD detection performance.
Hence, we propose to produce sharp-multi-modal Dirichlet distributions, where probability densities are uniformly spread across each corner of the probability simplex, for OOD examples to maximize their representational gap from the in-domain examples. See the above figure, where we present the desired behaviours of DPN classifiers to indicate different predictive uncertainty types. 

We further show that, the DPN framework using RKL loss cannot produce sharp-multi-modal Dirichlet distributions for OOD examples. Hence, we propose a novel loss function that explicitely models the concentration parameters of the output Dirichlet distributions for our DPN.


## Descriptions of the codes:
Our models are trained and tested using `keras 2.1.2` and `tensorflow 1.9.0`

### Synthetic Dataset.
Please follow the jupyter notebook domonstration, `DPN_synthetic_demo.ipynb` in `synthetic_experiment_demo` to understand and visualize the motivation and advantage of our proposed loss function for the DPN models.

`toyNet.py`: A 2-layered network used for our experiments. We can simply choose a different network for complex datasets.

`uncertainty_metric.py`provides the code for calculating the uncertainty measures including `total uncertainty` measures i.e `Max-probability`, `Entropy` and `distributional uncertainty` measures i.e `mutual information`, `precision (or inverse-EPKL)` and the code for `differential entropy` for Dirichlet distributions. The functions provided in the code takes the `logit values` of the network as their inputs.

`synthetic_data.py`: To generate the synthetic data.


### Benchmark Dataset.
`C-10_DPN_training.py` in `Benchmark/C-10/` directory provides the training code for our `DPN^-` classifier for C10 classification task.

Training `DPN+`: Please follow the instructions inside the `C-10_DPN_training.py` code to modify the hyper-parameter of our proposed loss function for `DPN+` (as instructed inside the training code).

Similarly, the `DPN-` training codes for `C-100` and `TIM` classification task is provided in `Benchmark/C-100/c100_DPN_training.py` and `Benchmark/TIM/TIM_DPN_training.py` respectively.

`uncertainty_metric.py` (in `Benchmark` directory) provides the code for calculating the uncertainty measures

`klDiv_gaussians.py` (in `Benchmark` directory) provides the code to compute the KL-divergence between two Gaussian distributions given their mean and co-variance matrix. This measure is used in Table-3 of our paper.

## Citation

If our code or our results are useful in your reasearch, please consider citing:

```[bibtex]
@inproceedings{maximize-representation-gap_neurips2020,
  author={Jay Nandy and Wynne Hsu and Mong{-}Li Lee},
  title={Towards Maximizing the Representation Gap between In-Domain & Out-of-Distribution Examples},
  booktitle={NeurIPS},
  year={2020},
}
