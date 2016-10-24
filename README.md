# Domain Adversarial Neural Network (shallow implementation)

This python code has been used to conduct the experiments
presented in Section 5.1 of the following JMLR paper.


> Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle,
> François Laviolette, Mario Marchand, Victor Lempitsky.  
> Domain-Adversarial Training of Neural Networks.
> *Journal of Machine Learning Research*, 2016.  
http://jmlr.org/papers/v17/15-239.html

## Content

* ``DANN.py`` contains the learning algorithm. The ``fit()`` function is a very straightforward implementation of *Algorithm 1* of the paper.

* ``experiments_amazon.py`` contains an example of execution on the *Amazon sentiment analysis* dataset (a copy of the dataset files is contained in the folder ``data``). Computes the target test risk (see Table 1 of the paper) and the *Proxy-A-Distance* (see Figure 3 of the paper).

* ``experiments_moons.py`` contains the code used to produce Figure 2 of the paper (experiments on the *inter-twinning moons* toy problem).

* ``mSDA.py`` contains the functions used to generate the mSDA representations (these are literal translations of Chen et al. (2012) Matlab code)
