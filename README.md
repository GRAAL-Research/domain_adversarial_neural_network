# domain_adversarial_neural_network
Domain Adaption Learning Algorithm (accepted for publication in JMLR 2015 see [1])


## Dependencies
This Python code depends on numpy and scipy librairies

## Usage
Each bound computation routine is contained in a single file.
The name of the file refers to the bound number in the paper :
* pac_bound_0.py
* pac_bound_1.py
* pac_bound_1p.py
* pac_bound_2.py
* pac_bound_2p.py

You can compute a bound value by either:
* Importing the file in a python project and calling the contained function, or
* Executing the file from the command-line.

For instance:
``` bash
$ python pac_bound_2.py
----------------------------------------------------------------------------------------------------
Usage: pac_bound_2.py empirical_gibbs_risk empirical_disagreement [m] [KLQP] [delta]
----------------------------------------------------------------------------------------------------
  PAC Bound TWO of Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

    Compute a PAC-Bayesian upper bound on the Bayes risk by
    using the C-Bound. To do so, we bound *simultaneously*
    the disagreement and the joint error.

    empirical_gibbs_risk : Gibbs risk on the training set
    empirical_disagreement : Expected disagreement on the training set
    m : number of training examples
    KLQP : Kullback-Leibler divergence between prior and posterior
    delta : confidence parameter (default=0.05)


$ python pac_bound_2.py 0.2 0.3 1000 2.0
empirical_gibbs_risk = 0.2
empirical_disagreement = 0.3
m = 1000
KLQP = 2.0
delta = 0.05
bayes risk bound = 0.360433
```

## References
[1] Pascal Germain, Alexandre Lacasse, François Laviolette, Mario Marchand and Jean-Francis Roy. 
"Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm". 
Journal of Machine Learning Research (JMLR), volume 16 (Apr) p. 787-860, 2015.
http://www.jmlr.org/papers/v16/germain15a.html

Gani, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F.,
Marchand, M., and Lempitsky, V. Domain-adversarial training of neural networks. arXiv
preprint arXiv:1505.07818 (2015)

