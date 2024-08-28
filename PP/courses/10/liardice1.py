import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

npl = 7
mynf = 3

with pm.Model() as liardice_model:
    Nf = pm.Binomial('Nf', n = 5 * npl, p = 1 / 3)
    obs = pm.Binomial('obs', n = Nf, p = 1 / npl, observed = mynf)

with liardice_model:
    trace = pm.sample(40000, tune=10000, return_inferencedata=False, chains=1)


Nf_samples = trace["Nf"]
Nf_samples = np.bincount(Nf_samples, minlength = 5 * npl + 1)
Nf_samples = Nf_samples / 40000

plt.bar(np.arange(5 * npl + 1), Nf_samples)
plt.show()
