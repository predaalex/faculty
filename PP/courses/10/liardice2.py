import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

npl = 7
mynf = 3

with pm.Model() as liardice_model:
    Nf = pm.Binomial('Nf', n = 5 * (npl - 1), p = 1 / 3)

Nf_samples = 3 + pm.draw(Nf, draws = 40000)

Nf_samples = np.bincount(Nf_samples, minlength = 5 * npl + 1)
Nf_samples = Nf_samples / 40000

plt.bar(np.arange(5 * npl + 1), Nf_samples)
plt.show()
