import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

x = 6

def obs_logp(value, D):

    return pm.math.switch(pm.math.eq(D, 0),
           pm.math.switch(pm.math.le(value, 4), pm.math.log(1.0 / 4.0), -np.inf),
           pm.math.switch(pm.math.eq(D, 1),
           pm.math.switch(pm.math.le(value, 6), pm.math.log(1.0 / 6.0), -np.inf),
           pm.math.switch(pm.math.eq(D, 2),
           pm.math.switch(pm.math.le(value, 8), pm.math.log(1.0 / 8.0), -np.inf),
           pm.math.switch(pm.math.eq(D, 3),
           pm.math.switch(pm.math.le(value, 12), pm.math.log(1.0 / 12.0), -np.inf),
           pm.math.switch(pm.math.eq(D, 4),
           pm.math.switch(pm.math.le(value, 20), pm.math.log(1.0 / 20.0), -np.inf),
           -np.inf)
           )
           )
           )
           )


with pm.Model() as model:

    D = pm.Categorical("D", p = [1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0], initval = 3)

    obs = pm.CustomDist(
        'obs',
        D,
        logp=obs_logp,
        observed = x
    )
    
with model:
    step = pm.CategoricalGibbsMetropolis(vars=[D])
    trace = pm.sample(400000, step=step, tune=100000, return_inferencedata=False, chains=1)

D_samples = trace["D"]

D_samples = np.bincount(D_samples, minlength = 5)
D_samples = D_samples / 400000

print(D_samples)

plt.bar(np.arange(5), D_samples)
plt.show()

