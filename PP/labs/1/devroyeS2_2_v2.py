import pymc as pm
import numpy as np

with pm.Model() as model:
    c = 10
    # X = c
    X = pm.Uniform("X", 0, 4 * c)
    Y = pm.Bernoulli("Y", X / (c + X))
    Yh = pm.Deterministic('Yh', pm.math.switch(X / (c + X) > 0.5, 1, 0))

### Mysterious code to be explained in Chapter 3.
with model:
    step = pm.Metropolis()
    trace = pm.sample(40000, tune=10000, chains=1, step=step, return_inferencedata=False)

Y_samples = trace['Y']
Yh_samples = trace['Yh']

print()
print()
print("Bayes Error:", (Yh_samples != Y_samples).mean())



