import pymc as pm
import numpy as np

with pm.Model() as model:
    c = 10
    # X = c
    X = pm.Uniform("X", 0, 4 * c)
    loss = pm.Deterministic('loss', pm.math.minimum(c, X) / (c + X))

### Mysterious code to be explained in Chapter 3.
with model:
    step = pm.Metropolis()
    trace = pm.sample(40000, tune=10000, chains=1, step=step, return_inferencedata=False)

loss_samples = trace['loss']

print()
print()
print("Bayes Error:", loss_samples.mean())


