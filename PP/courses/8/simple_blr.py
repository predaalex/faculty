import numpy as np
import pymc as pm
from matplotlib import pyplot as plt


n = 100 # number of data points
x_data = np.random.normal(0, 1, n)
y_data = x_data + 0.5 + np.random.normal(0, 0.35, n)

with pm.Model() as model:
    std = pm.Uniform("std", 0, 100)  

    beta = pm.Normal("beta", 0, 100)
    alpha = pm.Normal("alpha", 0, 100)
    linear_regress = x_data*alpha+beta 

    y = pm.Normal('y', linear_regress, std, observed=y_data)

with model:

    # Use the default NUTS for sampling
    trace = pm.sample(50000, tune=30000, return_inferencedata=False, chains=1)


alpha_samples = trace['alpha']
beta_samples = trace['beta']
std_samples = trace['std']

# histogram of the samples:

plt.hist(alpha_samples, density = True)
plt.show()

plt.hist(beta_samples, density = True)
plt.show()

plt.hist(std_samples, density = True)
plt.show()
