import numpy as np
import pytensor.tensor as at
import pymc as pm
import arviz as az
from matplotlib import pyplot as plt

challenger_data = np.genfromtxt("challenger_data.csv", skip_header=1,
                                usecols=[1, 2], missing_values="NA",
                                delimiter=",")
# drop the NA values
challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]

temperature = challenger_data[:, 0]
D = challenger_data[:, 1]  # defect or not?

def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))

# notice the`value` here. We explain why below.
with pm.Model() as model:
    beta = pm.Normal("beta", mu=0, tau=0.001, initval=0)
    alpha = pm.Normal("alpha", mu=0, tau=0.001, initval=0)
    p = pm.Deterministic("p", 1.0/(1. + at.exp(beta*temperature + alpha)))

# connect the probabilities in `p` with our observations through a
# Bernoulli random variable.
with model:
    observed = pm.Bernoulli("bernoulli_obs", p, observed=D)
    
    # Mysterious code to be explained in Chapter 3
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(120000, step=step, initvals=start, chains=1)
    #burned_trace = trace[100000::2]


alpha_samples = np.concatenate(trace.posterior.alpha.data[:,100000::2])[:, None]  # best to make them 1d
beta_samples = np.concatenate(trace.posterior.beta.data[:,100000::2])[:, None]

#histogram of the samples:
plt.subplot(211)
plt.title(r"Posterior distributions of the variables $\alpha, \beta$")
plt.hist(beta_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\beta$", color="#7A68A6", density=True)
plt.legend()

plt.subplot(212)
plt.hist(alpha_samples, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\alpha$", color="#A60628", density=True)
plt.legend()

#Here is the ArviZ version 
figure,ax = plt.subplots(2,1)


az.plot_posterior(trace, var_names=['beta'], kind='hist',bins=25,
                  figsize=(12.5,6),color="#7A68A6",ax=ax[0])
az.plot_posterior(trace, var_names=['alpha'], kind='hist',bins=25,
                  figsize=(12.5,6),color="#A60628",ax=ax[1])
plt.suptitle(r"Posterior distributions of the variables $\alpha, \beta$",fontsize=20)
ax[0].set_title(r"posterior of $\beta$")
ax[1].set_title(r"posterior of $\alpha$")
plt.plot()
plt.show()

prob_31 = logistic(31, beta_samples, alpha_samples)

plt.xlim(0.995, 1)
plt.hist(prob_31, bins=1000, density=True, histtype='stepfilled')
plt.title("Posterior distribution of probability of defect, given $t = 31$")
plt.xlabel("probability of defect occurring in O-ring")
plt.show()
