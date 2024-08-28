import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

class Bandits(object):

    """
    This class represents N bandits machines.

    parameters:
        p_array: a (n,) Numpy array of probabilities >0, <1.

    methods:
        pull( i ): return the results, 0 or 1, of pulling 
                   the ith bandit.
    """

    def __init__(self, p_array):
        self.p = p_array
        self.optimal = np.argmax(p_array)

    def pull(self, i):
        # i is which arm to pull
        return np.random.rand() < self.p[i]

    def __len__(self):
        return len(self.p)

class BayesianStrategy(object):

    """
    Implements a online, learning strategy to solve
    the Multi-Armed Bandit problem.
    
    parameters:
        bandits: a Bandit class with .pull method
    
    methods:
        sample_bandits(n): sample and train on n pulls.

    attributes:
        N: the cumulative number of samples
        choices: the historical choices as a (N,) array
        bb_score: the historical score as a (N,) array
    """

    def __init__(self, bandits):

        self.bandits = bandits
        n_bandits = len(self.bandits)
        self.wins = np.zeros(n_bandits)
        self.trials = np.zeros(n_bandits)
        self.N = 0
        self.choices = []
        self.bb_score = []

    def sample_bandits(self, n=1):

        bb_score = np.zeros(n)
        choices = np.zeros(n)

        P0_samples = [np.random.rand()]
        P1_samples = [np.random.rand()]
        P2_samples = [np.random.rand()]

        for k in range(n):
            # sample from the bandits's priors, and select the largest sample

            choice = np.argmax([np.random.choice(P0_samples), np.random.choice(P1_samples), np.random.choice(P2_samples)])

            print();
            print();
            print(k, choice);

            # sample the chosen bandit
            result = self.bandits.pull(choice)

            # update priors and score
            self.wins[choice] += result
            self.trials[choice] += 1
            bb_score[k] = result
            self.N += 1
            choices[k] = choice

            if choice == 0:
                with pm.Model() as model0:  
                    P0 = pm.Uniform('P0', 0, 1)

                    X0 = pm.Binomial('X0', n = self.trials[0]+1, p = P0, observed = self.wins[0])

                    trace = pm.sample(15000, tune=5000, chains=1, return_inferencedata=False)

                    P0_samples = trace["P0"]

            elif choice == 1:
                with pm.Model() as model1:  
                    P1 = pm.Uniform('P1', 0, 1)

                    X1 = pm.Binomial('X1', n = self.trials[1]+1, p = P1, observed = self.wins[1])

                    trace = pm.sample(15000, tune=5000, chains=1, return_inferencedata=False)

                    P1_samples = trace["P1"]

            else:
                with pm.Model() as model2:  
                    P2 = pm.Uniform('P2', 0, 1)

                    X2 = pm.Binomial('X2', n = self.trials[2]+1, p = P2, observed = self.wins[2])

                    trace = pm.sample(15000, tune=5000, chains=1, return_inferencedata=False)

                    P2_samples = trace["P2"]


        self.bb_score = np.r_[self.bb_score, bb_score]
        self.choices = np.r_[self.choices, choices]
        return

hidden_prob = np.array([0.85, 0.60, 0.75])
bandits = Bandits(hidden_prob)

bayesian_strat = BayesianStrategy(bandits)

bayesian_strat.sample_bandits(10000)

def regret(probabilities, choices):
    w_opt = probabilities.max()
    return (w_opt - probabilities[choices.astype(int)]).cumsum()

_regret = regret(hidden_prob, bayesian_strat.choices)
plt.plot(_regret, lw=3)

plt.title("Total Regret of Bayesian Bandits Strategy (MCMC)")
plt.xlabel("Number of pulls")
plt.ylabel("Regret after $n$ pulls")
plt.show()
