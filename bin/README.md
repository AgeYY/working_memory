# Miscellaneous

- There are many stop training conditions in /core/default.py, including stop\_cost, stop\_noise\_error, stop\_xxx etc. However, we set them properly so that the training always terminates after achieving the min\_trials (2e5) trials when we observed a relatively stable memory error. The purpose of this project is not high memory accuracy anyway.

- bump.py and simple\_fig.py: neural activity depends not only on the prior distribution but also on the trial stimulus and the stochastic training process. Persistent neural activity can be observed even in the biased distribution if the trajectory is unluckily near the attractors. To reproduce the transient neural activity, one can try using a different stimulus or use another model (thus different distribution of fixpoints).

- rnn\_noise\_bay\_drift.py: the algorithm was designed for weak drift force (usually implies a more uniform prior distribution), hence sigma = 12.5 may not be fitted properly. One could fine-tune the hyperparameters or simply exclude this prior distribution.



