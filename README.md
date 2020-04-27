# alpha-regex

This repo implement AlphaGo Zero style regular expression generation for optimizing F1 score on a dataset of goog/suspicious http requests in the domain of cybersecurity.

I implemented a regular expression generator in a markov decision process style. At each time step, the Monte Carlo Tree Search does rollouts with a policy and a value network as in BinPacking Algorithm and select the next best character to expand the regular expression accordingly to its estimation of performance on differentiating goog from bad requests.

* AlphaGo Zero: https://www.nature.com/articles/nature24270
* BinPacking: https://arxiv.org/abs/1807.01672

To train the policy:
```bash
python request_trainer.py
```

See Exemple_notebook for more information.
