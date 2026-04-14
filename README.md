# A/B Testing with Multi-Armed Bandits

## Experiment Setup
- Bandits: [1, 2, 3, 4]  
- Number of trials: 20000  
- Algorithms implemented:
  - Epsilon-Greedy (ε = 1/t)
  - Thompson Sampling (known precision)


## 1. Learning Process Visualization
The learning process for both algorithms is visualized using cumulative reward plots.

> Insert plot1() screenshot here


## 2. Cumulative Rewards Comparison
The cumulative rewards of Epsilon-Greedy and Thompson Sampling are plotted together for comparison.

> Insert combined reward plot here


## 3. CSV Storage
The rewards are stored in CSV files with the following format:

- Columns: `Bandit`, `Reward`, `Algorithm`
- Files generated:
  - `epsilon_greedy.csv`
  - `thompson_sampling.csv`

---

## 4. Cumulative Reward

- **Epsilon-Greedy:** 80045.37  
- **Thompson Sampling:** 80027.68  


## 5. Cumulative Regret

- **Epsilon-Greedy:** 7  
- **Thompson Sampling:** 50  


## Conclusion
- Epsilon-Greedy achieved a slightly higher cumulative reward in this run.
- Epsilon-Greedy also resulted in lower cumulative regret.
- In this specific experiment, Epsilon-Greedy performed better than Thompson Sampling.
