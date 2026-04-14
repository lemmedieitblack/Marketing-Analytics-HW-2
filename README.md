# A/B Testing with Multi-Armed Bandits

## Experiment Setup
- Bandits: [1, 2, 3, 4]  
- Number of trials: 20000  
- Algorithms implemented:
  - Epsilon-Greedy (ε = 1/t)
  - Thompson Sampling (known precision)


## 1. Learning Process Visualization
The learning process for both algorithms is visualized using cumulative reward plots.

<img width="952" height="708" alt="image" src="https://github.com/user-attachments/assets/867b976c-5ef0-4eb6-8443-00d8b90e04f3" />


## 2. Performance Comparison
The performance of Epsilon-Greedy and Thompson Sampling is compared using cumulative regret over time.

<img width="954" height="701" alt="image" src="https://github.com/user-attachments/assets/89fbe870-f3fa-4782-aa46-a0a5ac37a640" />



## 3. CSV Storage
The rewards are stored in CSV files with the following format:

- Columns: `Bandit`, `Reward`, `Algorithm`
- Files generated:
  - `epsilon_greedy.csv`
  - `thompson_sampling.csv`


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
