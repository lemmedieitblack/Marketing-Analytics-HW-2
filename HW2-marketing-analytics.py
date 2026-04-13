############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ABSTRACT CLASS (DO NOT TOUCH)

class Bandit(ABC):
    """
Abstract base class for multi-armed bandit algorithms.
Defines the required interface for all bandit strategies.
"""

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        pass


# EPSILON GREEDY

class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy algorithm for solving the multi-armed bandit problem.
    Balances exploration and exploitation using a decaying epsilon.
    """

    def __init__(self, p):
        self.p = p
        self.k = len(p)
        self.counts = np.zeros(self.k)
        self.values = np.zeros(self.k)
        self.total_reward = 0
        self.rewards = []
        self.regret = []
        self.name = "EpsilonGreedy"

    def __repr__(self):
        return f"EpsilonGreedy Bandit with {self.k} arms"

    def pull(self, arm):
        """
        Pulls the selected bandit arm and returns a reward.

        Args:
            arm (int): Index of the chosen bandit.

        Returns:
            float: Reward sampled from the bandit's distribution.
        """
        return np.random.normal(self.p[arm], 1)

    def update(self, arm, reward):
        """
        Updates the estimated value of the selected arm.
        Args:
            arm (int): Index of the chosen bandit.
            reward (float): Observed reward.
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]

        # incremental mean update
        self.values[arm] = ((n - 1) * value + reward) / n

    def experiment(self, trials=20000):
        """
        Runs the bandit experiment over a number of trials.

        Args:
            trials (int): Number of iterations (default: 20000).
        """
        best = np.max(self.p)

        for t in range(1, trials + 1):
            epsilon = 1 / t  # decay

            if np.random.rand() < epsilon:
                arm = np.random.randint(self.k)
            else:
                arm = np.argmax(self.values)

            reward = self.pull(arm)
            self.update(arm, reward)

            self.total_reward += reward
            self.rewards.append(self.total_reward)

            regret = best - self.p[arm]
            self.regret.append(regret if len(self.regret) == 0 else self.regret[-1] + regret)

    def report(self):
        """
        Saves results to CSV and logs total reward and regret.
        """
        df = pd.DataFrame({
            "Bandit": np.arange(len(self.rewards)),
            "Reward": self.rewards,
            "Algorithm": self.name
        })

        df.to_csv("epsilon_greedy.csv", index=False)

        logger.info(f"{self.name} Total Reward: {self.total_reward}")
        logger.info(f"{self.name} Total Regret: {self.regret[-1]}")


# THOMPSON SAMPLING

class ThompsonSampling(Bandit):

    """
    Thompson Sampling algorithm using a Bayesian approach.
    Selects actions based on sampled reward distributions.
    """

    def __init__(self, p):
        self.p = p
        self.k = len(p)
        self.precision = 1  # known precision
        self.means = np.zeros(self.k)
        self.lambda_ = np.ones(self.k)

        self.total_reward = 0
        self.rewards = []
        self.regret = []
        self.name = "ThompsonSampling"

    def __repr__(self):
        return f"ThompsonSampling Bandit with {self.k} arms"

    def pull(self, arm):
        """
        Pulls the selected bandit arm and returns a reward.

        Args:
            arm (int): Index of the chosen bandit.

        Returns:
            float: Reward sampled from the bandit's distribution.
        """
        return np.random.normal(self.p[arm], 1)

    def update(self, arm, reward):
        """
        Updates the Bayesian estimate (posterior approximation) for the selected arm.

        Args:
            arm (int): Index of the chosen bandit.
            reward (float): Observed reward.
        """
        self.lambda_[arm] += self.precision
        self.means[arm] = (
            (self.lambda_[arm] - self.precision) * self.means[arm] + self.precision * reward
        ) / self.lambda_[arm]

    def experiment(self, trials=20000):
        """
        Runs Thompson Sampling over a number of trials.

        Args:
            trials (int): Number of iterations (default: 20000).
        """
        best = np.max(self.p)

        for t in range(trials):
            samples = np.random.normal(self.means, 1 / np.sqrt(self.lambda_))
            arm = np.argmax(samples)

            reward = self.pull(arm)
            self.update(arm, reward)

            self.total_reward += reward
            self.rewards.append(self.total_reward)

            regret = best - self.p[arm]
            self.regret.append(regret if len(self.regret) == 0 else self.regret[-1] + regret)

    def report(self):
        """
        Saves results to CSV and logs total reward and regret.
        """
        df = pd.DataFrame({
            "Bandit": np.arange(len(self.rewards)),
            "Reward": self.rewards,
            "Algorithm": self.name
        })

        df.to_csv("thompson_sampling.csv", index=False)

        logger.info(f"{self.name} Total Reward: {self.total_reward}")
        logger.info(f"{self.name} Total Regret: {self.regret[-1]}")


# VISUALIZATION

class Visualization():
    """
    Handles visualization of cumulative rewards and regrets
    for bandit algorithms comparison.
    """

    def plot1(self, eg, ts):
        """
        Plots cumulative rewards for Epsilon-Greedy vs Thompson Sampling.

        Args:
            eg (EpsilonGreedy): Epsilon-Greedy algorithm instance.
            ts (ThompsonSampling): Thompson Sampling instance.
        """
        plt.figure()
        plt.plot(eg.rewards, label="Epsilon Greedy")
        plt.plot(ts.rewards, label="Thompson Sampling")
        plt.title("Cumulative Rewards")
        plt.legend()
        plt.show()

    def plot2(self, eg, ts):
        """
        Plots cumulative regret for Epsilon-Greedy vs Thompson Sampling.

        Args:
            eg (EpsilonGreedy): Epsilon-Greedy algorithm instance.
            ts (ThompsonSampling): Thompson Sampling instance.
        """
        plt.figure()
        plt.plot(eg.regret, label="Epsilon Greedy")
        plt.plot(ts.regret, label="Thompson Sampling")
        plt.title("Cumulative Regret")
        plt.legend()
        plt.show()


# COMPARISON

def comparison():
    """
    Runs both Epsilon-Greedy and Thompson Sampling experiments
    and visualizes their performance comparison.
    """
    Bandit_Reward = [1, 2, 3, 4]

    eg = EpsilonGreedy(Bandit_Reward)
    ts = ThompsonSampling(Bandit_Reward)

    eg.experiment()
    ts.experiment()

    eg.report()
    ts.report()

    viz = Visualization()
    viz.plot1(eg, ts)
    viz.plot2(eg, ts)


# MAIN

if __name__ == '__main__':

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    comparison()