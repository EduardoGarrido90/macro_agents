import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from visual.metrics_plotter import MetricsPlotter

class MetricsCallback(BaseCallback):
    def __init__(self, num_agents, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.num_agents = num_agents
        # Initialize lists for each agent
        self.losses = [[] for _ in range(num_agents)]
        self.value_losses = [[] for _ in range(num_agents)]
        self.approx_kls = [[] for _ in range(num_agents)]
        self.explained_variance = [[] for _ in range(num_agents)]
        self.timesteps = []

    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        for agent_id in range(self.num_agents):
            # Get rewards, returns, and values for this agent
            agent_rewards = self.model.rollout_buffer.rewards[:, agent_id]
            agent_returns = self.model.rollout_buffer.returns[:, agent_id]
            agent_values = self.model.rollout_buffer.values[:, agent_id]
            agent_advantages = self.model.rollout_buffer.advantages[:, agent_id]

            # Calculate value loss (MSE between returns and values)
            agent_value_loss = np.mean((agent_returns - agent_values) ** 2)
            self.value_losses[agent_id].append(agent_value_loss)

            # Policy loss: You would need access to the policy probabilities to calculate this.
            # This is an example of a placeholder (PPO calculates this during the learning step)
            agent_loss = np.mean(agent_advantages)  # Placeholder, needs actual policy loss calc
            self.losses[agent_id].append(agent_loss)

            # Approx KL divergence (you will need old and new policy probs for this calculation)
            agent_approx_kl = np.mean(np.abs(agent_advantages))  # Placeholder, needs real KL
            self.approx_kls[agent_id].append(agent_approx_kl)

            # Explained variance
            agent_exp_var = 1 - np.var(agent_returns - agent_values) / np.var(agent_returns)
            self.explained_variance[agent_id].append(agent_exp_var)

        self.timesteps.append(self.num_timesteps)

from simulator.market_environment import MarketEnv
def make_env():
    env = MarketEnv()
    check_env(env)
    return env

from typing import Callable
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv
from config.argument_parser import ArgumentParser
matplotlib.use('Agg')  # Use the 'Agg' backend for rendering without a display

if __name__ == '__main__':
    
    # Encapsulate the argument parsing in the ArgumentParser class
    argument_parser = ArgumentParser()
    args = argument_parser.parse_args()
    agents_number = args.agents_number
    test_periods = args.test_periods
    total_training_timesteps = args.total_training_timesteps
    seed = args.seed

    env = SubprocVecEnv([make_env for _ in range(agents_number)])

    # Set up the PPO model for the environment
    model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=1, seed=seed)

    metrics_callback = MetricsCallback(agents_number)

    # Train the agent
    model.learn(total_timesteps=total_training_timesteps, callback=metrics_callback, progress_bar=False)

    # Test the agent
    obs = env.reset()
    ppo_performance = 0.0
    ppo_performance = [0.0] * agents_number  # Store performance for each agent
    accumulated_profits = [[0.0] for _ in range(agents_number)]  # One list per agent
    steps = [0]
    for step in range(test_periods):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated = env.step(action)
        env.render()
        # Update performance for each agent
        for i in range(agents_number):
            ppo_performance[i] += reward[i]
            accumulated_profits[i].append(ppo_performance[i])
        steps.append(step+1)
        print(f"Step: {step+1}, Action: {action}, Reward: {reward}")

        #Commented because from the moment it never ends and it is a vector of environments.
        if any(terminated) or any(truncated):
                obs = env.reset()
                print(f"Reset environments where agents are done")
    
    for i in range(agents_number):
        print(f"The accumulated profit of the PPO agent {i+1} is: {ppo_performance[i]}")


    #We plot all the training progress of the agents and the performance in a new scenario. 
    MetricsPlotter(metrics_callback, agents_number, accumulated_profits, steps, plot_everything=True)
