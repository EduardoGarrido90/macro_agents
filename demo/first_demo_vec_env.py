import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

PRODUCTION_NOISE=0.05

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

# Create the MarketEnv environment
class MarketEnv(gymnasium.Env):
    def __init__(self):
        super(MarketEnv, self).__init__()
        
        # Action space for the producer (the agent): produce 0 to 10 units
        self.action_space = spaces.Discrete(14)
        
        # Observation space: [current price, total supply, total demand]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        
        # Initial price and supply
        self.price = 10.0
        self.total_supply = 0
        self.total_demand = 0
        
        # Total production curve coefficients
        self.cost_coefficients = np.array([0.0, 4.0, -0.6, 0.03])
        
        # Number of competitors and their production ranges
        self.num_competitors = 3
        self.competitors_production_range = (0, 13)  # Each competitor can produce 0 to 13 units

    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)

        # Reset the environment to the initial state
        self.price = 10.0
        self.total_supply = 0
        self.total_demand = 0
        
        # Return the initial observation: [price, total_supply, total_demand]
        return np.array([self.price, self.total_supply, self.total_demand], dtype=np.float32), {}

    def step(self, action):
        
        # Producer (agent) action: how much to produce (quantity)
        producer_quantity = action
        
        # Competitors' actions (they also produce a random quantity within their range)
        competitors_quantities = np.random.randint(self.competitors_production_range[0], 
                                                   self.competitors_production_range[1]+1, 
                                                   self.num_competitors)
        
        # Calculate total supply (producer + competitors)
        self.total_supply = producer_quantity + competitors_quantities.sum()
        
        # Demand function: assume a linear demand curve (demand decreases as price increases)
        base_demand = 50  # Maximum demand when price is 0
        self.total_demand = max(0, base_demand - 3 * self.price)
        
        # Adjust price based on the market-clearing condition
        if self.total_demand > self.total_supply:
            self.price = min(self.price + 1, 100)  # Price cap at 100
        else:
            self.price = max(self.price - 1, 1)  # Minimum price of 1
        
        # Cubic production cost for the agent, falta el logaritmo en base 1.1 para hacerla mas plana. 
        production_cost = (self.cost_coefficients[3] * (producer_quantity ** 3) + self.cost_coefficients[2] * (producer_quantity ** 2) + self.cost_coefficients[1] * producer_quantity)*8.0
        production_cost = np.random.normal(loc=production_cost, scale=production_cost*PRODUCTION_NOISE)
        
        # Producer's revenue
        producer_revenue = self.price * producer_quantity
        
        # Calculate producer's profit: revenue - cost
        producer_profit = producer_revenue - production_cost
        
        # Observation: [price, total supply, total demand]
        observation = np.array([self.price, self.total_supply, self.total_demand], dtype=np.float32)
        
        # Reward is the producer's profit
        reward = producer_profit
        
        # No termination condition, so done is always False for now
        terminated = False
        truncated = False
        
        return observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        # Render the current state
        print(f"Price: {self.price}, Total Supply: {self.total_supply}, Total Demand: {self.total_demand}")



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
matplotlib.use('Agg')  # Use the 'Agg' backend for rendering without a display

if __name__ == '__main__':
    # Create and check the environment
    # Create a vectorized environment with 8 instances
    agents_number = 3
    test_periods = 10
    #total_training_timesteps = 100000
    total_training_timesteps = 25000
    seed = 1
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


    # TODO: Vectorizar el resto como este.
    # Ensure that you have data to plot
    if metrics_callback.losses[0]:
        plt.figure(figsize=(10, 6))
        for agent in range(agents_number):
            # Plot Model Loss
            plt.plot(metrics_callback.timesteps, metrics_callback.losses[agent], label='Total Loss Agent ' + str(agent))
        plt.xlabel('Timesteps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Model Losses over Time')
        plt.savefig('results/model_loss.pdf')
    
    if metrics_callback.value_losses[0]:
        plt.figure(figsize=(10, 6))
        for agent in range(agents_number):
            # Plot Model Loss
            plt.plot(metrics_callback.timesteps, metrics_callback.losses[agent], label='Value Loss Agent ' + str(agent))
        plt.xlabel('Timesteps')
        plt.ylabel('Value Loss')
        plt.legend()
        plt.title('Value Losses over Time')
        plt.savefig('results/value_loss.pdf')

    
    # Plot Approximate KL Divergence
    if metrics_callback.approx_kls[0]:
        plt.figure(figsize=(10, 6))
        for agent in range(agents_number):
            plt.plot(metrics_callback.timesteps, metrics_callback.approx_kls[agent], label='Approx KL Agent ' + str(agent))
        plt.xlabel('Timesteps')
        plt.ylabel('Approx KL Divergence')
        plt.legend()
        plt.title('Approximate KL Divergence over Time')
        plt.savefig('results/approx_kl.pdf')

    # Plot Explained Variance
    if metrics_callback.explained_variance[0]:
        plt.figure(figsize=(10, 6))
        for agent in range(agents_number):
            plt.plot(metrics_callback.timesteps, metrics_callback.explained_variance[agent], label='Explained variance Agent ' + str(agent))
        plt.xlabel('Timesteps')
        plt.ylabel('Explained Variance')
        plt.legend()
        plt.title('Explained Variance over Time')
        plt.savefig('results/exp_var.pdf')

    # Plot the accumulated profit over time
    plt.figure(figsize=(10, 6))
    for agent in range(3):  # Assuming 3 agents
        plt.plot(steps, accumulated_profits[agent], marker='o', linestyle='-', label=f'Agent {agent+1} Accumulated Profit')
    plt.title('Accumulated Profit of each Agent Over Test Steps')
    plt.xlabel('Step')
    plt.ylabel('Accumulated Profit')
    plt.grid(True)
    plt.legend()

    # Save the plot to a PDF file
    plt.savefig('results/profit_evolution.pdf')

    # Close the figure if you don't need it anymore
    plt.close()
