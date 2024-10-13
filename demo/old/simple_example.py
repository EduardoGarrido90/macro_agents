import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Create the MarketEnv environment
class MarketEnv(gymnasium.Env):
    def __init__(self):
        super(MarketEnv, self).__init__()
        
        # Action space for the producer (the agent): produce 0 to 10 units
        self.action_space = spaces.Discrete(11)
        
        # Observation space: [current price, total supply, total demand]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        
        # Initial price and supply
        self.price = 10.0
        self.total_supply = 0
        self.total_demand = 0
        
        # Quadratic cost coefficient for the producer
        self.cost_coefficient = 1.0
        
        # Number of competitors and their production ranges
        self.num_competitors = 3
        self.competitors_production_range = (0, 10)  # Each competitor can produce 0 to 10 units

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
        
        # Cubic production cost for the agent
        # TODO: make this more complex ax**3 + bx**2 + cx + d
        production_cost = self.cost_coefficient * (producer_quantity ** 3)
        
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

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use the 'Agg' backend for rendering without a display

# Create and check the environment
env = MarketEnv()
check_env(env)

# Set up the PPO model for the environment
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Test the agent
obs, info = env.reset()
ppo_performance = 0.0
accumulated_profits = [0.0]
steps = [0]
for step in range(10):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    ppo_performance += reward
    accumulated_profits.append(ppo_performance)
    steps.append(step+1)
    print(f"Step: {step+1}, Action: {action}, Reward: {reward}")

    if terminated or truncated:
            obs, info = env.reset()
print(f"The accumulated profit of the PPO agent is: {ppo_performance}")

# Plot the accumulated profit over time
plt.figure(figsize=(10, 6))
plt.plot(steps, accumulated_profits, marker='o', linestyle='-', color='b', label='Accumulated Profit')
plt.title('Accumulated Profit of the Agent Over Test Steps')
plt.xlabel('Step')
plt.ylabel('Accumulated Profit')
plt.grid(True)
plt.legend()

# Save the plot to a PDF file
plt.savefig('profit_evolution.pdf')

# Close the figure if you don't need it anymore
plt.close()
