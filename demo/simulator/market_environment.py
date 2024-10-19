import gymnasium
from gymnasium import spaces
import numpy as np

PRODUCTION_NOISE=0.05

# Create the MarketEnv environment
class MarketEnv(gymnasium.Env):
    def __init__(self, max_actions):
        super(MarketEnv, self).__init__()

        self.production_limit_per_producer = max_actions
        
        self.competitors_production_range = (0, self.production_limit_per_producer-1)  # Each competitor can produce 0 to 13 units
        self.num_competitors = 3 #Needs to be parametrized.

        # Competitors' actions (they also produce a random quantity within their range)
        self.competitors_quantities = np.random.randint(self.competitors_production_range[0],
                                                   self.competitors_production_range[1]+1,
                                                   self.num_competitors)

        # Action space for the producer (the agent): produce 0 to 10 units
        self.action_space = spaces.Discrete(self.production_limit_per_producer)

        # Observation space: [current price, total supply, total demand, units produced of competitors]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3+self.num_competitors,), dtype=np.float32)

        # Initial price and supply
        self.price = 10.0
        self.total_supply = 0
        self.total_demand = 0

        # Fixed costs per day.
        self.fixed_costs_company = -100.0

        # Total production curve coefficients
        self.cost_coefficients = np.array([0.0, 4.0, -0.6, 0.03])


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        # Reset the environment to the initial state
        self.price = 10.0
        self.total_supply = 0
        self.total_demand = 0
        self.competitors_quantities = np.random.randint(self.competitors_production_range[0],
                                                   self.competitors_production_range[1]+1,
                                                   self.num_competitors)

        # Return the initial observation: [price, total_supply, total_demand, units produced of competitors]
        observation = np.array([self.price, self.total_supply, self.total_demand], dtype=np.float32)
        observation = np.append(observation, self.competitors_quantities)
        return observation, {}


    def step(self, action):

        # Producer (agent) action: how much to produce (quantity)
        producer_quantity = action

        #Simulate competitors quantities.
        # Generate random steps for each competitor: -1, 0, or +1
        random_steps = np.random.choice([-1, 0, 1], size=self.num_competitors)

        # Update the competitors' quantities based on the random walk
        self.competitors_quantities += random_steps

        # Ensure the quantities stay within the defined range (0 to 14)
        self.competitors_quantities = np.clip(self.competitors_quantities, 0, self.production_limit_per_producer)

        # Calculate total supply (producer + competitors)
        self.total_supply = producer_quantity + self.competitors_quantities.sum()

        # Demand function: assume a linear demand curve (demand decreases as price increases)
        base_demand = self.production_limit_per_producer * 3.5  # Maximum demand when price is 0
        self.total_demand = max(0, base_demand - 3 * self.price)

        # Cubic production cost for the agent, falta el logaritmo en base 1.1 para hacerla mas plana.
        production_cost = (self.cost_coefficients[3] * (producer_quantity ** 3) + self.cost_coefficients[2] * (producer_quantity ** 2) + self.cost_coefficients[1] * producer_quantity)*8.0
        production_cost = np.random.normal(loc=production_cost, scale=production_cost*PRODUCTION_NOISE)


        # Adjust price based on the market-clearing condition and production cost
        if self.total_demand > self.total_supply:
            self.price = min(self.price + 1, 200)  # Price cap at 200
            #self.price = min(self.price + 1, 100, production_cost+1)  # Price cap at 100
        else:
            #self.price = max(self.price - 1, production_cost+1)  # Minimum price of 1
            self.price = max(self.price - 1, 1)  # Minimum price of 1


        # Producer's revenue
        producer_revenue = self.price * producer_quantity

        # Calculate producer's profit: revenue - cost
        producer_profit = producer_revenue - production_cost - self.fixed_costs_company

        # Observation: [price, total supply, total demand, production of competitors]
        observation = np.array([self.price, self.total_supply, self.total_demand], dtype=np.float32)
        observation = np.append(observation, self.competitors_quantities)

        # Reward is the producer's profit
        reward = producer_profit

        # No termination condition, so done is always False for now
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        # Render the current state
        print(f"Price: {self.price}, Total Supply: {self.total_supply}, Total Demand: {self.total_demand}")




