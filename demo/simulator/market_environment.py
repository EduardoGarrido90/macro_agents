import gymnasium
from gymnasium import spaces
import numpy as np

PRODUCTION_NOISE=0.05

# Create the MarketEnv environment
class MarketEnv(gymnasium.Env):
    def __init__(self, max_actions, min_prod, num_competitors, initial_price, max_fixed_costs, min_fixed_costs, cost_coef_0, cost_coef_1, cost_coef_2, cost_coef_3):
        super(MarketEnv, self).__init__()

        self.production_limit_per_producer = max_actions
        
        self.minimum_production = min_prod
        self.competitors_production_range = (self.minimum_production, self.production_limit_per_producer-1)  # Each competitor can produce 0 to 13 units
        self.num_competitors = num_competitors 

        # Competitors' actions (they also produce a random quantity within their range)
        self.competitors_quantities = np.random.randint(self.competitors_production_range[0],
                                                   self.competitors_production_range[1]+1,
                                                   self.num_competitors)

        # Action space for the producer (the agent): produce 0 to 10 units
        self.action_space = spaces.Discrete(self.production_limit_per_producer)

        # Observation space: [current price, total supply, total demand, progress action, units produced of competitors]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5+self.num_competitors,), dtype=np.float32)

        # Initial price and supply
        #self.price = 10.0
        self.price = initial_price
        self.total_supply = 0
        self.total_demand = 0

        # Fixed costs per day.
        self.max_fixed_costs = max_fixed_costs  # Maximum fixed costs when production is 0
        self.min_fixed_costs = min_fixed_costs   # Minimum fixed costs when production is high
        
        self.fixed_costs_company = np.random.normal(6.0, 1.0) #Can be higher if production is lower. 

        # Total production curve coefficients
        self.cost_coefficients = np.array([cost_coef_0, cost_coef_1, cost_coef_2, cost_coef_3])
        self.previous_action = 0
        self.up = 0
        self.equal = 1
        self.down = 2
        self.progress_action = self.equal
        self.timestep = 0  # Inicializas el contador en el constructor

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Reset the environment to the initial state
        #self.price = 10.0
        self.price = 15.0
        self.total_supply = 0
        self.total_demand = 0
        self.previous_action = 0
        self.timestep = 0
        self.progress_action = self.equal
        self.competitors_quantities = np.random.randint(self.competitors_production_range[0],
                                                   self.competitors_production_range[1]+1,
                                                   self.num_competitors)

        self.price_log = np.array([])
        self.total_supply_log = np.array([])
        self.total_demand_log = np.array([])
        self.progress_action_log = np.array([])

        # Return the initial observation: [price, total_supply, total_demand, units produced of competitors]
        observation = np.array([self.price, self.total_supply, self.total_demand, self.progress_action, self.timestep % 100], dtype=np.float32)
        observation = np.append(observation, self.competitors_quantities)
        return observation, {}


    def step(self, action):

        # Producer (agent) action: how much to produce (quantity)
        producer_quantity = action

        self.timestep += 1

        #Simulate competitors quantities as response to our previous action in a random walk.
        if producer_quantity > self.previous_action:
            random_steps = np.random.choice([-1, 0], size=self.num_competitors)
            self.progress_action = self.up
        elif producer_quantity == self.previous_action:
            random_steps = np.random.choice([-1, 0, 1], size=self.num_competitors)
            self.progress_action = self.equal
        else:
            random_steps = np.random.choice([0, 1], size=self.num_competitors)
            self.progress_action = self.down

        self.previous_action = producer_quantity

        # Update the competitors' quantities based on the random walk
        self.competitors_quantities += random_steps

        # Ensure the quantities stay within the defined range (0 to 14)
        self.competitors_quantities = np.clip(self.competitors_quantities, 0, self.production_limit_per_producer)

        # Calculate total supply (producer + competitors)
        self.total_supply = producer_quantity + self.competitors_quantities.sum()

        # Demand function: assume a linear demand curve (demand decreases as price increases)
        base_demand = self.production_limit_per_producer * 3.1  # Maximum demand when price is 0
        elasticity = 1.02  # Aumentar este valor hace que la demanda sea más sensible a los cambios de precio
        #self.total_demand = max(0, base_demand - (self.price ** elasticity))
        demand_fluctuation = np.random.normal(0, self.total_demand * 0.01)
        self.total_demand = max(0, base_demand - 3 * self.price ** elasticity + demand_fluctuation) #TODO: La demanda deberia ser cuadratica, y ademas es baja.
        
        # Variacion sinusoidal de la demanda estacional inelastica a precio.
        demand_variation = np.sin((self.timestep / 100) * 200 * np.pi) * 0.05  # Un pequeño patrón cíclico para la demanda.
        self.total_demand = self.total_demand * (1.0 + demand_variation)


        # Cubic production cost for the agent, falta el logaritmo en base 1.1 para hacerla mas plana.
        production_cost = (self.cost_coefficients[3] * (producer_quantity ** 3) + self.cost_coefficients[2] * (producer_quantity ** 2) + self.cost_coefficients[1] * producer_quantity)*8.0
        production_cost = np.random.normal(loc=production_cost, scale=production_cost*PRODUCTION_NOISE)
        
        # Costos de insumos fluctúan en función de la oferta total y de la época del año.
        # Se implementa como una variación sinusoidal que modifica el coste de produccion.
        supply_variation = np.cos((self.timestep / 100) * 200 * np.pi) * 0.05  # Un pequeño patrón cíclico.
        production_cost = production_cost * (1.0 + supply_variation)

        if producer_quantity > self.total_demand:
            excess_units = producer_quantity - self.total_demand
            storage_factor = 2
            storage_penalty = excess_units ** 2 * storage_factor  # Penalización cuadrática por exceso
            production_cost += storage_penalty

        # Adjust price based on the market-clearing condition and production cost
        if self.total_demand > self.total_supply:
            price_adjustment = (self.total_demand - self.total_supply) / (self.total_supply + 1) * 0.05
            #self.price = min(self.price + price_adjustment, 250) # Price cap at 250
            self.price = min(self.price + 1, 200) # Price cap at 200. Because of all the competitors.
        else:
            price_adjustment = (self.total_supply - self.total_demand) / (self.total_demand + 1) * 0.05
            #self.price = max(self.price - price_adjustment, 1)
            self.price = max(self.price - 1, 1)


        # Producer's revenue
        producer_revenue = self.price * producer_quantity

        # Fixed costs. Higher if production is not done. 
        self.fixed_costs_company = np.random.normal(self.min_fixed_costs + (self.max_fixed_costs - self.min_fixed_costs) * np.exp(-0.5 * producer_quantity), 1.0)

        # Calculate producer's profit: revenue - cost
        producer_profit = producer_revenue - production_cost - self.fixed_costs_company

        # Brand effects: if more production is done, the profit is incremented in a percentage.
        # Logarithmic brand effect on profit
        max_brand_effect = 0.3  # Maximum 30% profit increase due to brand effect

        # Calculate the brand effect as a logarithmic function of production level
        brand_effect_percentage = np.random.normal(max_brand_effect * np.log(1 + producer_quantity), 1)

        # Calculate the profit with brand effects
        producer_profit = producer_profit * (1 + brand_effect_percentage)

        # Maximum subsidy at the highest production level (e.g., 13 units)
        max_subsidy = 10.0

        # Linear scaling of subsidy based on production level
        subsidy = np.random.normal(max_subsidy * (self.production_limit_per_producer / 13.0))

        # Apply the subsidy to the profit
        producer_profit += subsidy

        # Observation: [price, total supply, total demand, production of competitors]
        observation = np.array([self.price, self.total_supply, self.total_demand, self.progress_action, self.timestep % 100], dtype=np.float32)
        observation = np.append(observation, self.competitors_quantities)

        # Reward is the producer's profit scaled to force the agent to retrieve good results in training time.
        #if producer_profit > 0:
            #reward = producer_profit * 2.0
        #else:
            #reward = producer_profit * 3.0
        reward = producer_profit

        # No termination condition, so done is always False for now
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, {"price" : self.price, "supply" : self.total_supply, "demand" : self.total_demand, "progress" : self.progress_action}

    
    def render(self, mode='human'):
        # Render the current state
        print(f"Price: {self.price}, Total Supply: {self.total_supply}, Total Demand: {self.total_demand}")


