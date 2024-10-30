import argparse

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Process some parameters.")
        self._add_arguments()

    def _add_arguments(self):
        # Add all the arguments here
        self.parser.add_argument('--agents_number', type=int, default=10, help='Number of agents')
        self.parser.add_argument('--test_periods', type=int, default=200, help='Number of test periods')
        self.parser.add_argument('--total_training_timesteps', type=int, default=50000, help='Total training timesteps')
        self.parser.add_argument('--seed', type=int, default=1, help='Random seed')
        self.parser.add_argument('--max_actions', type=int, default=14, help='Maximum number of actions')
        self.parser.add_argument('--min_prod', type=int, default=0, help='Minimum number of actions')
        self.parser.add_argument('--num_competitors', type=int, default=3, help='Number of competitors')
        self.parser.add_argument('--initial_price', type=float, default=15, help='Initial price of the asset')
        self.parser.add_argument('--max_fixed_costs', type=float, default=10, help='Max fixed costs')
        self.parser.add_argument('--min_fixed_costs', type=float, default=1, help='Min fixed costs')
        self.parser.add_argument('--cost_coef_0', type=float, default=0, help='Constant cost coef')
        self.parser.add_argument('--cost_coef_1', type=float, default=4.0, help='Linear cost coef')
        self.parser.add_argument('--cost_coef_2', type=float, default=-0.6, help='Quad cost coef')
        self.parser.add_argument('--cost_coef_3', type=float, default=0.03, help='Cubic cost coef')
        self.parser.add_argument('--elasticity', type=float, default=1.02, help='Elasticity')
        self.parser.add_argument('--base_demand', type=float, default=3.1, help='Base demand')
        self.parser.add_argument('--prod_noise', type=float, default=0.05, help='Prod noise')
        self.parser.add_argument('--storage_factor', type=float, default=2, help='Storage factor')
        self.parser.add_argument('--brand_effect', type=float, default=0.3, help='Brand effect')
        self.parser.add_argument('--max_subsidy', type=float, default=10, help='Subsidies')
        self.parser.add_argument('--number_random_agents', type=int, default=5, help='Random agents to compare')
        self.parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')

    def parse_args(self):
        # Parse and return the arguments
        return self.parser.parse_args()
