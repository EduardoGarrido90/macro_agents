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

    def parse_args(self):
        # Parse and return the arguments
        return self.parser.parse_args()
