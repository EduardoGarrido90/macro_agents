from config.config_parser import ConfigParser
import numpy as np
# Example usage
if __name__ == '__main__':
    n_samples = 25

    config_parser = ConfigParser('./config/config_scenario.json')
    parameters = config_parser.get_parameters()
    print(parameters)
    environments = []
    for i in range(n_samples):
        environments.append(config_parser.sample_configuration())
    print(environments)

    #TODO: Train all the agents in the 25 samples.
    #TODO: Recopilate the result of the agents.
    #TODO: Perform statistical hypothesis testing.
