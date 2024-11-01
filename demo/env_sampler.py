from config.config_parser import ConfigParser

# Example usage
if __name__ == '__main__': 
    config_parser = ConfigParser('./config/config_scenario.json')
    parameters = config_parser.get_parameters()
    print(parameters)

    sampled_config = config_parser.sample_configuration()
    print(sampled_config)
