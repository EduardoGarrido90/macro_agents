import json
import random

class ConfigParser:
    def __init__(self, config_file):
        self.config_data = {}
        self.load_config(config_file)

    def load_config(self, config_file):
        """Loads the configuration file and stores parameters in a dictionary."""
        try:
            with open(config_file, 'r') as file:
                data = json.load(file)
                # Extract parameters into a dictionary
                for param, details in data.get("parameters", {}).items():
                    self.config_data[param] = {
                        "type": details.get("type"),
                        "lower": details.get("lower"),
                        "upper": details.get("upper")
                    }
        except FileNotFoundError:
            print(f"Error: {config_file} not found.")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse {config_file}. Please ensure it's valid JSON.")

    def get_parameters(self):
        """Returns the parsed configuration data as a dictionary."""
        return self.config_data

    def sample_configuration(self):
        """Samples a random configuration from the stored parameters with correct data types."""
        sampled_config = {}
        for param, details in self.config_data.items():
            if details["type"] == "int":
                sampled_config[param] = random.randint(details["lower"], details["upper"])
            elif details["type"] == "float":
                sampled_config[param] = random.uniform(details["lower"], details["upper"])
        return sampled_config

