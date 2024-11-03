from config.config_parser import ConfigParser
import numpy as np

def create_shell_script(parameters):
    # Open the file in write mode
    file_name = "run_sampled_experiment.sh"
    with open(file_name, "w") as file:
        # Write the shebang line
        file.write("#!/bin/bash\n\n")
        
        # Command to clear results
        file.write("rm -rf ../results/*\n\n")
        
        # Start building the Python command
        command = "python3 ../main.py"
        
        # Add parameters to the command
        for key, value in parameters.items():
            command += f" --{key} {value}"
        
        # Write the command to the file
        file.write(command + "\n")
    
    # Make the .sh file executable (for Unix-based systems)
    import os
    os.chmod(file_name, 0o755)

if __name__ == '__main__':
    n_samples = 25

    config_parser = ConfigParser('../config/config_scenario.json')
    #parameters = config_parser.get_parameters()
    #print(parameters)
    environments = []
    for i in range(n_samples):
        environments.append(config_parser.sample_configuration())
    print(environments)

    for i in range(n_samples):
        #Train all the agents in the 25 samples.
        #1. Generate the sh file.
        create_shell_script(environments[i])
        #TODO 2. Execute the sh file and wait for the result.
        #TODO 3. Read the result and continue.
    


    #TODO: Perform statistical hypothesis testing.
