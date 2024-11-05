from config.config_parser import ConfigParser
import numpy as np

file_name = "run_sampled_experiment.sh"

def create_shell_script(parameters):
    # Open the file in write mode
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

def execute_shell_script():
    # Run the shell script and wait for it to complete
    try:
        result = subprocess.run([file_name], capture_output=True, text=True, check=True)
        print("Script executed successfully.")
        print("Output:")
        print(result.stdout)  # Print the output of the script
    except subprocess.CalledProcessError as e:
        print("An error occurred while executing the script.")
        print("Error output:")
        print(e.stderr)  # Print the error output if the script fails

#Invoke this class as: python script_name.py n_samples
if __name__ == '__main__':
    if len(sys.argv) > 1:
        n_samples = int(sys.argv[1])
    else:
        n_samples = 25  # Default value if no argument is provided

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
        #2. Execute the sh file and wait for the result.
        execute_shell_script()
        #TODO 3. Read the result and continue.
    


    #TODO: Perform statistical hypothesis testing.
