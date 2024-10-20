import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from visual.metrics_plotter import MetricsPlotter
from visual.drl_metrics_callback import MetricsCallback
from simulator.market_environment import MarketEnv
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv
from config.argument_parser import ArgumentParser
matplotlib.use('Agg')  # Use the 'Agg' backend for rendering without a display


def make_env(max_actions):
    def _init():
        env = MarketEnv(max_actions)  # Crear la instancia de MarketEnv
        return env
    return _init  # Retorna la función _init, no el entorno directamente

if __name__ == '__main__':
    
    # Encapsulate the argument parsing in the ArgumentParser class
    argument_parser = ArgumentParser()
    args = argument_parser.parse_args()
    agents_number = args.agents_number
    test_periods = args.test_periods
    total_training_timesteps = args.total_training_timesteps
    seed = args.seed
    max_actions = args.max_actions
    number_random_agents = 5

    env = SubprocVecEnv([make_env(max_actions) for _ in range(agents_number+max_actions+number_random_agents)]) #agents+default agents+random agent

    # Set up the PPO model for the environment
    #model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=1, seed=seed)
    model = PPO("MlpPolicy", env, learning_rate=5e-5, verbose=1, seed=seed)

    metrics_callback = MetricsCallback(agents_number)

    # Train the agent
    model.learn(total_timesteps=total_training_timesteps, callback=metrics_callback, progress_bar=False)

    # Test the agent
    obs = env.reset()
    ppo_performance = 0.0
    ppo_performance = [0.0] * agents_number  # Store performance for each agent
    default_agents_performance = [0.0] * max_actions #Store performance of fixed value production agents
    random_agent_performance = [0.0] * number_random_agents #Store performance of random production agents
    accumulated_profits = [[0.0] for _ in range(agents_number)]  # One list per agent
    default_accumulated_profits = [[0.0] for _ in range(max_actions)]  # One list per default agent
    random_accumulated_profits = [[0.0] for _ in range(number_random_agents)] #One list per random agent
    steps = [0]
    for step in range(test_periods):
        action, _states = model.predict(obs)
        agents_actions = action[0:agents_number]
        default_action_agents = np.linspace(0, max_actions-1, num=max_actions).astype(int)
        random_action_agent = np.random.randint(0, 3, number_random_agents)
        action = np.append(np.append(agents_actions, default_action_agents), random_action_agent)
        #Building the final action.
        obs, reward, terminated, truncated = env.step(action) #debug this an check how to create a fix agent, got to be easy do not worry, for tomorrow. 
        env.render()
        # Update performance for each agent
        # Normal agents
        for i in range(agents_number):
            ppo_performance[i] += reward[i]
            accumulated_profits[i].append(ppo_performance[i])
        
        # Default agents
        for i in range(agents_number, agents_number + max_actions):
            default_agents_performance[i-agents_number] += reward[i]
            default_accumulated_profits[i-agents_number].append(default_agents_performance[i-agents_number])

        # Random agents
        for i in range(agents_number + max_actions, agents_number + max_actions + number_random_agents):
            random_agent_performance[i-agents_number-max_actions] += reward[i]
            random_accumulated_profits[i-agents_number-max_actions].append(random_agent_performance[i-agents_number-max_actions])

        steps.append(step+1)
        print(f"Step: {step+1}, Action: {action}, Reward: {reward}")

        #Commented because from the moment it never ends and it is a vector of environments.
        if any(terminated) or any(truncated):
                obs = env.reset()
                print(f"Reset environments where agents are done")
    
    for i in range(agents_number):
        print(f"The accumulated profit of the PPO agent {i+1} is: {ppo_performance[i]}")

    for i in range(len(default_agents_performance)):
        print(f"The accumulated profit of the default agent {i+1} is: {default_agents_performance[i]}")

    for i in range(len(random_agent_performance)):
        print(f"The accumulated profit of the random agent is: {random_agent_performance[i]}")

    #We plot all the training progress of the agents and the performance in a new scenario. 
    MetricsPlotter(metrics_callback, agents_number, accumulated_profits, default_accumulated_profits, random_accumulated_profits, steps, plot_everything=True)
