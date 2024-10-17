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

    env = SubprocVecEnv([make_env(max_actions) for _ in range(agents_number)])

    # Set up the PPO model for the environment
    model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=1, seed=seed)

    metrics_callback = MetricsCallback(agents_number)

    # Train the agent
    model.learn(total_timesteps=total_training_timesteps, callback=metrics_callback, progress_bar=False)

    # Test the agent
    obs = env.reset()
    ppo_performance = 0.0
    ppo_performance = [0.0] * agents_number  # Store performance for each agent
    default_agents_performance = [0.0] * max_actions #Store performance of fixed value production agents
    random_agent_performance = 0.0 #Store performance of random production agent
    accumulated_profits = [[0.0] for _ in range(agents_number)]  # One list per agent
    default_accumulated_profits = [[0.0] for _ in range(max_actions)]  # One list per default agent
    random_accumulated_profits = [0.0] #One list for the random agent
    steps = [0]
    for step in range(test_periods):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated = env.step(action) #debug this an check how to create a fix agent, got to be easy do not worry, for tomorrow. 
        env.render()
        # Update performance for each agent
        for i in range(agents_number):
            ppo_performance[i] += reward[i]
            accumulated_profits[i].append(ppo_performance[i])
        steps.append(step+1)
        print(f"Step: {step+1}, Action: {action}, Reward: {reward}")

        #Commented because from the moment it never ends and it is a vector of environments.
        if any(terminated) or any(truncated):
                obs = env.reset()
                print(f"Reset environments where agents are done")
    
    for i in range(agents_number):
        print(f"The accumulated profit of the PPO agent {i+1} is: {ppo_performance[i]}")

    #We plot all the training progress of the agents and the performance in a new scenario. 
    MetricsPlotter(metrics_callback, agents_number, accumulated_profits, steps, plot_everything=True)
