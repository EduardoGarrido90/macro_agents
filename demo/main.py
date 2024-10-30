import numpy as np
from stable_baselines3 import PPO, A2C, DQN #Models.
from stable_baselines3.common.env_checker import check_env
from visual.metrics_plotter import MetricsPlotter
from visual.report_maker import ReportMaker
from visual.drl_metrics_callback import MetricsCallback
from simulator.market_environment import MarketEnv
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv
from config.argument_parser import ArgumentParser
matplotlib.use('Agg')  # Use the 'Agg' backend for rendering without a display


def make_env(max_actions, min_prod, num_competitors, initial_price, max_fixed_costs, min_fixed_costs, cost_coef_0, cost_coef_1, cost_coef_2, cost_coef_3):
    def _init():
        env = MarketEnv(max_actions, min_prod, num_competitors, initial_price, max_fixed_costs, min_fixed_costs, cost_coef_0, cost_coef_1, cost_coef_2, cost_coef_3)  # Crear la instancia de MarketEnv
        return env
    return _init  # Retorna la función _init, no el entorno directamente

if __name__ == '__main__':
    
    # Encapsulate the argument parsing in the ArgumentParser class
    argument_parser = ArgumentParser()
    args = argument_parser.parse_args()
    agents_number = args.agents_number #ok
    test_periods = args.test_periods #ok 
    total_training_timesteps = args.total_training_timesteps #ok
    seed = args.seed #ok
    max_actions = args.max_actions #ok
    min_prod = args.min_prod #ok
    num_competitors = args.num_competitors #ok
    initial_price = args.initial_price #ok
    max_fixed_costs = args.max_fixed_costs #ok
    min_fixed_costs = args.min_fixed_costs #ok
    cost_coef_0 = args.cost_coef_0 #ok
    cost_coef_1 = args.cost_coef_1 #ok
    cost_coef_2 = args.cost_coef_2 #ok
    cost_coef_3 = args.cost_coef_3 #ok
    elasticity = args.elasticity
    base_demand = args.base_demand
    prod_noise = args.prod_noise
    storage_factor = args.storage_factor
    brand_effect = args.brand_effect
    max_subsidy = args.max_subsidy
    number_random_agents = args.number_random_agents #ok 
    #TODO: Parametrize the code with the rest of hyperparameters.
    learning_rate_models = args.learning_rate #ok 1e-4. TODO: Should have an optimization mode.


    alternative_drl_agents = 2 #TODO: Make it parametrizable, type and number.
    total_agents = agents_number + max_actions + number_random_agents 
    simulator_logs = {"price" : np.zeros([test_periods, total_agents]), "supply" : np.zeros([test_periods, total_agents]), \
            "demand" : np.zeros([test_periods, total_agents]), "progress" : np.zeros([test_periods, total_agents])}

    env = SubprocVecEnv([make_env(max_actions, min_prod, num_competitors, initial_price, max_fixed_costs, min_fixed_costs, cost_coef_0, cost_coef_1, cost_coef_2, cost_coef_3) for _ in range(total_agents)]) 
    env_a2c = MarketEnv(max_actions, min_prod, num_competitors, initial_price, max_fixed_costs, min_fixed_costs, cost_coef_0, cost_coef_1, cost_coef_2, cost_coef_3)
    env_dqn = MarketEnv(max_actions, min_prod, num_competitors, initial_price, max_fixed_costs, min_fixed_costs, cost_coef_0, cost_coef_1, cost_coef_2, cost_coef_3)

    # Set up the models for the environment
    model = PPO("MlpPolicy", env, learning_rate=learning_rate_models, verbose=1, seed=seed)
    model_a2c = A2C("MlpPolicy", env_a2c, learning_rate=learning_rate_models, verbose=1, seed=seed)
    model_dqn = DQN("MlpPolicy", env_dqn, learning_rate=learning_rate_models, verbose=1, seed=seed)

    metrics_callback = MetricsCallback(agents_number)

    # Train the agent
    model.learn(total_timesteps=total_training_timesteps, callback=metrics_callback, progress_bar=False)
    model_a2c.learn(total_timesteps=total_training_timesteps, progress_bar=False)
    model_dqn.learn(total_timesteps=total_training_timesteps, progress_bar=False)

    # Test the agent
    obs = env.reset()
    obs_a2c = env_a2c.reset()[0]
    obs_dqn = env_dqn.reset()[0]
    ppo_performance = 0.0
    ppo_performance = [0.0] * agents_number  # Store performance for each agent
    a2c_performance = 0.0   # Store performance for each agent
    dqn_performance = 0.0   # Store performance for each agent
    default_agents_performance = [0.0] * max_actions #Store performance of fixed value production agents
    random_agent_performance = [0.0] * number_random_agents #Store performance of random production agents
    accumulated_profits = [[0.0] for _ in range(agents_number)]  # One list per agent
    default_accumulated_profits = [[0.0] for _ in range(max_actions)]  # One list per default agent
    random_accumulated_profits = [[0.0] for _ in range(number_random_agents)] #One list per random agent
    a2c_accumulated_profits = [0.0] #One list per random agent
    dqn_accumulated_profits = [0.0] #One list per random agent
    steps = [0]
    for step in range(test_periods):
        action, _states = model.predict(obs)
        action_a2c, _states_a2c = model_a2c.predict(obs_a2c)
        action_dqn, _states_dqn = model_dqn.predict(obs_dqn)
        agents_actions = action[0:agents_number]
        default_action_agents = np.linspace(0, max_actions-1, num=max_actions).astype(int)
        random_action_agent = np.random.randint(0, max_actions-1, number_random_agents)
        action = np.append(np.append(agents_actions, default_action_agents), random_action_agent)
        #Building the final action.
        obs, reward, terminated, info = env.step(action) #debug this an check how to create a fix agent, got to be easy do not worry, for tomorrow. 
        obs_a2c, reward_a2c, terminated_a2c, truncated_a2c, info_a2c = env_a2c.step(action_a2c) #debug this an check how to create a fix agent, got to be easy do not worry, for tomorrow. 
        obs_dqn, reward_dqn, terminated_dqn, truncated_a2c, info_dqn = env_dqn.step(action_dqn) #debug this an check how to create a fix agent, got to be easy do not worry, for tomorrow. 
        for index_agent, agent_information in enumerate(info):
            simulator_logs["price"][step][index_agent] = agent_information["price"]
        for index_agent, agent_information in enumerate(info):
            simulator_logs["supply"][step][index_agent] = agent_information["supply"]
        for index_agent, agent_information in enumerate(info):
            simulator_logs["demand"][step][index_agent] = agent_information["demand"]
        for index_agent, agent_information in enumerate(info):
            simulator_logs["progress"][step][index_agent] = agent_information["progress"]
        env.render()
        env_a2c.render()
        env_dqn.render()
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

        a2c_performance += reward_a2c
        dqn_performance += reward_dqn
        a2c_accumulated_profits.append(a2c_performance)
        dqn_accumulated_profits.append(dqn_performance)

        steps.append(step+1)
        print(f"Step: {step+1}, Action: {action}, Reward: {reward}")

        #Commented because from the moment it never ends and it is a vector of environments.
        if any(terminated):# or any(truncated):
                obs = env.reset()
                obs_a2c = env_a2c.reset()
                obs_dqn = env_dqn.reset()
                print(f"Reset environments where agents are done")
    
    for i in range(agents_number):
        print(f"The accumulated profit of the PPO agent {i+1} is: {ppo_performance[i]}")

    for i in range(len(default_agents_performance)):
        print(f"The accumulated profit of the default agent {i+1} is: {default_agents_performance[i]}")

    for i in range(len(random_agent_performance)):
        print(f"The accumulated profit of the random agent is: {random_agent_performance[i]}")
    
    print(f"The accumulated profit of the a2c agent is: {a2c_performance}")
    print(f"The accumulated profit of the dqn agent is: {dqn_performance}")

    
    #We plot all the training progress of the agents and the performance in a new scenario. 
    MetricsPlotter(metrics_callback, agents_number, accumulated_profits, default_accumulated_profits, random_accumulated_profits, steps, simulator_logs, a2c_accumulated_profits, dqn_accumulated_profits, plot_everything=True)

    #We generate a report of the results in profit terms of the last day.
    ReportMaker(ppo_performance, default_agents_performance, random_agent_performance, a2c_performance, dqn_performance, test_periods)
