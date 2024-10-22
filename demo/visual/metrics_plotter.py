import matplotlib.pyplot as plt
import numpy as np

class MetricsPlotter:

    def __init__(self, metrics_callback, agents_number, accumulated_profits, default_accumulated_profits, random_accumulated_profits, steps, simulator_logs, plot_everything=False):
        self.metrics_callback = metrics_callback
        self.agents_number = agents_number
        self.accumulated_profits = accumulated_profits
        self.default_accumulated_profits = default_accumulated_profits
        self.random_accumulated_profits = random_accumulated_profits
        self.steps = steps
        self.simulator_logs = simulator_logs
        if plot_everything:
            self.plot_model_loss()
            self.plot_value_loss()
            self.plot_approx_kl()
            self.plot_explained_variance()
            self.plot_accumulated_profit()
            self.plot_simulator_logs()


    def plot_simulator_logs(self):
        total_agents = self.agents_number + len(self.default_accumulated_profits) + len(self.random_accumulated_profits) 
        price_log = self.simulator_logs["price"]
        supply_log = self.simulator_logs["supply"]
        demand_log = self.simulator_logs["demand"]
        progress_log = self.simulator_logs["progress"]
        timesteps = len(price_log)-1 #Each log is of a timestep.
        timesteps_x = np.linspace(0, timesteps, timesteps+1).astype(int)
       
        for i in range(total_agents): 
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps_x, price_log[:, i])
            plt.xlabel('Timesteps')
            plt.ylabel('Price')
            plt.title('Price over Time')
            plt.savefig('../results/price_timesteps_' + str(i) + '.pdf')
            plt.close()

        for i in range(total_agents): 
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps_x, supply_log[:, i])
            plt.xlabel('Timesteps')
            plt.ylabel('Supply')
            plt.title('Supply over Time')
            plt.savefig('../results/supply_timesteps_' + str(i) + '.pdf')
            plt.close()

        for i in range(total_agents): 
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps_x, demand_log[:, i])
            plt.xlabel('Timesteps')
            plt.ylabel('Demand')
            plt.title('Demand over Time')
            plt.savefig('../results/demand_timesteps_' + str(i) + '.pdf')
            plt.close()

        for i in range(total_agents): 
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps_x, progress_log[:, i])
            plt.xlabel('Timesteps')
            plt.ylabel('Progress')
            plt.title('Progress over Time')
            plt.savefig('../results/progress_timesteps_' + str(i) + '.pdf')
            plt.close()


    def plot_model_loss(self):
        if self.metrics_callback.losses[0]:
            plt.figure(figsize=(10, 6))
            for agent in range(self.agents_number):
                # Plot Model Loss
                plt.plot(self.metrics_callback.timesteps, self.metrics_callback.losses[agent], label='Total Loss Agent ' + str(agent))
            plt.xlabel('Timesteps')
            plt.ylabel('Loss')
            plt.legend(loc='upper left')
            plt.title('Model Losses over Time')
            plt.savefig('../results/model_loss.pdf')
            plt.close()

    def plot_value_loss(self):
        if self.metrics_callback.value_losses[0]:
            plt.figure(figsize=(10, 6))
            for agent in range(self.agents_number):
                # Plot Model Loss
                plt.plot(self.metrics_callback.timesteps, self.metrics_callback.losses[agent], label='Value Loss Agent ' + str(agent))
            plt.xlabel('Timesteps')
            plt.ylabel('Value Loss')
            plt.legend(loc='upper left')
            plt.title('Value Losses over Time')
            plt.savefig('../results/value_loss.pdf')
            plt.close()

    def plot_approx_kl(self):
        # Plot Approximate KL Divergence
        if self.metrics_callback.approx_kls[0]:
            plt.figure(figsize=(10, 6))
            for agent in range(self.agents_number):
                plt.plot(self.metrics_callback.timesteps, self.metrics_callback.approx_kls[agent], label='Approx KL Agent ' + str(agent))
            plt.xlabel('Timesteps')
            plt.ylabel('Approx KL Divergence')
            plt.legend(loc='upper left')
            plt.title('Approximate KL Divergence over Time')
            plt.savefig('../results/approx_kl.pdf')
            plt.close()

    def plot_explained_variance(self):
        # Plot Explained Variance
        if self.metrics_callback.explained_variance[0]:
            plt.figure(figsize=(10, 6))
            for agent in range(self.agents_number):
                plt.plot(self.metrics_callback.timesteps, self.metrics_callback.explained_variance[agent], label='Explained variance Agent ' + str(agent))
            plt.xlabel('Timesteps')
            plt.ylabel('Explained Variance')
            plt.legend(loc='upper left')
            plt.title('Explained Variance over Time')
            plt.savefig('../results/exp_var.pdf')
            plt.close()

    def plot_accumulated_profit(self):
        # Plot the accumulated profit over time
        plt.figure(figsize=(10, 6))
        for agent in range(self.agents_number): 
            plt.plot(self.steps, self.accumulated_profits[agent], marker='o', linestyle='-', label=f'Agent {agent+1} Accumulated Profit')
        for agent in range(self.agents_number, self.agents_number + len(self.default_accumulated_profits)): 
            plt.plot(self.steps, self.default_accumulated_profits[agent-self.agents_number], marker='o', linestyle='-', label=f'Default Agent {agent-self.agents_number} Accumulated Profit')
        for agent in range(self.agents_number + len(self.default_accumulated_profits), self.agents_number + len(self.default_accumulated_profits) + len(self.random_accumulated_profits)):
            plt.plot(self.steps, self.random_accumulated_profits[agent-self.agents_number-len(self.default_accumulated_profits)], marker='o', linestyle='-', label=f'Random Agent {agent-self.agents_number-len(self.default_accumulated_profits)} Accumulated Profit') 

        plt.title('Accumulated Profit of each Agent Over Test Steps')
        plt.xlabel('Step')
        plt.ylabel('Accumulated Profit')
        plt.grid(True)
        plt.legend(loc='upper left')

        # Save the plot to a PDF file
        plt.savefig('../results/profit_evolution.pdf')
        plt.close()

