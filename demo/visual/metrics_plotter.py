import matplotlib.pyplot as plt

class MetricsPlotter:

    def __init__(self, metrics_callback, agents_number, accumulated_profits, default_accumulated_profits, random_accumulated_profits, steps, plot_everything=False):
        self.metrics_callback = metrics_callback
        self.agents_number = agents_number
        self.accumulated_profits = accumulated_profits
        self.default_accumulated_profits = default_accumulated_profits
        self.random_accumulated_profits = random_accumulated_profits
        self.steps = steps
        if plot_everything:
            self.plot_model_loss()
            self.plot_value_loss()
            self.plot_approx_kl()
            self.plot_explained_variance()
            self.plot_accumulated_profit()


    def plot_model_loss(self):
        if self.metrics_callback.losses[0]:
            plt.figure(figsize=(10, 6))
            for agent in range(self.agents_number):
                # Plot Model Loss
                plt.plot(self.metrics_callback.timesteps, self.metrics_callback.losses[agent], label='Total Loss Agent ' + str(agent))
            plt.xlabel('Timesteps')
            plt.ylabel('Loss')
            plt.legend()
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
            plt.legend()
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
            plt.legend()
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
            plt.legend()
            plt.title('Explained Variance over Time')
            plt.savefig('../results/exp_var.pdf')
            plt.close()

    def plot_accumulated_profit(self):
        # Plot the accumulated profit over time
        plt.figure(figsize=(10, 6))
        for agent in range(self.agents_number): 
            plt.plot(self.steps, self.accumulated_profits[agent], marker='o', linestyle='-', label=f'Agent {agent+1} Accumulated Profit')
        for agent in range(self.agents_number-1, self.agents_number + len(default_accumulated_profits)): 
            plt.plot(self.steps, self.default_accumulated_profits[agent-self.agents_number], marker='o', linestyle='-', label=f'Default Agent {agent-self.agents_number} Accumulated Profit')
        plt.plot(self.steps, self.random_accumulated_profits, marker='o', linestyle='-', label=f'Random Agent Accumulated Profit') 

        plt.title('Accumulated Profit of each Agent Over Test Steps')
        plt.xlabel('Step')
        plt.ylabel('Accumulated Profit')
        plt.grid(True)
        plt.legend()

        # Save the plot to a PDF file
        plt.savefig('../results/profit_evolution.pdf')
        plt.close()

