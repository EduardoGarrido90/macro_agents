import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class MetricsCallback(BaseCallback):
    def __init__(self, num_agents, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.num_agents = num_agents
        # Initialize lists for each agent
        self.losses = [[] for _ in range(num_agents)]
        self.value_losses = [[] for _ in range(num_agents)]
        self.approx_kls = [[] for _ in range(num_agents)]
        self.explained_variance = [[] for _ in range(num_agents)]
        self.timesteps = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        for agent_id in range(self.num_agents):
            # Get rewards, returns, and values for this agent
            agent_rewards = self.model.rollout_buffer.rewards[:, agent_id]
            agent_returns = self.model.rollout_buffer.returns[:, agent_id]
            agent_values = self.model.rollout_buffer.values[:, agent_id]
            agent_advantages = self.model.rollout_buffer.advantages[:, agent_id]

            # Calculate value loss (MSE between returns and values)
            agent_value_loss = np.mean((agent_returns - agent_values) ** 2)
            self.value_losses[agent_id].append(agent_value_loss)

            # Policy loss: You would need access to the policy probabilities to calculate this.
            # This is an example of a placeholder (PPO calculates this during the learning step)
            agent_loss = np.mean(agent_advantages)  # Placeholder, needs actual policy loss calc
            self.losses[agent_id].append(agent_loss)

            # Approx KL divergence (you will need old and new policy probs for this calculation)
            agent_approx_kl = np.mean(np.abs(agent_advantages))  # Placeholder, needs real KL
            self.approx_kls[agent_id].append(agent_approx_kl)

            # Explained variance
            agent_exp_var = 1 - np.var(agent_returns - agent_values) / np.var(agent_returns)
            self.explained_variance[agent_id].append(agent_exp_var)

        self.timesteps.append(self.num_timesteps)
