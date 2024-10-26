import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway
from fpdf import FPDF
import seaborn as sns

class ReportMaker:

    def __init__(self, ppo_performance, default_agents_performance, random_agent_performance, a2c_performance, dqn_performance, test_periods):
        self.ppo_performance = ppo_performance
        self.default_agents_performance = default_agents_performance
        self.random_agent_performance = random_agent_performance
        self.a2c_performance = a2c_performance
        self.dqn_performance = dqn_performance
        self.test_periods = test_periods
        self.create_report()

    def create_report(self):
        # Calcular medias y varianzas
        ppo_mean = np.mean(self.ppo_performance)
        ppo_var = np.var(self.ppo_performance)

        default_mean = np.mean(self.default_agents_performance)
        default_var = np.var(self.default_agents_performance)

        random_mean = np.mean(self.random_agent_performance)
        random_var = np.var(self.random_agent_performance)

        # Contrastes de hipótesis
        t_stat_default, p_value_default = stats.ttest_ind(self.ppo_performance, self.default_agents_performance, equal_var=False)
        t_stat_random, p_value_random = stats.ttest_ind(self.ppo_performance, self.random_agent_performance, equal_var=False)

        # ANOVA de una vía
        anova_stat, anova_p_value = f_oneway(self.ppo_performance, self.default_agents_performance, self.random_agent_performance)

        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Agent Performance Report", 0, 1, "C")

        pdf.set_font("Arial", "", 12)
        pdf.cell(200, 10, "Mean and Variance of Profits", 0, 1)
        pdf.cell(200, 10, f"PPO - Mean: {ppo_mean:.2f}, Variance: {ppo_var:.2f}", 0, 1)
        pdf.cell(200, 10, f"Default - Mean: {default_mean:.2f}, Variance: {default_var:.2f}", 0, 1)
        pdf.cell(200, 10, f"Random - Mean: {random_mean:.2f}, Variance: {random_var:.2f}", 0, 1)
        
        pdf.cell(200, 10, f"A2C - Point Estimate: {self.a2c_performance}", 0, 1)
        pdf.cell(200, 10, f"DQN - Point Estimate: {self.dqn_performance}", 0, 1)
        
        pdf.cell(200, 10, "Hypothesis Testing Results", 0, 1)
        pdf.cell(200, 10, f"PPO vs Default: t={t_stat_default:.2f}, p={p_value_default:.4f}", 0, 1)
        pdf.cell(200, 10, f"PPO vs Random: t={t_stat_random:.2f}, p={p_value_random:.4f}", 0, 1)

        pdf.cell(200, 10, f"One way ANOVA: s={anova_stat:.2f}, p={anova_p_value:.4f}", 0, 1)

        # Save the PDF report
        pdf_output = "../results/agent_performance_report.pdf"
        pdf.output(pdf_output)


        # Plot Boxplots and lines for A2C and DQN
        plt.figure()
        plt.boxplot(self.ppo_performance)
        plt.axhline(self.a2c_performance, color='red', linestyle='--', label='A2C')
        plt.axhline(self.dqn_performance, color='green', linestyle='--', label='DQN')
        plt.title("PPO Performance Boxplot")
        plt.legend()
        plt.savefig("../results/ppo_boxplot.pdf")


        plt.figure()
        plt.boxplot(self.default_agents_performance)
        #plt.axhline(self.a2c_performance, color='red', linestyle='--', label='A2C')
        #plt.axhline(self.dqn_performance, color='green', linestyle='--', label='DQN')
        plt.title("Default Agent Performance Boxplot")
        plt.legend()
        plt.savefig("../results/default_boxplot.pdf")

        plt.figure()
        plt.boxplot(self.random_agent_performance)
        plt.axhline(self.a2c_performance, color='red', linestyle='--', label='A2C')
        plt.axhline(self.dqn_performance, color='green', linestyle='--', label='DQN')
        plt.title("Random Agent Performance Boxplot")
        plt.legend()
        plt.savefig("../results/random_boxplot.pdf")

        plt.figure(figsize=(8, 6))
        data = [self.ppo_performance, self.default_agents_performance, self.random_agent_performance]
        plt.boxplot(data, labels=['PPO', 'Default', 'Random'])
        plt.title("Combined Boxplot of Agent Performances")
        plt.ylabel("Profit")
        plt.savefig("../results/combined_boxplot.pdf")

        # Violin plot
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=data)
        plt.xticks(ticks=[0, 1, 2], labels=['PPO', 'Default', 'Random'])
        plt.title("Violin Plot of Agent Performances")
        plt.ylabel("Profit")
        plt.savefig("../results/violin_plot.pdf")

        # Bar plot de medias
        means = [np.mean(self.ppo_performance), np.mean(self.default_agents_performance), np.mean(self.random_agent_performance),
                 self.a2c_performance, self.dqn_performance]
        labels = ['PPO', 'Default', 'Random', 'A2C', 'DQN']
        
        plt.figure(figsize=(8, 6))
        plt.bar(labels, means, color=['blue', 'orange', 'green', 'red', 'purple'])
        plt.title("Mean Profit Comparison Across Agents")
        plt.ylabel("Mean Profit")
        plt.savefig("../results/mean_bar_plot.pdf")


        
