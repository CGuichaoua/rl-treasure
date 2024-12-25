
class RLRunner:
    """Class to handle running training and testing an agent."""

    def __init__(self, agent, env, *,
                 total_epochs=10000, eval_interval=1000, eval_episodes=5, verbose=True):
        self.agent = agent
        self.env = env
        self.total_epochs = total_epochs
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.verbose = verbose

        self.reward_history = []

    def train_agent(self):
        """Train the agent with regular evaluation loops."""
        total_epochs = self.total_epochs
        for epoch_no in range(total_epochs):
            self.env.reset()
            self.agent.train(total_timesteps=self.eval_interval)
            if self.verbose:
                print(f"Epoch {epoch_no +
                      1}/{total_epochs} - Training complete")
            mean_reward, _ = self.agent.evaluate(n_episodes=self.eval_episodes)
            self.reward_history.append(mean_reward)
            if self.verbose:
                print(f"Epoch {epoch_no +
                               1}/{total_epochs} - Mean reward: {mean_reward}")
