from rl.utils.plot import portrait_critic, portrait_actor
import tensorflow as tf


class Hook:
    def __init__(self, agent):
        """
        The abstract Hook class
        :param agent: the RL agent
        """
        self.agent = agent

    def __call__(self):
        """
        Generic hook.
        Better if it doesn't return anything and instead exports the data itself (e.g. pickle, image)
        """
        raise(NotImplementedError)

    @property
    def count(self):
        return(self.agent.state.episode)


class PortraitHook(Hook):
    # TODO: Move this hook to a non-blocking thread
    def __call__(self):
        if self.agent.state.done:
            if self.agent.training:
                file_name = "portrait"
            else:
                file_name = "portrait_test"
            file_name = "{}_".format(self.count) + file_name + ".png"
            portrait_actor(self.agent.actor, save_figure=True, figure_file=("figures/actor/" + file_name))
            portrait_critic(self.agent.critic, save_figure=True, figure_file=("figures/critic/" + file_name))


class TrajectoryHook(Hook):
    """Records the trajectory of the agent"""
    def __call__(self):
        pass


class TensorboardHook(Hook):
    def __init__(self, agent):
        super().__init__(agent)
        self.summary_writer = tf.summary.FileWriter('./logs')

    def __call__(self):
        # Step summaries
        for summary in self.agent.state.step_summaries:
            # FIXME: Use only one summary
            self.summary_writer.add_summary(summary, self.agent.state.step)

        # Episode summaries
        if self.agent.state.done:
            episode_summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=self.agent.state.episode_reward), ])
            self.summary_writer.add_summary(episode_summary, self.agent.state.episode)
