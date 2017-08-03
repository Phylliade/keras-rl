class State:
    """Class storing the internal state of the RL Agent"""
    pass


class EpisodicState(State):
    def __init__(self):
        # Document here all internal variables acting in the state
        self.done = None
        self.episode = 0
        self.step = 0
        self.observation_0 = None
        self.observation_1 = None
        self.episode_reward = None
        self.episode_step = None
        self.step_summaries = None

    def reset_episode(self):
        self.episode += 1
        self.episode_reward = 0
        self.episode_beginning = True
        # Set episode_step before the initial value (1)
        self.episode_step = 0
        self.done = False
        self.observation_0 = None
        self.observation_1 = None

    def init_step(self):
        self.step += 1
        self.episode_step += 1
        self.reward = 0

    def reset_step(self):
        # Update the observation
        if self.observation_1 is not None:
            self.observation_0 = self.observation_1
        else:
            raise(ValueError("observation_1 is None, whereas the episode is not terminated"))
        # Increment the step and the episode step
        self.init_step()


class OfflineState(State):
    pass
