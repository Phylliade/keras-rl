# -*- coding: utf-8 -*-
import warnings
from copy import deepcopy

import numpy as np
from keras.callbacks import History
import keras.backend as K

from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
from rl.hooks import PortraitHook, TensorboardHook
from rl.state import EpisodicState

DEBUG = True

STEPS_TERMINATION = 1
EPISODES_TERMINATION = 2


class Agent(object):
    """Abstract base class for all implemented agents.

    Each agent interacts with the environment (as defined by the `Env` class) by first observing the
    state of the environment. Based on this observation the agent changes the environment by performing
    an action.

    Do not use this abstract base class directly but instead use one of the concrete agents implemented.
    Each agent realizes a reinforcement learning algorithm. Since all agents conform to the same
    interface, you can use them interchangeably.

    To implement your own agent, you have to implement the following methods:

    - `forward`
    - `backward`
    - `compile`
    - `load_weights`
    - `save_weights`
    - `layers`

    # Arguments
        processor (`Processor` instance): See [Processor](#processor) for details.
    """

    def __init__(self, processor=None):
        # Use a default identity processor if not provided
        if processor is None:
            self.processor = Processor()
        self.training = False
        self.step = 0
        # Use the same session as Keras
        self.session = K.get_session()
        # self.session = tf.Session()

        # State machine
        self.state = EpisodicState()
        # FIXME: won't work with offline training

        # Hooks
        self.hooks = [PortraitHook(self), TensorboardHook(self)]

    def get_config(self):
        """Configuration of the agent for serialization.
        """
        return {}

    def _run(self,
             env,
             nb_steps=None,
             nb_episodes=None,
             training=True,
             action_repetition=1,
             callbacks=None,
             verbose=1,
             visualize=False,
             nb_max_start_steps=0,
             start_step_policy=None,
             log_interval=10000,
             nb_max_episode_steps=None,
             reward_scaling=1.):
        """Trains the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            nb_episodes (integer): Number of episodes to perform
            training (boolean): Whether to train or test the agent
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.
            reward_scaling (float): The amount with which the reward will be scaled

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.'
            )
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(
                action_repetition))

        # Process the different cases when either nb_steps or nb_episodes are specified
        if (nb_steps is None and nb_episodes is None):
            raise (ValueError(
                "Please specify one (and only one) of nb_steps and nb_episodes"
            ))
        elif (nb_steps is not None and nb_episodes is None):
            termination_criterion = STEPS_TERMINATION
        elif (nb_steps is None and nb_episodes is not None):
            termination_criterion = EPISODES_TERMINATION
        elif (nb_steps is not None and nb_episodes is not None):
            raise (ValueError(
                "Please specify one (and only one) of nb_steps and nb_episodes"
            ))

        self.training = training

        if DEBUG:
            print("Training mode:".format(self.training))

        # Initilize callbacks
        callbacks = [] if not callbacks else callbacks[:]

        if self.training:
            if verbose == 1:
                callbacks += [TrainIntervalLogger(interval=log_interval)]
            elif verbose > 1:
                callbacks += [TrainEpisodeLogger()]
        else:
            if verbose >= 1:
                callbacks += [TestLogger()]

        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]

        callbacks = CallbackList(callbacks)

        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)

        callbacks._set_env(env)
        if termination_criterion == STEPS_TERMINATION:
            params = {
                'nb_steps': nb_steps,
            }
        elif termination_criterion == EPISODES_TERMINATION:
            params = {
                'nb_episodes': nb_episodes,
                'nb_steps': 1,
            }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        if self.training:
            self._on_train_begin()
        else:
            self._on_test_begin()

        callbacks.on_train_begin()

        # Setup
        # TODO: Create an init_episode function
        self.state.reset_episode()
        self.state.init_step()
        did_abort = False

        if termination_criterion == STEPS_TERMINATION:

            def termination():
                return (self.state.step > nb_steps)
        elif termination_criterion == EPISODES_TERMINATION:

            def termination():
                return (self.state.episode > nb_episodes)

        try:
            # Run steps (and episodes) until the termination criterion is met
            while not (termination()):
                # If we are at the beginning of a new episode, execute a startup sequence
                if self.state.episode_beginning:
                    callbacks.on_episode_begin(self.state.episode)

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    self.state.observation_0 = deepcopy(env.reset())
                    self.state.observation_0 = self.processor.process_observation(
                        self.state.observation_0)

                    # Perform random steps at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    if nb_max_start_steps != 0:
                        self.state.observation_0 = self._perform_random_steps(
                            nb_max_start_steps, start_step_policy, env,
                            self.state.observation_0, callbacks)

                else:
                    # We are in the middle of an episode
                    self.state.reset_step()

                # Increment the current step in both cases
                self.step += 1

                # Run a single step.
                callbacks.on_step_begin(self.state.episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).

                # state_0 -- (foward) --> action
                action = self.forward(self.state.observation_0)

                # Process the action
                action = self.processor.process_action(action)

                # action -- (step) --> (reward, state_1, terminal)
                reward = 0.
                accumulated_info = {}

                # Apply the action and observe the result
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    self.state.observation_1, r, self.state.done, info = env.step(action)
                    self.state.observation_1 = deepcopy(self.state.observation_1)

                    self.state.observation_1, r, self.state.done, info = self.processor.process_step(
                        self.state.observation_1, r, self.state.done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(action)

                    reward += r

                    if self.state.done:
                        break

                # Scale the reward
                reward = reward * reward_scaling

                # TODO: Move this when bacwkard won't use the terminal information
                # Stop episode if reached the step limit
                if nb_max_episode_steps and self.state.episode_step >= nb_max_episode_steps:
                    # Force a terminal state.
                    self.state.done = True

                # Use the step information to train the algorithm
                metrics, self.state.step_summaries = self.backward(
                    self.state.observation_0,
                    action,
                    reward,
                    self.state.observation_1,
                    terminal=self.state.done,
                    epoch=self.state.step)

                # Collect statistics
                self.state.episode_reward += reward
                step_logs = {
                    'action': action,
                    'observation': self.state.observation_1,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': self.state.episode,
                    'info': accumulated_info,
                }

                callbacks.on_step_end(self.state.episode_step, step_logs)

                self.state.reset_step()

                # Call the hooks
                for hook in self.hooks:
                    hook()

                # Reset the episode
                if self.state.done:
                    # Callback
                    episode_logs = {
                        'episode_reward': np.float_(self.state.episode_reward),
                        'nb_episode_steps': np.float_(self.state.episode_step),
                        'nb_steps': np.float_(self.state.step),
                    }
                    callbacks.on_episode_end(self.state.episode, logs=episode_logs)

                    # Reset the episode variables
                    self.state.reset_episode()

        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True

        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history

    def _perform_random_steps(self, nb_max_start_steps, start_step_policy, env,
                              observation, callbacks):
        nb_random_start_steps = np.random.randint(nb_max_start_steps)
        for _ in range(nb_random_start_steps):
            if start_step_policy is None:
                action = env.action_space.sample()
            else:
                action = start_step_policy(observation)
            action = self.processor.process_action(action)
            callbacks.on_action_begin(action)
            observation, reward, done, info = env.step(action)
            observation = deepcopy(observation)
            observation, reward, done, info = self.processor.process_step(
                observation, reward, done, info)
            callbacks.on_action_end(action)

            if done:
                warnings.warn(
                    'Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.
                    format(nb_random_start_steps))
                observation = deepcopy(env.reset())
                observation = self.processor.process_observation(observation)
                break
        return (observation)

    def fit(self, **kwargs):
        """
        Train the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            or nb_episodes
        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        return(self._run(training=True, **kwargs))

    def fit_offline(self):
        """Train the networks in offline mode"""
        raise (NotImplementedError)

    def test(self, **kwargs):
        """
        Test the agent on the given environment.
        In training mode, noise is removed.
        """
        return(self._run(training=False, **kwargs))

    def reset_states(self):
        """Resets all internally kept states after an episode is completed.
        """
        pass

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            The next action to be executed in the environment.
        """
        raise NotImplementedError()

    def backward(self, observation_0, action, reward, observation_1, terminal):
        """
        Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.

        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.
        """
        raise NotImplementedError()

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.

        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        raise NotImplementedError()

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.

        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.

        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        raise NotImplementedError()

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).

        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.
        """
        raise NotImplementedError()

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        pass

    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        pass

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        pass

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        pass


class Processor(object):
    """Abstract base class for implementing processors.

    A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
    be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlaying
    implementation of the agent or environment.

    Do not use this abstract base class directly but instead use one of the concrete implementations
    or write your own.
    """

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.

        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.

        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.
        """
        return observation

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.
        """
        return reward

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.
        """
        return info

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.
        """
        return action

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.
        """
        return batch

    @property
    def metrics(self):
        """The metrics of the processor, which will be reported during training.

        # Returns
            List of `lambda y_true, y_pred: metric` functions.
        """
        return []

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []


class MultiInputProcessor(Processor):
    """Converts observations from an environment with multiple observations for use in a neural network
    policy.

    In some cases, you have environments that return multiple different observations per timestep
    (in a robotics context, for example, a camera may be used to view the scene and a joint encoder may
    be used to report the angles for each joint). Usually, this can be handled by a policy that has
    multiple inputs, one for each modality. However, observations are returned by the environment
    in the form of a tuple `[(modality1_t, modality2_t, ..., modalityn_t) for t in T]` but the neural network
    expects them in per-modality batches like so: `[[modality1_1, ..., modality1_T], ..., [[modalityn_1, ..., modalityn_T]]`.
    This processor converts observations appropriate for this use case.

    # Arguments
        nb_inputs (integer): The number of inputs, that is different modalities, to be used.
            Your neural network that you use for the policy must have a corresponding number of
            inputs.
    """

    def __init__(self, nb_inputs):
        self.nb_inputs = nb_inputs

    def process_state_batch(self, state_batch):
        input_batches = [[] for x in range(self.nb_inputs)]
        for state in state_batch:
            processed_state = [[] for x in range(self.nb_inputs)]
            for observation in state:
                assert len(observation) == self.nb_inputs
                for o, s in zip(observation, processed_state):
                    s.append(o)
            for idx, s in enumerate(processed_state):
                input_batches[idx].append(s)
        return [np.array(x) for x in input_batches]


# Note: the API of the `Env` and `Space` classes are taken from the OpenAI Gym implementation.
# https://github.com/openai/gym/blob/master/gym/core.py


class Env(object):
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym implementation, this class only defines the abstract methods without any actual
    implementation.
    """
    reward_range = (-np.inf, np.inf)
    action_space = None
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        # Arguments
            action (object): An action provided by the environment.

        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        raise NotImplementedError()

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        raise NotImplementedError()

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)

        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        raise NotImplementedError()

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)


class Space(object):
    """Abstract model for a space that is used for the state and action spaces. This class has the
    exact same API that OpenAI Gym uses so that integrating with it is trivial.
    """

    def sample(self, seed=None):
        """Uniformly randomly sample a random element of this space.
        """
        raise NotImplementedError()

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space
        """
        raise NotImplementedError()
