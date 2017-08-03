from __future__ import division
import os

import numpy as np
import tensorflow as tf
import keras.backend as K
from collections import namedtuple

from rl.core import Agent
from rl.util import huber_loss, clone_model, get_soft_target_model_ops
from rl.utils.numerics import gradient_inverter

Batch = namedtuple("Batch", ("state_0", "action", "reward", "state_1",
                             "terminal_1"))

# Whether to use Keras inference engine
USE_KERAS_INFERENCE = False


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class DDPGAgent(Agent):
    """
    Deep Deterministic Policy Gradient Agent.

    :param nb_actions:
    :param actions_low:
    :param actions_high:
    :param actor:
    :param critic:
    :param env:
    :param memory:
    :param gamma:
    :param batch_size:
    :param nb_steps_warmup_critic:
    :param nb_steps_warmup_actor:
    :param train_interval:
    :param delta_range:
    :param delta_clip:
    :param random_process:
    :param custom_model_objects:
    :param target_critic_update:
    :param target_actor_update:
    """

    def __init__(self,
                 nb_actions,
                 actions_low,
                 actions_high,
                 actor,
                 critic,
                 env,
                 memory,
                 gamma=.99,
                 batch_size=32,
                 nb_steps_warmup_critic=1000,
                 nb_steps_warmup_actor=1000,
                 train_interval=1,
                 memory_interval=1,
                 delta_clip=np.inf,
                 random_process=None,
                 custom_model_objects=None,
                 target_critic_update=.001,
                 target_actor_update=1,
                 invert_gradients=False,
                 gradient_inverter_min=-1.,
                 gradient_inverter_max=1.,
                 **kwargs):

        if custom_model_objects is None:
            custom_model_objects = {}
        if hasattr(actor.output, '__len__') and len(actor.output) > 1:
            raise ValueError(
                'Actor "{}" has more than one output. DDPG expects an actor that has a single output.'.
                format(actor))
        if hasattr(critic.output, '__len__') and len(critic.output) > 1:
            raise ValueError(
                'Critic "{}" has more than one output. DDPG expects a critic that has a single output.'.
                format(critic))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError(
                'Critic "{}" does not have enough inputs. The critic must have at exactly two inputs, one for the action and one for the observation.'.
                format(critic))

        super(DDPGAgent, self).__init__(**kwargs)

        # Get placeholders
        self.variables = env.variables

        # Parameters.
        self.nb_actions = nb_actions
        self.actions_low = actions_low
        self.actions_high = actions_high
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_critic_update = process_parameterization_variable(
            target_critic_update)
        self.target_actor_update = process_parameterization_variable(
            target_actor_update)
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects
        self.invert_gradients = invert_gradients
        if invert_gradients:
            self.gradient_inverter_max = gradient_inverter_max
            self.gradient_inverter_min = gradient_inverter_min

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()

        # Tensorboard
        # Each network has summaries attached
        # Since they are trained in using different functions
        self.actor_summaries = []
        self.critic_summaries = []

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or self.critic.uses_learning_phase

    def load_memory(self, memory):
        """Loads the given memory as the replay buffer"""
        del (self.memory)
        self.memory = memory

    def compile(self):
        # def clipped_error(y_true, y_pred):
        # return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        # Compile target networks. We only use them in feed-forward mode, hence we can pass any
        # optimizer and loss since we never use it anyway.
        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(self.critic,
                                         self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        # We also compile the actor. We never optimize the actor using Keras but instead compute
        # the policy gradient ourselves. However, we need the actor in feed-forward mode, hence
        # we also compile it with any optimizer
        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic for the same reason
        self.critic.compile(optimizer='sgd', loss='mse')

        # Compile the critic optimizer
        critic_optimizer = tf.train.AdamOptimizer()
        self.critic_target = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        # Clip the critic gradient using the huber loss
        critic_loss = K.mean(
            huber_loss(
                self.critic([self.variables.state, self.variables.action]), self.critic_target,
                self.delta_clip))
        critic_gradient_vars = critic_optimizer.compute_gradients(
            critic_loss, var_list=self.critic.trainable_weights)

        # Compute the norm as a metric
        critic_gradients_norms = [
            tf.norm(grad_var[0]) for grad_var in critic_gradient_vars
        ]
        critic_gradient_norm = tf.reduce_sum(critic_gradients_norms)

        self.critic_train_fn = critic_optimizer.apply_gradients(
            critic_gradient_vars)

        # Target critic optimizer
        if self.target_critic_update < 1.:
            # Include soft target model updates.
            self.target_critic_train_fn = get_soft_target_model_ops(
                self.target_critic.weights, self.critic.weights,
                self.target_critic_update)

        # Target actor optimizer
        if self.target_actor_update < 1.:
            # Include soft target model updates.
            self.target_actor_train_fn = get_soft_target_model_ops(
                self.target_actor.weights, self.actor.weights,
                self.target_actor_update)

        # Actor optimizer
        actor_optimizer = tf.train.AdamOptimizer()
        # Be careful to negate the gradient
        # Since the optimizer wants to minimize the value
        actor_loss = -tf.reduce_mean(
            self.critic([self.variables.state, self.actor(self.variables.state)]))

        actor_gradient_vars = actor_optimizer.compute_gradients(
            actor_loss, var_list=self.actor.trainable_weights)
        # Gradient inverting
        # as described in https://arxiv.org/abs/1511.04143
        if self.invert_gradients:
            actor_gradient_vars = [(gradient_inverter(
                x[0], self.gradient_inverter_min, self.gradient_inverter_max),
                                    x[1]) for x in actor_gradient_vars]

        # Compute the norm as a metric
        actor_gradients_norms = [
            tf.norm(grad_var[0]) for grad_var in actor_gradient_vars
        ]
        actor_gradient_norm = tf.reduce_sum(actor_gradients_norms)

        # The actual train function
        self.actor_train_fn = actor_optimizer.apply_gradients(
            actor_gradient_vars)

        # Collect metrics
        self.critic_summaries.append(
            tf.summary.scalar("critic/loss", critic_loss))
        self.critic_summaries.append(
            tf.summary.scalar("critic/gradient", critic_gradient_norm))
        for var, norm in zip(self.critic.trainable_weights,
                             critic_gradients_norms):
            self.critic_summaries.append(
                tf.summary.scalar("critic/{}".format(var.name), norm))

        self.actor_summaries.append(
            tf.summary.scalar("actor/loss", -actor_loss))
        self.actor_summaries.append(
            tf.summary.scalar("actor/gradient", actor_gradient_norm))
        for var, norm in zip(self.actor.trainable_weights,
                             actor_gradients_norms):
            self.actor_summaries.append(
                tf.summary.scalar("actor/{}".format(var.name), norm))

        # FIXME: Use directly Keras backend
        # This is a kind of a hack
        # Taken from the "initialize_variables" of the Keras Tensorflow backend
        # https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py#L330
        # It permits to only initialize variables that are not already initialized
        # Without that, the networks and target networks get initialized again, to different values (stochastic initialization)
        # This is a problem when a network and it's target network do not begin with the same parameter values...
        variables = tf.global_variables()
        uninitialized_variables = []
        for v in variables:
            if not hasattr(v,
                           '_keras_initialized') or not v._keras_initialized:
                uninitialized_variables.append(v)
                v._keras_initialized = True
        self.session.run(tf.variables_initializer(uninitialized_variables))
        # self.session.run(tf.global_variables_initializer())

        self.merged_summary = tf.summary.merge_all()

        self.compiled = True

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def save(self, name="DDPG"):
        """Save the model as an HDF5 file"""
        self.actor.save(name + "_actor.h5")
        self.critic.save(name + "_critic.h5")

    def update_target_critic_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())

    def update_target_actor_hard(self):
        self.target_actor.set_weights(self.actor.get_weights())

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()

        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, state):
        # [state] is the unprocessed version of a batch
        batch_state = self.process_state_batch([state])
        # We get a batch of 1 action
        # action = self.actor.predict_on_batch(batch_state)[0]
        action = self.session.run(
            self.actor(self.variables.state),
            feed_dict={self.variables.state: batch_state,
                       K.learning_phase(): 0})[0]
        assert action.shape == (self.nb_actions, )

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        # Clip the action value, even if the noise is making it exceed its bounds
        action = np.clip(action, self.actions_low, self.actions_high)
        return action

    def forward(self, observation):
        # Select an action.
        # state = self.memory.get_recent_state(observation)
        # action = self.select_action(state)
        action = self.select_action(observation)

        if self.processor is not None:
            action = self.processor.process_action(action)

        return (action)

    def backward(self,
                 observation_0,
                 action,
                 reward,
                 observation_1,
                 terminal=False,
                 fit_actor=True,
                 fit_critic=True,
                 epoch=1):

        # Default values
        metrics = [np.nan for _ in self.metrics_names]
        summaries = []

        # Stop here if not training
        if not self.training:
            return((metrics, summaries))

        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(observation_0, action, reward, observation_1,
                               terminal)

        # Train the network on a single stochastic batch.
        if self.step % self.train_interval == 0:
            # Update critic, if warm up is over.
            fit_critic = (fit_critic and
                          self.step > self.nb_steps_warmup_critic)
            # Update critic, if warm up is over.
            fit_actor = (fit_actor and self.step > self.nb_steps_warmup_actor)

            # Hard update the target nets if necessary
            hard_update_target_actor = self.step % self.target_actor_update == 0
            hard_update_target_critic = self.step % self.target_critic_update == 0

            if (fit_actor or fit_critic):
                summaries = self.fit_nets(
                    fit_critic=fit_critic,
                    fit_actor=fit_actor,
                    hard_update_target_critic=hard_update_target_critic,
                    hard_update_target_actor=hard_update_target_actor,
                    epoch=epoch)

        return((metrics, summaries))

    def fit_offline(self,
                    fit_critic=True,
                    fit_actor=True,
                    hard_update_target_critic=False,
                    hard_update_target_actor=False,
                    epochs=1):
        for epoch in range(epochs):
            # TODO: Implement callback support
            print("Training epoch {}".format(epoch))
            self.fit_nets(
                fit_critic=fit_critic,
                fit_actor=fit_actor,
                hard_update_target_critic=hard_update_target_critic,
                hard_update_target_actor=hard_update_target_actor)

    def fit_nets(self,
                 fit_critic=True,
                 fit_actor=True,
                 hard_update_target_critic=False,
                 hard_update_target_actor=False,
                 epoch=1):
        """Fit the actor and critic networks"""
        # TODO: Export metrics to tensorboard

        if not (fit_actor or fit_critic):
            return (None)
        else:
            batch = self.get_batch()

            metrics = []
            if fit_critic:
                metrics_critic = self.fit_critic(batch)
                metrics += metrics_critic

            if fit_actor:
                metrics_actor = self.fit_actor(batch)
                metrics += metrics_actor

            # Hard update target networks, only if necessary
            if self.target_actor_update >= 1:
                if hard_update_target_actor:
                    self.update_target_actor_hard()
            else:
                self.session.run(self.target_actor_train_fn)
            # Hard update target networks, only if necessary
            if self.target_critic_update >= 1:
                if hard_update_target_critic:
                    self.update_target_critic_hard()
            else:
                self.session.run(self.target_critic_train_fn)
            return (metrics)

    def fit_critic(self, batch, sgd_iterations=1):
        """Fit the critic network"""
        # Get the target action
        # \pi(s_t)
        if USE_KERAS_INFERENCE:
            target_actions = self.target_actor.predict_on_batch(batch.state_1)
        else:
            target_actions = self.session.run(
                self.target_actor(self.variables.state),
                feed_dict={self.variables.state: batch.state_1,
                           K.learning_phase(): 0})
        assert target_actions.shape == (self.batch_size, self.nb_actions)

        # Get the target Q value
        # Q(s_t, \pi(s_t))
        if USE_KERAS_INFERENCE:
            target_q_values = self.target_critic.predict_on_batch(
                [batch.state_1, target_actions]).flatten()
        else:
            target_q_values = self.session.run(
                self.target_critic([self.variables.state, self.variables.action]),
                feed_dict={
                    self.variables.state: batch.state_1,
                    self.variables.action: target_actions
                }).flatten()

        # Also works
        assert target_q_values.shape == (self.batch_size, )

        # Compute the critic targets:
        # r_t + gamma * Q(s_t, \pi(s_t))
        # but only for the affected output units (as given by action_batch).
        discounted_reward_batch = self.gamma * target_q_values
        discounted_reward_batch *= batch.terminal_1
        assert discounted_reward_batch.shape == batch.reward.shape
        critic_targets = (batch.reward + discounted_reward_batch).reshape(
            self.batch_size, 1)

        # Perform a single batch update on the critic network.
        for _ in range(sgd_iterations):
            # FIXME: metrics collection won't work with more than one iteration
            _, metrics = self.session.run(
                [self.critic_train_fn, self.critic_summaries],
                feed_dict={
                    self.variables.state: batch.state_0,
                    self.variables.action: batch.action,
                    self.critic_target: critic_targets
                })

        # FIXME: Remove this
        # if self.processor is not None:
        #     metrics += self.processor.metrics

        return (metrics)

    def fit_actor(self, batch, sgd_iterations=1):
        """Fit the actor network"""
        # TODO: implement metrics for actor

        inputs = [batch.state_0, +batch.state_0]

        if self.uses_learning_phase:
            inputs += [self.training]

        for _ in range(sgd_iterations):
            # FIXME: metrics collection won't work with more than one iteration
            _, metrics = self.session.run(
                [self.actor_train_fn, self.actor_summaries],
                feed_dict={self.variables.state: batch.state_0,
                           K.learning_phase(): 1})

        return (metrics)

    def get_batch(self):
        """
        Get and process a batch
        Split each a batch of experiences into batches of state_0, action etc...
        """
        # TODO: Remove this function
        # Store directly the different batches into memory
        experiences = self.memory.sample(self.batch_size)
        assert len(experiences) == self.batch_size

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        # TODO: Remove this, we do not need processors for now
        state0_batch = self.process_state_batch(state0_batch)
        state1_batch = self.process_state_batch(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        action_batch = np.array(action_batch)
        assert reward_batch.shape == (self.batch_size, )
        assert terminal1_batch.shape == reward_batch.shape
        assert action_batch.shape == (self.batch_size, self.nb_actions)

        batch = Batch(
            state_0=state0_batch,
            action=action_batch,
            reward=reward_batch,
            terminal_1=terminal1_batch,
            state_1=state1_batch)
        return (batch)


def process_parameterization_variable(param):
    # Soft vs hard target model updates.
    if param < 0:
        raise ValueError('`target_model_update` must be >= 0, currently at {}'.
                         format(param))
    elif param >= 1:
        # Hard update every `target_model_update` steps.
        param = int(param)
    else:
        # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
        param = float(param)
    return (param)
