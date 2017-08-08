# Vinci
This is a generic, easy to use, and keras-compatible deep RL framework.

It began as a fork of [keras-rl](https://github.com/matthiasplappert/keras-rl) but is now a separated project.

# WARNING
This repository points to an **OLDER** version of Vinci, the latest one can be found at:

https://github.com/Phylliade/vinci

# Features

* Define your Deep Nets using Keras
* Simulate on the OpenAI Gym Environments
* Easy to implement a new algorithm, using a well-defined API
* Advanced training capabilities: Offline training, critic-only (or actor-only) training...
* Easy logging : Tensorboard, Terminal...

# Installation
Run :
```
pip install  git+ssh://git@github.com/Phylliade/vinci.git
```

# Creating the Deep Networks with Keras
Vinci is designed to seamlessy use Keras's networks.
You can design your networks as always, using the Sequential or Functional API.


## Environment-agnostic networks
Vinci also adds some utilities to make the network creation **environment agnostic**, which can be nice!

To do this, the `env` object (a wrapper around a gym env, of type `rl.EnvWrapper`) provides different utilities, depending if you're using the Sequential or Functional APIs.

### Using the functional API
You just have to design your Keras model using the functional API and the `state` and `action` placeholders  of the `env` object.

For example, for a simple critic:
```python
# Inputs
observation = env.state
action = env.action
# Concatenate the inputs for the critic
inputs = concatenate([observation, action])

# Hidden layer
x = Dense(100)(inputs)
x = Activation('relu')(x)

# Output layer
x = Dense(1)(x)
x = Activation('linear')(x)

# Final model
critic = Model(inputs=[observation, action], outputs=[x])
```

### Using the Sequential API
Since you have to specify the input shapes by hand, you can use the `state_space_dim` and `action_space_dim` attributes of the EnvWrapper.

For example of an actor:
```python
actor = Sequential()

# Hidden layers
actor.add(Dense(400, input_shape=(env.state_space_dim,)))
actor.add(Activation("relu"))
actor.add(Dense(300))
actor.add(Activation("relu"))

# Output layer
actor.add(Dense(env.action_space_dim, activation="tanh"))
```

## Efficiency of Keras models
Internally, Keras models are used in a functional fashion:

```
out = keras_model(in)
```

Some may wonder about some potential leaks with this usage, and they're right!
With a traditional function, each time `keras_model(in)` is called, a new `Tensor` is created (and every underlying ops) and added to the Graph.

But, Keras uses a cache for the computations, so each call to `keras_model(in)` always resulsts in the same variable.
