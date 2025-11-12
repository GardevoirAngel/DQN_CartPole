
"""
CartPole-v1 -- Deep Q-learning
"""
import os
import random
from collections import deque

import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

class CustomRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.theta_idx = 2  # Pole angle (obs[2])
        self.pos_idx = 0    # Cart position (obs[0])
    
    def step(self, action):
        # Get original step result
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Custom reward: -1 * (pole angle + position)
        pole_angle = obs[self.theta_idx]
        cart_position = obs[self.pos_idx]
        custom_reward = -1 * (pole_angle + cart_position)
        
        # Optional: Scale if too small (e.g., multiply by 10 for bigger signal)
        # custom_reward *= 10  # Uncomment/test if rewards feel weak
        
        # Replace original reward (+1/step) with custom
        return obs, custom_reward, terminated, truncated, info
        
class Agent:
    def __init__(self, state_size, action_size, batch_size=64, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state)[0])

    def experience_replay(self):
        # Updates the network weights after enough data is collected
        if self.batch_size >= len(self.memory):
            return

        # Samples a batch from the memory
        random_batch = random.sample(self.memory, self.batch_size)

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = random_batch[i][0]
            action.append(random_batch[i][1])
            reward.append(random_batch[i][2])
            next_state[i] = random_batch[i][3]
            done.append(random_batch[i][4])

        # Batch prediction to save compute costs
        target = self.model.predict(state)
        target_next = self.model(next_state)

        for i in range(len(random_batch)):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(
            np.array(state),
            np.array(target),
            batch_size=self.batch_size,
            verbose=0
        )

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load_weights(self, weights_file):
        self.epsilon = self.epsilon_min
        self.model.load_weights(weights_file)

    def save_weights(self, weights_file):
        self.model.save_weights(weights_file)


if __name__ == "__main__":
    # Flag used to enable or disable screen recording
    recording_is_enabled = False

    # Initializes the environment
    env = gym.make('CartPole-v1')
    env = CustomRewardWrapper(env)  # Wrap with custom rewards

    # Records the environment
    if recording_is_enabled:
        env = gym.wrappers.Monitor(env, "recording", video_callable=lambda episode_id: True, force=True)

    # Defines training related constants
    num_episodes = 200
    num_episode_steps = env.spec.max_episode_steps
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]
    max_reward = 0
    rewards = []  # List to store per-episode rewards (episode lengths)

    # Creates the agent
    agent = Agent(state_size=state_size, action_size=action_size)

    # Loads the weights
    if os.path.isfile("third_run.weights.h5"):
        agent.load_weights("third_run.weights.h5")

    for episode in range(num_episodes):
        # Defines the total reward per episode
        total_reward = 0

        # Resets the environment
        observation, _ = env.reset()

        # Gets the state
        state = np.reshape(observation, [1, state_size])

        for episode_step in range(num_episode_steps):
            # Renders the screen after new environment observation
            # env.render(mode="human")  # Disabled for Colab (headless)

            # Gets a new action
            action = agent.act(state)

            # Takes action and calculates the total reward
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward = reward if not done else -10
            total_reward += reward

            # Gets the next state
            observation = next_observation
            next_state = np.reshape(observation, [1, state_size])

            # Memorizes the experience
            agent.memorize(state, action, reward, next_state, done)

            # Updates the state
            state = next_state

            # Updates the network weights
            agent.experience_replay()

            if done:
                episode_length = episode_step + 1
                print("Episode %d/%d finished after %d episode steps with total reward = %f."
                      % (episode + 1, num_episodes, episode_length, total_reward))
                rewards.append(episode_length)
                break

            elif episode_step >= num_episode_steps - 1:
                episode_length = num_episode_steps
                print("Episode %d/%d timed out at %d with total reward = %f."
                      % (episode + 1, num_episodes, episode_step + 1, total_reward))
                rewards.append(episode_length)

        # Saves the network weights
        if total_reward >= max_reward:
            agent.save_weights("third_run.weights.h5")
            max_reward = total_reward

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='blue', linewidth=1.5, label='Episode Reward')
    plt.axhline(y=-1, color='red', linestyle='--', alpha=0.7, label='Solved Threshold (195+)')
    plt.xlabel('Episode Number')
    plt.ylabel('Custom Reward -|theta|')
    plt.title('DQN CartPole: Reward vs. Episode (200 Episodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.savefig('third_run.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'third_run.png'. Open it to view the graph!")

    # Closes the environment
    env.close()

