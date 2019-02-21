from collections import deque

import numpy as np
import torch
from unityagents import UnityEnvironment

from ddpg_agent import Agent

random_seed = 42
env = UnityEnvironment(file_name='Reacher.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = Agent(state_size=state_size, action_size=action_size, random_seed=random_seed)


def ddpg(n_episodes=1000, print_every=100):
    scores_window = deque(maxlen=100)  # last 100 scores
    all_scores = []  # list containing scores from each episode
    avg_scores_window = []
    noise_damp = 0
    max_score = 0

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(num_agents)
        while True:
            actions = agent.act(states,noise_damp)
            env_info = env.step(actions)[brain_name]

            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones, num_updates=1)
            states = next_states
            scores += rewards
            if np.any(dones):
                break
        avg_score = np.mean(scores)
        scores_window.append(avg_score)

        all_scores.append(avg_score)
        avg_scores_window.append(np.mean(scores_window))
        noise_damp = np.mean(scores_window)
        print(
            '\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_window), avg_score),
            end="")
        if i_episode % print_every == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    return all_scores


scores = ddpg()
env.close
