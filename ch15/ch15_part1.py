#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 15 Making Decisions in Complex Environments with Reinforcement Learning
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Setting up the working environment

# ## Installing OpenAI Gym 

import gymnasium as gym
print(gym.envs.registry.keys())


# # Solving the FrozenLake environment with dynamic programming

# ## Simulating the FrozenLake environment

env = gym.make("FrozenLake-v1", render_mode="rgb_array")
 
n_state = env.observation_space.n
print(n_state)
n_action = env.action_space.n
print(n_action)


env.reset(seed=0)


import matplotlib.pyplot as plt
plt.imshow(env.render())  


new_state, reward, terminated, truncated, info = env.step(2)
is_done = terminated or truncated
    
env.render()
print(new_state)
print(reward)
print(is_done)
print(info)


plt.imshow(env.render())


def run_episode(env, policy):
    state, _ = env.reset()
    total_reward = 0
    is_done = False
    while not is_done:
        action = policy[state].item()
        state, reward, terminated, truncated, info = env.step(action)
        is_done = terminated or truncated
        total_reward += reward
        if is_done:
            break
    return total_reward


import torch

n_episode = 1000

total_rewards = []
for episode in range(n_episode):
    random_policy = torch.randint(high=n_action, size=(n_state,))
    total_reward = run_episode(env, random_policy)
    total_rewards.append(total_reward)

print(f'Average total reward under random policy: {sum(total_rewards)/n_episode}')


print(env.env.P[6])


# ## Solving FrozenLake with the value iteration algorithm

def value_iteration(env, gamma, threshold):
    """
    Solve a given environment with value iteration algorithm
    @param env: Gymnasium environment
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: values of the optimal policy for the given environment
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.empty(n_state)
        for state in range(n_state):
            v_actions = torch.zeros(n_action)
            for action in range(n_action):
                for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                    v_actions[action] += trans_prob * (reward + gamma * V[new_state])
            V_temp[state] = torch.max(v_actions)
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V


gamma = 0.99
threshold = 0.0001

V_optimal = value_iteration(env, gamma, threshold)
print('Optimal values:\n', V_optimal)



def extract_optimal_policy(env, V_optimal, gamma):
    """
    Obtain the optimal policy based on the optimal values
    @param env: Gymnasium environment
    @param V_optimal: optimal values
    @param gamma: discount factor
    @return: optimal policy
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    optimal_policy = torch.zeros(n_state)
    for state in range(n_state):
        v_actions = torch.zeros(n_action)
        for action in range(n_action):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                v_actions[action] += trans_prob * (reward + gamma * V_optimal[new_state])
        optimal_policy[state] = torch.argmax(v_actions)
    return optimal_policy


optimal_policy = extract_optimal_policy(env, V_optimal, gamma)
print('Optimal policy:\n', optimal_policy)


def run_episode(env, policy):
    state, _ = env.reset()
    total_reward = 0
    is_done = False
    while not is_done:
        action = policy[state].item()
        state, reward, terminated, truncated, info = env.step(action)
        is_done = terminated or truncated
        total_reward += reward
        if is_done:
            break
    return total_reward


n_episode = 1000
total_rewards = []
for episode in range(n_episode):
    total_reward = run_episode(env, optimal_policy)
    total_rewards.append(total_reward)

print('Average total reward under the optimal policy:', sum(total_rewards) / n_episode)


# ## Solving FrozenLake with the policy iteration algorithm

def policy_evaluation(env, policy, gamma, threshold):
    """
    Perform policy evaluation
    @param env: Gymnasium  environment
    @param policy: policy matrix containing actions and their probability in each state
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: values of the given policy
    """
    n_state = policy.shape[0]
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(n_state):
            action = policy[state].item()
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                V_temp[state] += trans_prob * (reward + gamma * V[new_state])
        max_delta = torch.max(torch.abs(V - V_temp))
        V = V_temp.clone()
        if max_delta <= threshold:
            break
    return V


def policy_improvement(env, V, gamma):
    """
    Obtain an improved policy based on the values
    @param env: Gymnasium  environment
    @param V: policy values
    @param gamma: discount factor
    @return: the policy
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.zeros(n_state)
    for state in range(n_state):
        v_actions = torch.zeros(n_action)
        for action in range(n_action):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                v_actions[action] += trans_prob * (reward + gamma * V[new_state])
        policy[state] = torch.argmax(v_actions)
    return policy


def policy_iteration(env, gamma, threshold):
    """
    Solve a given environment with policy iteration algorithm
    @param env: Gymnasium  environment
    @param gamma: discount factor
    @param threshold: the evaluation will stop once values for all states are less than the threshold
    @return: optimal values and the optimal policy for the given environment
    """
    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.randint(high=n_action, size=(n_state,)).float()
    while True:
        V = policy_evaluation(env, policy, gamma, threshold)
        policy_improved = policy_improvement(env, V, gamma)
        if torch.equal(policy_improved, policy):
            return V, policy_improved
        policy = policy_improved


gamma = 0.99
threshold = 0.0001


V_optimal, optimal_policy = policy_iteration(env, gamma, threshold)
print('Optimal values:\n', V_optimal)
print('Optimal policy:\n', optimal_policy)


# ---

# Readers may ignore the next cell.



