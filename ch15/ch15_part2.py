#!/usr/bin/env python
# coding: utf-8

# Source codes for Python Machine Learning By Example 4th Edition (Packt Publishing)
# 
# Chapter 15 Making Decisions in Complex Environments with Reinforcement Learning
# 
# Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)

# # Performing Monte Carlo learning

# ## Simulating the Blackjack environment

import gymnasium as gym

env = gym.make('Blackjack-v1')

env.reset(seed=0)


env.step(1)


env.step(1)


env.step(0)


# ## Performing Monte Carlo policy evaluation

def run_episode(env, hold_score):
    state, _ = env.reset()
    rewards = []
    states = [state]
    while True:
        action = 1 if state[0] < hold_score else 0
        state, reward, terminated, truncated, info = env.step(action)
        is_done = terminated or truncated
        states.append(state)
        rewards.append(reward)
        if is_done:
            break
    return states, rewards


from collections import defaultdict

def mc_prediction_first_visit(env, hold_score, gamma, n_episode):
    V = defaultdict(float)
    N = defaultdict(int)
    for episode in range(n_episode):
        states_t, rewards_t = run_episode(env, hold_score)
        return_t = 0
        G = {}
        for state_t, reward_t in zip(states_t[1::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[state_t] = return_t
        for state, return_t in G.items():
            if state[0] <= 21:
                V[state] += return_t
                N[state] += 1
    for state in V:
        V[state] = V[state] / N[state]
    return V


gamma = 1
hold_score = 18
n_episode = 500000

value = mc_prediction_first_visit(env, hold_score, gamma, n_episode)

print(value)

print('Number of states:', len(value))


# ## Performing on-policy Monte Carlo control

import torch

def run_episode(env, Q, n_action):
    state, _ = env.reset()
    rewards = []
    actions = []
    states = []
    action = torch.randint(0, n_action, [1]).item()
    while True:
        actions.append(action)
        states.append(state)
        state, reward, terminated, truncated, info = env.step(action)
        is_done = terminated or truncated
        rewards.append(reward)
        if is_done:
            break
        action = torch.argmax(Q[state]).item()
    return states, actions, rewards


def mc_control_on_policy(env, gamma, n_episode):
    n_action = env.action_space.n
    G_sum = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.empty(n_action))
    for episode in range(n_episode):
        states_t, actions_t, rewards_t = run_episode(env, Q, n_action)
        return_t = 0
        G = {}
        for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[(state_t, action_t)] = return_t
        for state_action, return_t in G.items():
            state, action = state_action
            if state[0] <= 21:
                G_sum[state_action] += return_t
                N[state_action] += 1
                Q[state][action] = G_sum[state_action] / N[state_action]
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy


gamma = 1
n_episode = 500000

optimal_Q, optimal_policy = mc_control_on_policy(env, gamma, n_episode)

print(optimal_policy)


def simulate_hold_episode(env, hold_score):
    state, _ = env.reset()
    while True:
        action = 1 if state[0] < hold_score else 0
        state, reward, terminated, truncated, info = env.step(action)
        is_done = terminated or truncated
        if is_done:
            return reward
        


def simulate_episode(env, policy):
    state, _ = env.reset()
    while True:
        action = policy[state]
        state, reward, terminated, truncated, info = env.step(action)
        is_done = terminated or truncated
        if is_done:
            return reward
        


n_episode = 100000
hold_score = 18
n_win_opt = 0
n_win_hold = 0

for _ in range(n_episode):
    reward = simulate_hold_episode(env, hold_score)
    if reward == 1:
        n_win_hold += 1
    reward = simulate_episode(env, optimal_policy)
    if reward == 1:
        n_win_opt += 1


print(f'Winning probability under the simple policy: {n_win_hold/n_episode}')
print(f'Winning probability under the optimal policy: {n_win_opt/n_episode}')


# # Solving the Blackjack problem with the Q-learning algorithm

# ## Developing the Q-learning algorithm

def epsilon_greedy_policy(n_action, epsilon, state, Q):
    probs = torch.ones(n_action) * epsilon / n_action
    best_action = torch.argmax(Q[state]).item()
    probs[best_action] += 1.0 - epsilon
    action = torch.multinomial(probs, 1).item()
    return action


def q_learning(env, gamma, n_episode, alpha, epsilon, final_epsilon):
    n_action = env.action_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    epsilon_decay = epsilon / (n_episode / 2)  
    for episode in range(n_episode):
        state, _ = env.reset()
        is_done = False
        epsilon = max(final_epsilon, epsilon - epsilon_decay)

        while not is_done:
            action = epsilon_greedy_policy(n_action, epsilon, state, Q)
            next_state, reward, terminated, truncated, info = env.step(action)
            is_done = terminated or truncated
            delta = reward + gamma * torch.max(Q[next_state]) - Q[state][action]
            Q[state][action] += alpha * delta
            total_reward_episode[episode] += reward
            if is_done:
                break
            state = next_state
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy


n_episode = 10000
epsilon = 1.0
final_epsilon = 0.1

gamma = 1
alpha = 0.003

total_reward_episode = [0] * n_episode

optimal_Q, optimal_policy = q_learning(env, gamma, n_episode, alpha, epsilon, final_epsilon)


rolling_avg_reward = [total_reward_episode[0]]
for i, reward in enumerate(total_reward_episode[1:], 1):
    rolling_avg_reward.append((rolling_avg_reward[-1]*i + reward)/(i+1))


import matplotlib.pyplot as plt
plt.plot(rolling_avg_reward)
plt.title('Average reward over time')
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.ylim([-1, 1])
plt.show()


n_episode = 100000
n_win_opt = 0

for _ in range(n_episode):
    reward = simulate_episode(env, optimal_policy)
    if reward == 1:
        n_win_opt += 1


print(f'Winning probability under the optimal policy: {n_win_opt/n_episode}')


# ---

# Readers may ignore the next cell.

get_ipython().system('jupyter nbconvert --to python ch15_part2.ipynb --TemplateExporter.exclude_input_prompt=True')

