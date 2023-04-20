"""
Inspired by: https://pettingzoo.farama.org/tutorials/cleanrl/implementing_PPO/
"""

import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.distributions.categorical import Categorical
from pettingzoo.mpe import simple_push_v2
import matplotlib.pyplot as plt
import time

# simple simultaneous gradient descent optimizer
class sim_gd(object):
    
    def __init__(self,
                 learning_rate,
                 device,
                ):
        
        self.device = device
        self.learning_rate = learning_rate
    
    def step(self, loss_list):

        # compute gradient of actors' w.r.t. parameters
        grad_list = [
            autograd.grad(loss_list[0], adversary.parameters(), retain_graph=True, allow_unused=True),
            autograd.grad(loss_list[1], agent.parameters(), retain_graph=True, allow_unused=True),
        ]
        # perform SimGD update based on loss gradient
        for grad, param in zip(grad_list[0], [param for param in adversary.parameters()]):
            param.data -= grad * self.learning_rate
        for grad, param in zip(grad_list[1], [param for param in agent.parameters()]):
            param.data -= grad * self.learning_rate

# actor / critic network model
class ActorCritic(nn.Module):

    def __init__(self, obs_size):

        # define simpel NN for actor / cricit networks
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(10, 5), std=0.01)
        self.critic = self._layer_init(nn.Linear(10, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):

        # initialize final layer of NN
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):

        # evaluate critic
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):

        # compute action and values
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

def batchify(x, device):

    # stacks tensors and returns result
    x = np.stack([x[a] for a in x], axis=0)
    x = torch.tensor(x).to(device)

    return x

def nn_zero_grad(parameters):

    # removes existing gradient values from parameters
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.detach()
            parameter.grad.zero_()


if __name__ == "__main__":

    # report anomalies
    torch.autograd.set_detect_anomaly(True)

    # define parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gamma = 0.99
    batch_size = 32
    max_cycles = 25
    total_episodes = 1000
    adversary_obs_size = 8
    agent_obs_size = 19

    # set up environment
    env = simple_push_v2.parallel_env(
        render_mode="rgb_array", max_cycles=max_cycles
    )
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    # set up learners
    adversary = ActorCritic(8).to(device)
    agent = ActorCritic(19).to(device)
    optimizer = sim_gd(0.00001, device)

    # define batch / episodic storage
    end_step = 0
    total_episodic_return = 0
    rb_logprobs = torch.zeros((batch_size, max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((batch_size, max_cycles, num_agents)).to(device)
    rb_advantages = torch.zeros((batch_size, max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((batch_size, max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((batch_size, max_cycles, num_agents)).to(device)
    batch_gradient_losses = torch.zeros((batch_size, num_agents)).to(device)

    episodic_returns = np.zeros(total_episodes)

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        
        # train using batch
        for trajectory in range(batch_size):
            
            # reset environment and return values
            next_obs = env.reset(seed=None)
            total_episodic_return = 0

            # train until the max step
            for step in range(0, max_cycles):

                # collect observations from the adversary and agent
                adversary_obs = torch.tensor(next_obs["adversary_0"]).to(device)
                agent_obs = torch.tensor(next_obs["agent_0"]).to(device)

                # get action from the adversary and agent
                adversary_actions, adversary_logprobs, _, adversary_values = adversary.get_action_and_value(adversary_obs)
                agent_actions, agent_logprobs, _, agent_values = agent.get_action_and_value(agent_obs)

                # execute the environment and log data
                adversary_actions = adversary_actions.cpu().numpy()
                agent_actions = agent_actions.cpu().numpy()
                next_obs, rewards, terms, truncs, infos = env.step(
                    {
                        "adversary_0": adversary_actions,
                        "agent_0": agent_actions,
                    }
                )

                # combine logprobs and values
                logprobs = torch.stack([adversary_logprobs, agent_logprobs], dim=0)
                values = torch.stack([adversary_values, agent_values], dim=0)

                # add to batch / episode storage
                rb_rewards[trajectory, step] = batchify(rewards, device)
                rb_terms[trajectory, step] = batchify(terms, device)
                rb_logprobs[trajectory, step] = logprobs
                rb_values[trajectory, step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[trajectory, step].cpu().numpy()

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    break

            # compute advantages
            for t in reversed(range(end_step)):
                delta = (
                    rb_rewards[trajectory, t].clone()
                    + gamma * rb_values[trajectory, t + 1].clone() * rb_terms[trajectory, t + 1].clone()
                    - rb_values[trajectory, t].clone()
                )
                rb_advantages[trajectory, t] = delta + gamma * gamma * rb_advantages[trajectory, t + 1].clone()

            # create gamma tensor
            gamma_tensor = torch.vander(torch.tensor([gamma]), N=max_cycles, increasing=True).T

            # compute gradient losses per (16) in https://arxiv.org/pdf/2111.08565.pdf
            for agent_ in range(num_agents):
                batch_gradient_losses[trajectory, agent_] = -torch.sum(rb_logprobs[trajectory, :, agent_].clone() * gamma_tensor * rb_advantages[trajectory, :, agent_].clone())
   
        # take expectation using mean
        expected_gradient_losses = torch.mean(batch_gradient_losses, dim=0)

        # zero gradients and step
        nn_zero_grad(adversary.parameters())
        nn_zero_grad(agent.parameters())
        optimizer.step(expected_gradient_losses)

        # print episode results
        print(f"Training episode {episode}")
        print(f"Episodic Return: {total_episodic_return[1]}")
        print(f"Episode Length: {end_step}")
        print("")
        print("\n-------------------------------------------\n")

""" - ENVIRONMENT RENDERING
env = simple_push_v2.parallel_env(render_mode="human")
adversary.eval()
agent.eval()

with torch.no_grad():
    # render 5 episodes out
    for episode in range(5):
        obs = env.reset(seed=None)
        terms = [False]
        truncs = [False]
        while not any(terms) and not any(truncs):
            adversary_obs = torch.tensor(obs["adversary_0"]).to(device)
            agent_obs = torch.tensor(obs["agent_0"]).to(device)
            adversary_actions, adversary_logprobs, _, adversary_values = adversary.get_action_and_value(adversary_obs)
            agent_actions, agent_logprobs, _, agent_values = agent.get_action_and_value(agent_obs)
            adversary_actions = adversary_actions.cpu().numpy()
            agent_actions = agent_actions.cpu().numpy()
            obs, rewards, terms, truncs, infos = env.step(
                {
                    "adversary_0": adversary_actions,
                    "agent_0": agent_actions,
                }
            )
            terms = [terms[a] for a in terms]
            truncs = [truncs[a] for a in truncs]
            time.sleep(0.1)
"""

plt.scatter(np.arange(total_episodes), episodic_returns)
plt.show()
