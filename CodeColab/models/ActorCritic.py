#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 10:54:01 2023

@author: goyal
"""
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

device = torch.device('cpu')

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        hidden_lyr_n =  64;
        if (state_dim) >= 16:
            hidden_lyr_n = 256
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space :
            # Action is considered to be bidirectional [-1,1]
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, hidden_lyr_n),
                            nn.Tanh(),
                            nn.Linear(hidden_lyr_n, hidden_lyr_n),
                            nn.Tanh(),
                            nn.Linear(hidden_lyr_n, action_dim),
                            nn.Tanh()
                        )             
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, hidden_lyr_n),
                            nn.Tanh(),
                            nn.Linear(hidden_lyr_n, hidden_lyr_n),
                            nn.Tanh(),
                            nn.Linear(hidden_lyr_n, action_dim),
                            nn.Softmax(dim=-1)
                        )

        #print(self.actor)
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, hidden_lyr_n),
                        nn.Tanh(),
                        nn.Linear(hidden_lyr_n, hidden_lyr_n),
                        nn.Tanh(),
                        nn.Linear(hidden_lyr_n, 1)
                    )
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
    

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        
        #print(f"distribution {dist}")
        action = dist.sample()
        # modified to clip the actions to given range
        action = torch.clip(action,-1,1) # always return action between +1 and -1
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
