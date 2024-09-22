import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler


import numpy as np
import torch
from torch import distributions

from rob831.infrastructure import pytorch_util as ptu
from rob831.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 learning_rate_step=1000,
                 learning_rate_gamma=0.9,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.learning_rate_step = learning_rate_step
        self.learning_rate_gamma = learning_rate_gamma
        
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            # print(f"mean_net {self.mean_net}")
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        # Initialize the learning rate scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=learning_rate_step, gamma=learning_rate_gamma)

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        # Convert observation to PyTorch tensor
        observation = ptu.from_numpy(observation)

        # Forward pass: compute the action
        if self.discrete:
            print(f"discerete")
            logits = self.logits_na(observation)
            action_distribution = torch.distributions.Categorical(logits=logits)
            action = action_distribution.sample()
        else:
            # print(f"NOT discerete")
            # print(f"observation: {observation}")
            mean = self.mean_net(observation)
            action = mean
            # print(f"mean: {mean}")
            
            # action_distribution = torch.distributions.Normal(mean, torch.exp(self.logstd))
            # action = action_distribution.sample()

        # Convert action to NumPy array
        action = ptu.to_numpy(action)
        return action

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        print(f" ************ observations update: {observations} ************ " )
        # Convert observations and actions to PyTorch tensors
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)

        # Forward pass: compute predicted actions by passing observations through the network
        if self.discrete:
            predicted_actions = self.logits_na(observations)
            loss = F.cross_entropy(predicted_actions, actions)
        else:
            predicted_actions = self.mean_net(observations)
            loss = F.mse_loss(predicted_actions, actions)

        # Backpropagation: compute gradients
        self.optimizer.zero_grad()
        loss.backward()

        # Update the model parameters
        self.optimizer.step()

        # Step the scheduler
        self.scheduler.step()

        # Get the current learning rate
        current_lr = self.scheduler.get_last_lr()[0]

        return {
            'Training Loss': ptu.to_numpy(loss),
            'Learning Rate': current_lr,
        }

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        print(f" ************ observation forward: {observation} ************ " )
        if self.discrete:
            #forward pass to get logits
            logits = self.logits_na(observation)
            return logits
        else:
            #forward pass to get mean
            mean = self.mean_net(observation)
            return mean


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        
        # Convert observations and actions to PyTorch tensors
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)

        # Forward pass: compute predicted actions by passing observations through the network
        if self.discrete:
            predicted_actions = self.logits_na(observations)
        else:
            predicted_actions = self.mean_net(observations)

        # TODO: update the policy and return the loss
        # Compute the loss between predicted actions and actual actions
        loss = self.loss(predicted_actions, actions)

        # Backpropagation: compute gradients
        self.optimizer.zero_grad()
        loss.backward()

        # Update the model parameters
        self.optimizer.step()

        # Step the scheduler
        self.scheduler.step()

        # Get the current learning rate
        current_lr = self.scheduler.get_last_lr()[0]

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
            'Learning Rate': current_lr
        }
