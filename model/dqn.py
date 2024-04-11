import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torchcontrib.optim import SWA
from collections import deque
from util.preprocess import *
from util.helper_functions import set_manual_seed

set_manual_seed()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, model_size):

        print(f"input_dim:{input_dim}, output_dim:{output_dim}")
        super(DQN, self).__init__()
        self.input_dim = input_dim + 2 #change
        self.output_dim = output_dim

        layer_sizes = {
            "small": [32],
            "medium": [192, 48, 24],
            "big": [384, 192, 96, 48, 24],
            "large": [512, 384, 256, 192, 128, 64, 32]
        }.get(model_size, [32])

        layers = [nn.Linear(self.input_dim, layer_sizes[0]), nn.ReLU()]
        for i in range(len(layer_sizes) - 1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(layer_sizes[-1], self.output_dim))

        self.fc = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        return self.fc(state)
    
class DQNAgent:

    def __init__(self, dataset, buffer, config_dict, pre_trained_model=None):

        self.dataset = dataset 
        self.replay_buffer = buffer

        self.model = pre_trained_model or DQN(
            config_dict.get('input_dim', 768), 
            config_dict.get('output_dim', 1), 
            config_dict.get('model_size', 'small')
        )

        self.learning_rate = config_dict.get('learning_rate', 1e-4)
        print(f"self.learning_rate:{self.learning_rate}")
        self.gamma = config_dict.get('gamma', 0.99)
        self.tau = config_dict.get('tau', 0.005)
        self.swa = config_dict.get('swa', False)

        self.MSE_loss = nn.MSELoss()

        base_opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.swa:
            self.optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
        else:
            self.optimizer = base_opt


    def get_action(self, state, dataset=None):

        dataset = dataset if not dataset.empty else self.dataset

        inputs = get_multiple_model_inputs(state, state.remaining, dataset)
        
        model_inputs = autograd.Variable(torch.from_numpy(inputs).float().unsqueeze(0))
        expected_returns = self.model.forward(model_inputs)
        value, index = expected_returns.max(1)
        return state.remaining[index[0]]

    def compute_loss(self, batch, dataset, verbose=True):

        states, actions, rewards, next_states, dones = batch
        
        model_inputs = [ get_model_inputs(states[i], actions[i], dataset, False) for i in range(len(states))]
        model_inputs = torch.FloatTensor( np.array(model_inputs) )

        rewards = torch.FloatTensor( np.array(rewards) )
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(model_inputs)


        stacked_arrays = []
        for i in range(len(next_states[0].remaining)):

            temp = get_model_inputs(next_states[0], next_states[0].remaining[i], dataset)
            stacked_arrays.append(temp)

        if stacked_arrays:
            result_array = np.vstack(stacked_arrays)
            model_inputs = torch.FloatTensor(result_array)
            next_Q = self.model.forward(model_inputs)
            

            max_value, max_index = torch.max(next_Q, 1)
            max_next_Q = max_value.max().item() 
            # max_next_Q = torch.max(next_Q, 1)[0].max().item()
            max_next_Q_index = max_value.max(0)[1].item()

            

            expected_Q = rewards.squeeze(1) + (1 - dones) * self.gamma * max_next_Q
            
        else:
            expected_Q = rewards.squeeze(1)

        loss = self.MSE_loss(curr_Q.squeeze(0), expected_Q.detach())
        return loss, curr_Q, expected_Q

    def update(self, batch_size, verbose=False):
        batch = self.replay_buffer.sample(batch_size)

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch

        loss, curr_Q, expected_Q = self.compute_loss(batch, self.dataset, verbose)

        train_loss = loss.float()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.swa:
            self.optimizer.swap_swa_sgd()
        return train_loss, curr_Q