import torch
from typing import TypeVar
from torch.nn import Module
import numpy as np

Vector = TypeVar("Vector", torch.Tensor, torch.cuda.FloatTensor)
Tensor = TypeVar("Tensor", torch.Tensor, torch.cuda.FloatTensor)


class VanillaPolicy(Module):
    def __init__(self, n_action, max_length, regex_generator, evaluator):

        super(VanillaPolicy, self).__init__()

        self.mask = None
        self._n_actions = n_action
        self._max_path_len = max_length
        self.generator = regex_generator
        self.evaluator = evaluator

        self.init_network()

    @property
    def max_path_len(self) -> int:
        """This is the length of a complete architecture proposal."""
        return self._max_path_len

    @property
    def n_actions(self) -> int:
        """Maximum number of different possible actions."""
        return self._n_actions

    def init_network(self):
        """Initialize network parameters. This is an actor-critic build on top of a RNN cell. The
        actor is a fully connected layer, and the critic consists of two fully connected layers"""
        self.empty_actions = torch.zeros(1, self.n_actions)
        self.h = "internal"
        self.c = "internal"

    def forward_once(self, oh_action, h, c, action_ix, regex_state):
        if not regex_state:
            mask, new_regex_state = self.generator.reset_with_state()
            regex = None
        else:
            action = int(torch.argmax(oh_action, dim=-1))
            mask, regex, new_regex_state = self.generator.step_with_state(action, regex_state)

        if mask is not None:
            action_probs = mask / np.sum(mask)
            action_probs = torch.from_numpy(action_probs)
            action = np.random.choice(len(mask), None, p=action_probs)
            oh_action = torch.zeros(1, len(mask))
            oh_action[0, action] = 1.0
        else:
            action_probs = None

        if regex:
            reward = self.evaluator.evaluate(regex)
        else:
            reward = 0.0

        return action_probs, reward, h, c, oh_action, action, new_regex_state

    def init_tensors(self):

        return self.empty_actions, self.h, self.c
