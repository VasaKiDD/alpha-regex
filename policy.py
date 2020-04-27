import torch
from typing import TypeVar, Tuple
from torch.nn import Linear, LSTMCell, Module
from torch.nn.functional import softmax
from torch.nn.init import uniform_

Vector = TypeVar("Vector", torch.Tensor, torch.cuda.FloatTensor)
Tensor = TypeVar("Tensor", torch.Tensor, torch.cuda.FloatTensor)


class Policy(Module):
    """
    This class represents an Actor-Critic network that will be used to evaluate and predict
    regex proposals. Both the Actor and the Critic heads are connected to an LSTM
    cell, that is run on a sequence vector that represents the regex proposal.

    Args:
        mask: Mask of available actions at each step of the unrolling process.
        hidden_size: Number of hidden neurons in the LSTM cell architectures.
        uniform_init: Range of initialization of weights in the network.

    Attributes:
        rnn: LTSM block where the regex proposal components are fed one by one as a
            sequence.
        actor: Output of the Actor critic containing the probability distribution of choosing
            one specific action over all the possible ones.
        critic: This linear layer is in charge of predicting a value function estimate for a
            target regex proposal, even if the sequence conforming it is not finished.

    """

    def __init__(
        self,
        n_action: int,
        max_length: int,
        regex_generator: object,
        hidden_size: int = 100,
        uniform_init: Tuple[float, float] = (-0.1, 0.1),
    ):

        super(Policy, self).__init__()
        # Internal parameters
        self._uniform_init = uniform_init
        self.generator = regex_generator
        self._hidden_size = hidden_size
        self._n_actions = n_action
        self._max_path_len = max_length
        # Initialize network
        self.rnn = None
        self.actor = None
        self.middle_critic = None
        self.critic = None
        self.init_network()

    @property
    def max_path_len(self) -> int:
        """This is the length of a complete regex proposal."""
        return self._max_path_len

    @property
    def n_actions(self) -> int:
        """Maximum number of different possible actions."""
        return self._n_actions

    def init_network(self):
        """Initialize network parameters. This is an actor-critic build on top of a RNN cell. The
        actor is a fully connected layer, and the critic consists of two fully connected layers"""
        self.rnn = LSTMCell(self.n_actions, self._hidden_size)
        for p in self.rnn.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        self.actor = Linear(self._hidden_size, self.n_actions)
        for p in self.actor.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        self.middle_critic = Linear(self._hidden_size, self._hidden_size // 2)
        for p in self.middle_critic.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        self.critic = Linear(self._hidden_size // 2, 1)
        for p in self.critic.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

    def predict(
        self, oh_action: Vector, h: Vector, c: Vector
    ) -> Tuple[Vector, Vector, Vector, Vector]:
        """
        Run the model for the given internal state, which is represented by the last action that
        was taken, and the internal state of the LSTM cell.
        Args:
            oh_action: One hot encoded vector representing the last action that was taken.
            h: Previous output of the LSTM cell.
            c: Hidden state of the LSTM cell.
            mask: Mask of available actions at the current step of the unrolling process.

        Returns:
            tuple containing the output of the actor network, the output of the critic,
            the output of the LSTM cell, and the internal state of the LSTM.
        """
        h, c = self.rnn(oh_action, (h, c))
        actor_out = self.actor(h)
        actor_out = actor_out
        critic_out = torch.nn.functional.relu(self.middle_critic(h))
        critic_out = self.critic(critic_out)
        return actor_out, critic_out, h, c

    def evaluate_solution(self, arch_proposal: Vector) -> Tuple[Vector, Vector, Vector, Vector]:
        """
        Get the outputs that the model will give at every step when trying to build the
        proposed regex. For each one of the actions taken to build the regex
        proposal it will calculate the output of the actor and the critic.
        Args:
            arch_proposal: vector representing the full regex.

        Returns:
            actor_outs, critic_outs, h, c
        """
        oh_action = torch.zeros(1, self.n_actions)
        h = torch.zeros(1, self._hidden_size)
        c = torch.zeros(1, self._hidden_size)
        if torch.cuda.is_available():
            oh_action = oh_action.cuda()
            h = h.cuda()
            c = c.cuda()
        actor_outs = []
        critic_outs = []
        for ix, action in enumerate(arch_proposal):
            actor_out, critic_out, h, c = self.predict(oh_action, h, c)
            actor_out = torch.softmax(actor_out, dim=1)[0, action]  # Added
            actor_outs.append(actor_out)
            critic_outs.append(critic_out)
            # One hot encode the next action to be taken
            oh_action = torch.zeros_like(oh_action)
            oh_action[0, action] = 1.0

        # Get last value prediction from the critic
        h, c = self.rnn(oh_action, (h, c))
        critic_out = torch.nn.functional.relu(self.middle_critic(h))
        critic_out = self.critic(critic_out)
        critic_outs.append(critic_out)

        actor_outs = torch.stack(tuple(actor_outs), dim=0)
        actor_outs = torch.squeeze(actor_outs)
        critic_outs = torch.stack(tuple(critic_outs), dim=1)
        critic_outs = torch.squeeze(critic_outs)
        return actor_outs, critic_outs[1:], h, c

    def forward_once(self, oh_action, h, c, action_ix, regex_state):
        """
        Given a one hot encoded vetor representing the last action that was taken and the
        state of the recurrent cell, run the model to sample information for the next action to
        be chosen.
        Args:
            oh_action: One hot encoded vector representing the last action that was taken.
            h: Previous output of the LSTM cell.
            c: Hidden state of the LSTM cell.
            action_ix: Index of the current step in the whole unrolling process.

        Returns:
            action_probs, critic_out, h, c, oh_action, action_chosen
        """

        if not regex_state:
            mask, new_regex_state = self.generator.reset_with_state()
            reg = None
        else:
            action = int(torch.argmax(oh_action, dim=-1))
            mask, reg, new_regex_state = self.generator.step_with_state(action, regex_state)

        finish = True if mask is None else False

        with torch.no_grad():
            oh_action = oh_action if not torch.cuda.is_available() else oh_action.cuda()
            mask = torch.FloatTensor([mask]) if mask is not None else torch.ones_like(oh_action)
            mask = mask if not torch.cuda.is_available() else mask.cuda()
            actor_out, critic_out, h, c = self.predict(oh_action, h, c)
            action_probs = softmax(actor_out, dim=-1)
            action_probs = (action_probs * mask) / action_probs.sum()
            action = torch.multinomial(action_probs, 1)
            oh_action = torch.zeros_like(oh_action)
            oh_action[0, action[0, 0]] = 1.0
        action = torch.squeeze(action)

        if finish:
            action_probs = None

        return action_probs, critic_out, h, c, oh_action, int(action), new_regex_state

    def init_tensors(self) -> Tuple[Vector, Vector, Vector]:
        """
        Creates tensors representing the internal states of the controller filled with zeros
        Returns:
            init_action, h, c which are used as an starting point for the LSTM cell.
        """
        init_action = torch.zeros(1, self.n_actions)
        h = torch.zeros(1, self._hidden_size)
        c = torch.zeros(1, self._hidden_size)
        if torch.cuda.is_available():
            init_action, h, c = (init_action.cuda(), h.cuda(), c.cuda())

        return init_action, h, c

    def sample_regex(self, length):
        """
        Sample a regex just with policy lstm
        :param length: max length of Regex
        :return: actions choosen and regex string
        """

        oh_action, h, c = self.init_tensors()
        regex_state = None
        actions = []

        for i in range(length):
            action_probs, reg, h, c, oh_action, action, regex_state = self.forward_once(
                oh_action, h, c, i, regex_state
            )
            actions.append(action)
            if action_probs is None:
                break

        reg = self.generator.get_regex(regex_state)

        return actions, reg
