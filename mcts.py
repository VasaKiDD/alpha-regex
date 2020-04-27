import numpy as np
import torch
from typing import List
from policy import Vector, Tuple, Policy
from vanilla_policy import VanillaPolicy
import copy


class MCTS:
    """
    This class is used to perform a search over the state space for different paths by building
    a tree of visited states. Then this tree is used to get an estimation distribution of
    utility over actions.

    Args:
        policy: Policy to be used as a prior over actions given an state.
        c_puct: Constant that modifies the exploration-exploitation tradeoff of the MCTS algorithm.
        nodes_per_tree: The number of nodes that we will be visiting when building an MCTS tree.
        temperature: Another parameter that balances exploration-exploitation  in MCTS by
         adding noise to the priors output by the search.
    """

    def __init__(
        self,
        policy: object,
        c_puct: float = 1.0,
        nodes_per_tree: int = 1000,
        temperature: float = 1.0,
        use_dirichlet_noise: bool = False,
        dir_epsilon: float = 0.25,
        dir_noise: float = 0.03,
        exploit: bool = False,
    ):
        self.policy = policy
        self.c_puct = c_puct
        self.nodes_per_tree = nodes_per_tree
        self.temperature = temperature
        self.n_expanded_nodes = 0
        self.max_path_len = self.policy.max_path_len
        self.dirichlet_noise = use_dirichlet_noise
        self.dir_epsilon = dir_epsilon
        self.dir_noise = dir_noise
        self.exploit = exploit

    def expand_node(self, node: dict) -> Tuple[dict, float, Vector, Vector]:
        """
        Used for previously unvisited nodes. It evaluates each of the possible child and
        initializes them with an score derived from the prior output by the policy network.

        Args:
            node: Node to be expanded

        Returns:
            node, reward, h, c
        """

        def convert_to_one_hot(action):
            transformed_act = torch.zeros(1, self.policy.n_actions)
            if torch.cuda.is_available():
                transformed_act = transformed_act.cuda()
            transformed_act[0, action] = 1.0
            return transformed_act

        oh_action, h, c, action_ix, regex_state = (
            node["oh_action"],
            node["h_lstm"],
            node["c_lstm"],
            node["action_ix"],
            node["regex_state"],
        )

        priors, reward, h, c, oh_action, op, new_regex_state = self.policy.forward_once(
            oh_action, h, c, action_ix, regex_state
        )

        if priors is None:
            return node, reward, h, c

        priors = torch.squeeze(priors)
        priors = priors.cpu().numpy()

        # Initialize its children with its probability of being chosen
        for num_child in range(int(priors.size)):
            if priors[num_child] > 0.0:
                new_child = {
                    "parent": node,
                    "childs": [],
                    "visit_count": 0.0,
                    "total_action_value": 0.0,
                    "prior": float(priors[num_child]),  # probability of being chosen
                    "oh_action": convert_to_one_hot(num_child),
                    "h_lstm": h,
                    "c_lstm": c,
                    "action_ix": action_ix + 1,
                    "regex_state": copy.deepcopy(new_regex_state),
                }
                node["childs"].append(new_child)
        # This reward will be propagated backwards through the tree
        if isinstance(self.policy, Policy):
            reward = float(torch.sigmoid(reward))
        return node, reward, h, c

    def add_dirichlet_noise(self, node: dict):
        noise = np.random.dirichlet([self.dir_noise] * len(node["childs"]))

        for num in range(len(node["childs"])):
            child = node["childs"][num]
            if child["prior"] > 0.0:
                child["prior"] = (1 - self.dir_epsilon) * child[
                    "prior"
                ] + self.dir_epsilon * noise[num]

    def estimate_q_val(self, node: dict) -> Tuple[dict, int]:
        """
        Estimates the Q value over possible actions in a given node, and returns the action
        and the child that have the best estimated value.

        Args:
            node: Node to evaluate its possible actions.

        Returns:
            best child found, action that leads to the best child
        """
        best_child = None
        best_action = 0
        best_val = -np.inf
        # Iterate all the children to fill up the node dict and estimate Q val.
        # Then track the best child found according to the Q value estimation

        for num in range(len(node["childs"])):
            child = node["childs"][num]
            if child["prior"] > 0.0:
                q_val_action = (
                    child["total_action_value"] / child["visit_count"]
                    if child["visit_count"] > 0.0
                    else 0.0
                )
                action_utility = (
                    self.c_puct
                    * child["prior"]
                    * np.sqrt(node["visit_count"])
                    * (1.0 / (1.0 + child["visit_count"]))
                )
                # action_utility = 2 * np.sqrt(np.log(node["visit_count"])/(child["visit_count"] + 1.0))
                q_val_action += action_utility
                if q_val_action > best_val:
                    best_val = q_val_action
                    best_child = child
                    best_action = int(torch.argmax(child["oh_action"]))
        return best_child, best_action

    def sample_distributions(self, root_node: dict) -> Tuple[Vector, Vector]:
        """
        Select the action with the highest Q value from the root node in the MCTS tree.

        Args:
            root_node: Node to choose the best action from. It should be the root node of the tree.

        Returns:
            Probabilities over possible actions, action with highest probability.

        """
        n_children = self.policy.n_actions
        visits_policy = torch.zeros(1, n_children)
        qval_dist = torch.zeros(1, n_children)
        if torch.cuda.is_available():
            visits_policy = visits_policy.cuda()
            qval_dist = qval_dist.cuda()

        for i_child in range(len(root_node["childs"])):
            child = root_node["childs"][i_child]
            if child["prior"] > 0.0:
                q_val_act = (
                    child["total_action_value"] / child["visit_count"]
                    if child["visit_count"] > 0.0
                    else 0.0
                )
                visits_policy[0, i_child] = child["visit_count"]
                qval_dist[0, i_child] = float(q_val_act)
        return visits_policy, qval_dist

    def sample_policy(
        self, visits_policy: Vector, qval_dist: Vector, exploit: bool = False
    ) -> Tuple[Vector, int]:
        """
        Sample an action from the policies and q_value distributions that were previously sampled.

        Args:
            visits_policy: distribution of the number of times the root node has been visited.
            qval_dist: Distribution of the q_values over actions in the root node.
            exploit: Select the most probable action if True. Sample from the distribution if False.

        Returns:
            Tuple containing the sampled action and the probability distribution build normalizing visits_policy.
        """

        mcts_policy = visits_policy / visits_policy.sum()
        if exploit:
            return mcts_policy, int(mcts_policy.argmax())
        else:
            policy = torch.pow(visits_policy, self.temperature)
            policy = policy / policy.sum()
            sampled_act = int(torch.multinomial(policy, 1)[0, 0])
            return mcts_policy, sampled_act

    def sample_regex(self):
        """
        Perform an MCTS search over the parameter space and update the controller with the
        estimated Q value function.

        Returns:
            A list containing the actions chosen (a regex proposal), and the policies
            output by the actor at each time step.
        """

        sampled_oh_action, state_h, state_c = self.policy.init_tensors()

        root_node = {
            "parent": None,
            "childs": [],
            "visit_count": 1,
            "total_action_value": 0.0,
            "prior": None,
            "oh_action": sampled_oh_action,
            "h_lstm": state_h,
            "c_lstm": state_c,
            "action_ix": 0,
            "regex_state": None,
        }
        actions_chosen = []
        mcts_policies = []

        _ = self.expand_node(root_node)
        if self.dirichlet_noise:
            self.add_dirichlet_noise(root_node)

        self.n_expanded_nodes = 0
        while len(actions_chosen) < self.max_path_len:
            # Spend some time expanding the tree from your current root node
            for _ in range(self.nodes_per_tree):
                node = root_node  # go back to the root node
                curr_path = actions_chosen[:]
                while len(curr_path) < self.max_path_len:
                    if len(node["childs"]) == 0:
                        node, reward, h, c = self.expand_node(node)
                        break
                    else:
                        best_child, best_action = self.estimate_q_val(node)
                        node = best_child
                        curr_path.append(best_action)

                # If finished expanding the tree get a new reward estimate for the whole path
                if len(curr_path) == self.max_path_len:
                    oh_action, h, c, action_ix, regex_state = (
                        node["oh_action"],
                        node["h_lstm"],
                        node["c_lstm"],
                        node["action_ix"],
                        node["regex_state"],
                    )
                    _, reward, _, _, _, _, _ = self.policy.forward_once(
                        oh_action, h, c, self.max_path_len - 1, regex_state
                    )
                    # reward = self.policy.evaluator.evaluate(self.policy.generator.get_regex(node["regex_state"]))
                    if isinstance(self.policy, Policy):
                        reward = float(torch.sigmoid(reward))

                # Propagate information backwards
                while node["parent"] is not None:
                    node["total_action_value"] += reward
                    node["visit_count"] += 1
                    node = node["parent"]
                # Root node is not included in the whilegit
                root_node["total_action_value"] += reward
                root_node["visit_count"] += 1

                self.n_expanded_nodes += 1

            visits_policy, qval_dist = self.sample_distributions(root_node)
            mcts_policy, action_choice = self.sample_policy(visits_policy, qval_dist, self.exploit)

            actions_chosen.append(action_choice)
            mcts_policies.append(mcts_policy)

            if action_choice == self.policy.n_actions - 1:
                return actions_chosen, mcts_policies

            root_node = root_node["childs"][action_choice]
            root_node["parent"] = None

            if self.dirichlet_noise:
                self.add_dirichlet_noise(root_node)

            print("\n")
            print("New root")
            print("Regex state: ", root_node["regex_state"][0])
            print("Regex :", self.policy.generator.get_regex(root_node["regex_state"]))
            if isinstance(self.policy, VanillaPolicy):
                self.policy.evaluator.evaluate(
                    self.policy.generator.get_regex(root_node["regex_state"]), _print=True
                )
        return (
            actions_chosen,
            mcts_policies,
            self.policy.generator.get_regex(root_node["regex_state"]),
        )
