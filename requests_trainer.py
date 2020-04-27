import re
from regex import RegexGenerator
from policy import Policy
from evaluator import RequestEvaluator
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

if __name__ == "__main__":
    num_batch = 1000
    batch_size = 64
    max_len_regex = 10

    evaluator = RequestEvaluator("http_data/goodqueries.txt", "http_data/badqueries.txt")
    gen = RegexGenerator(128)
    policy = Policy(gen.action_space, 20, gen, hidden_size=256)

    policy_optimizer = optim.Adam(filter(lambda x: x.requires_grad, policy.parameters()), lr=0.01)

    for i in range(num_batch):
        policy_optimizer.zero_grad()
        actor_batch = []
        critic_batch = []
        reward_batch = []
        regex = []
        for j in range(batch_size):
            actions, reg = policy.sample_regex(max_len_regex)
            reward_batch.append(evaluator.evaluate(reg))
            actor_outs, critic_outs, _, _ = policy.evaluate_solution(actions)
            actor_batch.append(actor_outs)
            critic_batch.append(critic_outs)
            regex.append(reg)

        actor_batch = pad_sequence(actor_batch, True)
        critic_batch = pad_sequence(critic_batch, True)
        reward_batch = (
            torch.FloatTensor(reward_batch)
            if not torch.cuda.is_available()
            else torch.FloatTensor(reward_batch).cuda()
        )
        reward_batch = reward_batch.unsqueeze(1)

        print("max_reward :", reward_batch.max())
        print("regex max :", regex[int(reward_batch.argmax())])
        loss = -1.0 * (actor_batch * reward_batch).sum()
        print("loss :", loss)
        loss.backward()
        policy_optimizer.step()
