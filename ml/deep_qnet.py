import torch
import torch.nn as nn
import torch.optim as optim
from ml.experience_buffer import *


class DeepQNet:
    def __init__(
            self,
            input_shape,
            n_actions,
            qnet,
            k=4,
            learning_rate=2e-4,
            reward_decay=0.9,
            replace_target_iter=1000,
            memory_size=10000,
            batch_size=32,
            device='cpu',
    ):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.qnet = qnet
        self.k = k
        self.lr = learning_rate
        self.device = device
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.buffer = ExperienceBuffer(memory_size)
        self.learn_step_counter = 0

        self.target_net = qnet(input_shape, n_actions).to(self.device)
        self.eval_net = qnet(input_shape, n_actions).to(self.device)
        self.target_net.eval()
        self.optimizer = optim.RMSprop(params=self.eval_net.parameters(), lr=self.lr)

    def store_transition(self, s, a, r, done, s_):
        exp = Experience(s, a, r, done, s_)
        self.buffer.append(exp)
        return

    def choose_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            state_v = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # print(state_v.shape)
            q_values = self.eval_net(state_v).cpu()
            return int(q_values.argmax(1)[0])

    def learn(self):
        if len(self.buffer) >= self.batch_size:
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.optimizer.zero_grad()
            loss = self.calc_loss()
            loss.backward()
            self.optimizer.step()
            self.learn_step_counter += 1
            return loss.item()
        return 0

    def calc_loss(self):
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)
        states_v = torch.FloatTensor(states).to(self.device)
        actions_v = torch.LongTensor(actions).to(self.device)
        rewards_v = torch.FloatTensor(rewards).to(self.device)
        dones_v = torch.BoolTensor(dones).to(self.device)
        next_states_v = torch.FloatTensor(next_states).to(self.device)

        # print(self.eval_net(states_v).shape)
        # print(actions_v.shape)
        pred_q = self.eval_net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            target_q = self.target_net(next_states_v).max(1)[0]
            target_q[dones_v] = 0.0
            target_q = target_q.detach()

        expected_q = rewards_v + self.gamma * target_q
        # print(pred_q)
        # print(expected_q)
        return nn.MSELoss()(pred_q, expected_q)

    def save_model(self, fname):
        torch.save(self.eval_net.state_dict(), fname)

