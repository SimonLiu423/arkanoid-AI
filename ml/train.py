import torch
from ml.qnet import QNet
from ml.deep_qnet import *
from torch.utils.tensorboard import SummaryWriter
import os


class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.action_space = [
            "NONE",
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "SERVE_TO_LEFT",
            "SERVE_TO_RIGHT",
        ]
        self.k = 4
        self.max_bricks = 100
        self.state_shape = (9 + 4 * self.max_bricks + 1, )
        self.n_actions = len(self.action_space)
        self.agent = DeepQNet(self.state_shape, self.n_actions, QNet, self.k, device='cuda', learning_rate=1e-3)
        # path = os.path.dirname(os.path.abspath(__file__))
        # path = os.path.join(path, 'best_model_bp.pt')
        # self.agent.eval_net.load_state_dict(torch.load(path))
        # self.agent.target_net.load_state_dict(torch.load(path))
        self.ball_default_pos = (93, 395)
        self.eps = 1
        self.eps_decay = 0.995
        self.i_episode = 1
        self.total_reward = 0
        self.loss = 0
        self.prev_state = None
        self.prev_n_bricks = 0
        self.prev_n_hard_bricks = 0
        self.ball_prev_pos = self.ball_default_pos
        self.ball_velocity = [0, 0]
        self.writer = SummaryWriter()
        self.best_reward = -100
        self.ball_served = False
        self.step = 0

    def update(self, scene_info, *args, **kwargs):
        state = self.convert_state(scene_info)

        if scene_info["frame"] == 0:
            reward = self.get_reward(scene_info, 0, 0)
        else:
            reward = self.get_reward(scene_info, state[-2], state[-1])

        self.prev_n_bricks = state[-2]
        self.prev_n_hard_bricks = state[-1]
        done = (scene_info["status"] != "GAME_ALIVE")

        action = self.agent.choose_action(state, self.eps)

        if self.prev_state is not None:
            self.agent.store_transition(self.prev_state, action, reward, done, state)

        self.loss = self.agent.learn()

        self.prev_state = state
        self.total_reward += reward
        self.step += 1

        self.ball_prev_pos = scene_info["ball"]

        return self.action_space[action]

    def reset(self):
        print("Episode: {}, Reward: {}, Loss: {}, Epsilon: {}".format(
            self.i_episode, self.total_reward, self.loss, self.eps)
        )
        self.writer.add_scalar("Loss/episode", self.loss, self.i_episode)
        self.writer.add_scalar("Reward/episode", self.total_reward, self.i_episode)
        self.writer.add_scalar("Epsilon/episode", self.eps, self.i_episode)
        self.writer.flush()

        if self.total_reward > self.best_reward:
            path = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(path, 'best_model.pt')
            self.agent.save_model(path)
            self.best_reward = self.total_reward

        if self.i_episode % 50 == 0:
            path = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(path, 'saves')
            if not os.path.exists(path):
                os.mkdir(path)
            path = os.path.join(path, '{}.pt'.format(self.i_episode))
            self.agent.save_model(path)

        self.i_episode += 1
        self.prev_state = None
        self.prev_n_bricks = 0
        self.prev_n_hard_bricks = 0
        self.total_reward = 0
        self.eps = self.epsilon_compute(epsilon_decay=5000000)
        self.ball_prev_pos = self.ball_default_pos
        self.ball_served = False
        self.ball_velocity = [0, 0]

    def get_reward(self, scene_info, n_bricks, n_hard_bricks):
        # print(scene_info["ball"])
        reward = 0
        if scene_info["status"] == "GAME_OVER":
            reward -= abs((scene_info["platform"][0] + 20) - (scene_info["ball"][0] + 2.5)) / 170 * 5
        if scene_info["status"] == "GAME_PASS":
            reward += 5

        delta_hard = self.prev_n_hard_bricks - n_hard_bricks
        reward += delta_hard * 0.1
        reward += (self.prev_n_bricks + delta_hard - n_bricks) * 0.1
        if 385 < scene_info["ball"][1] < self.ball_prev_pos[1]:
            # print("Hit!")
            reward += 2
        return reward

    def convert_state(self, scene_info):
        bricks = scene_info["bricks"]
        hard_bricks = scene_info["hard_bricks"]
        n_bricks = len(bricks)
        n_hard_bricks = len(hard_bricks)
        for i in range(self.max_bricks - n_bricks):
            bricks.append([-1, -1])
        for i in range(self.max_bricks - n_hard_bricks):
            hard_bricks.append([-1, -1])

        bricks_pos = []
        hard_bricks_pos = []

        for brick in bricks:
            bricks_pos.append(brick[0])
            bricks_pos.append(brick[1])
            # if brick[0] == -1:
            #     bricks_pos.append(-1)
            #     bricks_pos.append(-1)
            # else:
            #     bricks_pos.append(brick[0] + 25)
            #     bricks_pos.append(brick[1] + 10)
                
        for hard_brick in hard_bricks:
            hard_bricks_pos.append(hard_brick[0])
            hard_bricks_pos.append(hard_brick[1])
            # if hard_brick[0] == -1:
            #     hard_bricks_pos.append(-1)
            #     hard_bricks_pos.append(-1)
            # else:
            #     hard_bricks_pos.append(hard_brick[0] + 25)
            #     hard_bricks_pos.append(hard_brick[1] + 10)

        # Get ball velocity
        ball_x, ball_y = scene_info["ball"]
        if not self.ball_served:
            if ball_y != self.ball_default_pos[1]:
                self.ball_served = True
        if self.ball_served:
            self.ball_velocity = [ball_x - self.ball_prev_pos[0], ball_y - self.ball_prev_pos[1]]

        state = [
            scene_info["frame"],
            *scene_info["ball"],
            scene_info["ball_served"],
            self.ball_velocity[0],
            self.ball_velocity[1],
            *scene_info["platform"],
            *bricks_pos,
            *hard_bricks_pos,
            n_bricks,
            n_hard_bricks,
        ]
        # print("Ball ({}, {}), V ({}, {})".format(ball_x, ball_y, *self.ball_velocity))
        return np.array(state)

    def epsilon_compute(self, epsilon_max=1.0, epsilon_min=0.02, epsilon_decay=1000):
        return max(epsilon_min, epsilon_max - self.step / epsilon_decay)


if __name__ == '__main__':
    pass
