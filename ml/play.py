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
        self.max_bricks = 50
        self.state_shape = (self.k, 11 + 4 * self.max_bricks)
        self.n_actions = len(self.action_space)
        self.i_episode = 1
        self.total_reward = 0
        self.loss = 0
        self.k_state = []
        self.k_reward = 0
        self.action = 0
        self.prev_n_bricks = 0
        self.ball_prev_y = 395

        self.agent = DeepQNet(self.state_shape, self.n_actions, QNet, self.k, device='cuda')
        path = os.path.dirname(os.path.abspath(__file__))
        # path = os.path.join(path, 'best_model_bp.pt')
        path = os.path.join(path, 'saves', '18000.pt')
        self.agent.eval_net.load_state_dict(torch.load(path))
        self.agent.eval_net.eval()

    def update(self, scene_info, *args, **kwargs):
        state = self.convert_state(scene_info)
        self.k_state.append(state[np.newaxis, ...])

        if scene_info["frame"] == 0:
            reward = self.get_reward(scene_info, 0)
        else:
            reward = self.get_reward(scene_info, self.prev_n_bricks - state[-1])

        self.prev_n_bricks = state[-1]
        self.k_reward += reward
        done = (scene_info["status"] != "GAME_ALIVE")

        if done:
            for i in range(self.k - len(self.k_state)):
                self.k_state.append(self.k_state[-1])

        if len(self.k_state) == self.k:
            stacked_state = np.concatenate(self.k_state, 0)

            self.total_reward += self.k_reward
            self.k_state = []
            self.k_reward = 0

            self.action = self.agent.choose_action(stacked_state, 0, scene_info["ball_served"])

        return self.action_space[self.action]

    def reset(self):
        print("Episode: {}, Reward: {}, Loss: {}, Epsilon: {}".format(
            self.i_episode, self.total_reward, self.loss, self.eps)
        )

    def get_reward(self, scene_info, bricks_hit):
        # print(scene_info["ball"])
        reward = 0
        if scene_info["status"] == "GAME_OVER":
            reward -= 5
        if scene_info["status"] == "GAME_PASS":
            reward += 5

        reward += bricks_hit * 2
        if 385 < scene_info["ball"][1] < self.ball_prev_y:
            reward += 1
        self.ball_prev_y = scene_info["ball"][1]
        return reward

    def convert_state(self, scene_info):
        bricks = scene_info["bricks"]
        n_bricks = len(bricks)
        for i in range(self.max_bricks - len(bricks)):
            bricks.append([-1, -1])
        bricks_pos = []
        for brick in bricks:
            bricks_pos.append(brick[0])
            bricks_pos.append(brick[1])
            if brick[0] == -1:
                bricks_pos.append(-1)
                bricks_pos.append(-1)
            else:
                bricks_pos.append(brick[0] + 25)
                bricks_pos.append(brick[1] + 10)

        state = [
            scene_info["frame"],
            *scene_info["ball"],
            scene_info["ball"][0] + 5,
            scene_info["ball"][1] + 5,
            scene_info["ball_served"],
            *scene_info["platform"],
            scene_info["platform"][0] + 40,
            scene_info["platform"][1] + 5,
            *bricks_pos,
            n_bricks,
        ]
        return np.array(state)


if __name__ == '__main__':
    pass
