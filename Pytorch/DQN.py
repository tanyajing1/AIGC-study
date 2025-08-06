import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym  # 使用 Gymnasium 替代 Gym

# 超参数
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # 最优选择动作百分比
GAMMA = 0.9                 # 奖励递减参数
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
MEMORY_CAPACITY = 2000      # 记忆库大小
env = gym.make('CartPole-v1', render_mode='human')  # 使用 v1 版本并指定 render_mode
N_ACTIONS = env.action_space.n  # 杆子能做的动作
N_STATES = env.observation_space.shape[0]   # 杆子能获取的环境信息数

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net = Net()  # 估计网络（实时更新）
        self.target_net = Net()  # 目标网络（定期更新）
        self.learn_step_counter = 0  # 学习步数计数器（用于目标网络更新）
        self.memory_counter = 0  # 经验池计数器
        # 经验池：存储(s, a, r, s')，形状为(2000, 4*2 + 2)（状态4维+动作1+奖励1+新状态4维）
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # 优化器
        self.loss_func = nn.MSELoss()  # 损失函数（均方误差）

    def choose_action(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 这里只输入一个 sample
        if np.random.uniform() < EPSILON:   # 选最优动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].item()  # 使用 item() 获取标量值
        else:   # 选随机动作
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # 针对做过的动作b_a, 来选 q_eval 的值
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()  # 定义 DQN 系统

for i_episode in range(400):
    s, _ = env.reset()  # Gymnasium 返回 (observation, info) 元组
    while True:
        a = dqn.choose_action(s)

        # 选动作, 得到环境反馈
        s_, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # 修改 reward, 使 DQN 快速学习
        x, x_dot, theta, theta_dot = s_
        r1 = (env.unwrapped.x_threshold - abs(x)) / env.unwrapped.x_threshold - 0.8
        r2 = (env.unwrapped.theta_threshold_radians - abs(theta)) / env.unwrapped.theta_threshold_radians - 0.5
        r = r1 + r2

        # 存记忆
        dqn.store_transition(s, a, r, s_)

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()  # 记忆库满了就进行学习

        if done:  # 如果回合结束, 进入下回合
            break

        s = s_

env.close()