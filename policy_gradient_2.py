from genRL import GenRL1
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Env:
    def __init__(self):
        self.maxPos = 5
        self.buyList = []
        self.done = False
        self.state = None
        self.history = []
        self.state = {}
        self.gen = GenRL1(300)
        self.getMinMax(self.gen)

    def reset(self):
        self.gen.reset()
        self.state['ts'], self.done = self.gen.next()
        self.state['st'] = np.array([0., 0.])
        self.buyList = []
        self.history = []
        self.done = False
        return self.state

    def getMinMax(self, gen):
        self.etf1max = max(gen.etf1)
        self.etf1min = min(gen.etf1)
        self.etf2max = max(gen.etf2)
        self.etf2min = min(gen.etf2)
        self.etf3max = max(gen.etf3)
        self.etf3min = min(gen.etf3)
        self.etf4max = max(gen.etf4)
        self.etf4min = min(gen.etf4)

    def normalization(self, state):
        self.state['ts'] = state
        self.state['ts'][0] = (state[0] - self.etf1min) / (self.etf1max - self.etf1min)
        self.state['ts'][1] = (state[1] - self.etf2min) / (self.etf2max - self.etf2min)
        self.state['ts'][2] = (state[2] - self.etf3min) / (self.etf3max - self.etf3min)
        self.state['ts'][3] = (state[3] - self.etf4min) / (self.etf4max - self.etf4min)

        l = len(self.buyList)
        if l > 0:
            ap = sum(self.buyList)/(len(self.buyList) - self.etf1min)/(self.etf1max-self.etf1min)
        else:
            ap = 0

        self.state['st'][0] = l/self.maxPos
        self.state['st'][1] = ap

    def step(self, action):
        # 0: hold, 1: buy, 2: sell
        reward = 0
        if action == 1 and len(self.buyList) < self.maxPos:
            self.buyList.append(self.state['ts'][0][-1][0] + 1e-7)
            # reward += 1e-4
        elif action == 2 and len(self.buyList) >= 1:
            sellPrc = self.state['ts'][0][-1][0]
            for buyPrc in self.buyList:
                tip = sellPrc * 0.0033
                # tip = 0
                reward += (sellPrc - tip - buyPrc)/buyPrc * 100
                self.buyList = []
                # print((sellPrc - tip - buyPrc)/buyPrc * 100)
            self.history.append(reward)
            self.buyList = []
            # reward = reward * 1000
        elif action == 0 and len(self.buyList) >= 1:
            avgBuyPrc = sum(self.buyList)/len(self.buyList)
            curPrc = self.state['ts'][0][-1][0]
            # reward = (avgBuyPrc - curPrc) / avgBuyPrc

        state, self.done = self.gen.next()
        if state is not None:
            self.normalization(state)
        reward -= 1e-7
        return self.state, reward, self.done

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 5)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, 5)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 16, 5)
        self.bn3 = nn.BatchNorm1d(16)
        self.l1_2 = nn.Linear(2, 64)
        self.l2 = nn.Linear(384, 64)
        self.l3 = nn.Linear(64, 3)

    def forward(self, x1, x2):
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))

        x2 = F.relu(self.l1_2(x2))
        x = F.relu(self.l2(torch.cat([x1.view(x1.size(0), -1), x2], 1)))
        x = F.softmax(self.l3(x))
        return x

def discount_rewards(r, gamma):
    running_add = 0
    discounted_r = r.copy()
    for i in reversed(range(len(r))):
        running_add = running_add * gamma + r[i]
        discounted_r[i] = running_add

    return discounted_r

''' main '''
env = Env()
history = []
model = Network()
model.cuda()

def run_episode(e, env, history, model, optim):
    state = env.reset()
    # state['ts'] = state['ts'].reshape(1, -1)
    xs1 = torch.cuda.FloatTensor([])
    xs2 = torch.cuda.FloatTensor([])
    ys = torch.cuda.FloatTensor([])
    rewards = []
    reward_sum = 0
    steps = 0

    while True:
        steps += 1
        x1 = torch.cuda.FloatTensor([[state['ts'].reshape(-1)]])
        # print(x1)
        x2 = torch.cuda.FloatTensor([state['st']])
        xs1 = torch.cat([xs1, x1])
        xs2 = torch.cat([xs2, x2])
        action_prob = model(Variable(x1), Variable(x2))

        y = F.softmax(torch.rand(1, 3).cuda())
        ys = torch.cat([ys, y.data])

        action = torch.min(y - action_prob, 1)[1].data[0]

        # print(action)
        # action = 1
        state, reward, done = env.step(action)

        rewards.append(reward)
        if reward != 0:
            reward_sum += reward
        if done:
            adv = discount_rewards(rewards, 0.999)
            adv = torch.FloatTensor(adv)
            # adv = ((adv - adv.mean())/(adv.std() + 1e-7)).cuda()
            adv = adv.cuda()

            loss = learn(xs1, xs2, ys, adv, model, optim)
            history.append(reward_sum)
            if (e+1) % 100 == 0:
                print("[Episode {:>5}]  reward_sum: {:>5} loss: {:>5}  pr: {:>5}  steps: {:>5}".format(e+1, round(reward_sum, 4), round(loss, 4), round(sum(env.history), 4), steps))
            break

def learn(xs1, xs2, ys, adv, model, optim):
    action_pred = model(Variable(xs1), Variable(xs2))
    ys = Variable(ys, requires_grad=True)
    adv = Variable(adv.view(-1, 1))

    log_lik = -ys * torch.log(action_pred)
    log_lik_adv = log_lik * adv
    loss = torch.sum(log_lik_adv, 1).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.data[0]

def draw(x):
    plt.plot(x)
    plt.show()

import _thread

optim = torch.optim.Adam(model.parameters(), 1e-4)
for e in range(100000):
    run_episode(e, env, history, model, optim)
    if e == 0:
        _thread.start_new_thread(draw, (env.gen.etf1, ))

print(history)
