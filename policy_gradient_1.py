from genRL import GenRL1
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Env:
    def __init__(self):
        self.maxPos = 5
        self.buyList = []
        self.done = False
        self.state = None

    def reset(self):
        self.gen = GenRL1(300)
        self.state, self.done = self.gen.next()
        return self.state

    def step(self, action):
        # 0: hold, 1: buy, 2: sell
        reward = 0
        if action == 1 and len(self.buyList) < 5:
            self.buyList.append(self.state[0][-1][0])
        elif action == 2 and len(self.buyList) >= 1:
            minBuyPrc = self.buyList[0]
            minIdx = 0
            for i in range(1, len(self.buyList)):
                if minBuyPrc > self.buyList[i]:
                    minBuyPrc = self.buyList[i]
                    minIdx = i
            del self.buyList[minIdx]
            sellPrc = self.state[0][-1][0]
            tip = sellPrc * 0.0033

            reward = (sellPrc - tip - minBuyPrc)/minBuyPrc * 100
        else:
            'holding'

        self.state, self.done = self.gen.next()

        return self.state, reward, self.done

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
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
model = Network(32, 3)
model.cuda()

def run_episode(e, env, history, model, optim):
    state = env.reset().reshape(1, -1)
    xs = torch.cuda.FloatTensor([])
    ys = torch.cuda.FloatTensor([])
    rewards = []
    reward_sum = 0

    while True:
        x = torch.cuda.FloatTensor([state.reshape(-1)])
        xs = torch.cat([xs, x])
        action_prob = model(Variable(x))

        y = F.softmax(torch.rand(1, 3).cuda())

        ys = torch.cat([ys, y.data])

        action = torch.min(y - action_prob, 1)[1].data[0]

        # print(action)
        # action = 1
        state, reward, done = env.step(action)
        reward -= 1e-7
        rewards.append(reward)
        if reward != 0:
            reward_sum += reward
        if done:
            adv = discount_rewards(rewards, 0.999)
            adv = torch.FloatTensor(adv)
            adv = ((adv - adv.mean())/(adv.std() + 1e-7)).cuda()

            loss = learn(xs, ys, adv, model, optim)
            history.append(reward_sum)
            print("[Episode {:>5}]  reward_sum: {:>5} loss: {:>5}".format(e, round(reward_sum, 4), round(loss, 4)))
            break

def learn(xs, ys, adv, model, optim):
    action_pred = model(Variable(xs))
    ys = Variable(ys, requires_grad=True)
    adv = Variable(adv.view(-1, 1))

    log_lik = -ys * torch.log(action_pred)
    log_lik_adv = log_lik * adv
    loss = torch.sum(log_lik_adv, 1).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.data[0]

optim = torch.optim.Adam(model.parameters(), 1e-4)
for e in range(10000):
    run_episode(e, env, history, model, optim)

print(history)
