import torch
import torch.nn.functional as F
import numpy as np


class Net(torch.nn.Module):
    '''
        FC network

    '''

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        self.out = torch.nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        del x
        return action_value


class DQN(object):

    def __init__(self, ENV, N_STATES, N_ACTIONS, MEMORY_CAPACITY, LR,
                 EPSILON, BATCH_SIZE, GAMMA,TARGET_REPLACE_ITER, mode='r'):
        '''
        Initialization

        :param N_STATES: dimensions of states
        :param N_ACTIONS: number of actions
        :param MEMORY_CAPACITY:
        :param LR:
        :param EPSILON:
        :param BATCH_SIZE:
        :param GAMMA:
        :param TARGET_REPLACE_ITER:
        '''
        self.N_STATES = N_STATES
        self.eval_net, self.target_net = Net(self.N_STATES, N_ACTIONS), Net(self.N_STATES, N_ACTIONS) # eval_net,

        self.env = ENV

        self.mode = mode

        self.N_ACTIONS = N_ACTIONS # number of actions
        self.MEMORY_CAPACITY = MEMORY_CAPACITY # memory size for experience replay
        self.EPSILON = EPSILON # epsilon greedy
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA # discount factor for TD error
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER # ???

        self.learn_step_counter = 0 # ???
        self.memory_counter = 0 # ???
        self.memory = np.zeros((MEMORY_CAPACITY, self.N_STATES * 2 + 2)) # initialization of memory memmory size * data sample size (s_t, a_t, r_t, s_{t+1})
        self.optimizer = torch.optim.Adam([
            {"params": self.eval_net.parameters()},
            {"params": self.env.W_T_1},
            {"params": self.env.W_T_2},
            {"params": self.env.W_p},
            {"params": self.env.W_p_},
            {"params": self.env.W_u},
            {"params": self.env.b_T}], lr=LR)
        self.loss_func = torch.nn.MSELoss()



    def choose_action(self, x):
        '''
        \epsilon greedy for generating actions
        :param x:
        :return: action
        '''

        if np.random.uniform() < self.EPSILON:
            with torch.no_grad():
                x = self.ensemble_state(x[0], x[1])
                action_value = self.eval_net.forward(x)
                action_value = action_value.numpy()
                action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.N_ACTIONS)

        return action


    def store_transition(self, s, a, r, s_):
        '''
        ???????
        :param s:
        :param a:
        :param r:
        :param s_:
        :return:
        '''
        s = self.ensemble_state(s[0], s[1])
        s_ = self.ensemble_state(s_[0], s_[1])
        index = self.memory_counter % self.MEMORY_CAPACITY  # If full, restart from the beginning
        self.memory[index, :] = np.hstack((s.detach().numpy(), np.array([[a, r]]), s_.detach().numpy()))
        self.memory_counter += 1



    def TD(self, memory):
        b_a = torch.LongTensor(memory[:, self.N_STATES:self.N_STATES + 1])
        b_s = torch.FloatTensor(memory[:, :self.N_STATES])
        b_r = torch.FloatTensor(memory[:, self.N_STATES + 1:self.N_STATES + 2])
        b_s_ = torch.FloatTensor(memory[:, -self.N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(-1, 1)

        return q_target, q_eval



    def priority(self, mode='r'):
        '''

        :param b_memory: memory for experience replay
        :param mode: mode='r', select reward-based, mode='TD'; select TD-based
        :return: priority score
        '''

        if mode == 'r':
            b_a = torch.tensor(self.memory[:, self.N_STATES:self.N_STATES + 1])
            p_score = b_a
            return p_score.view(-1)
        else:
            q_target, q_eval = self.TD(self.memory)
            p_score = q_target - q_eval
        return p_score.view(-1), q_target, q_eval

    def prob(self, x):
        return F.softmax(x,dim=0)



    def pooling(self, KG):
        '''
        hierarchical pooling for KG state
        :param KG: torch_geometric.data.Data, KG state
        :return: torch.tensor, N*1, one vector for KG
        '''
        entities_cat = KG.x[:KG.num_POI+KG.num_cat]
        entities_loc = torch.cat((KG.x[:KG.num_POI], KG.x[KG.num_POI + KG.num_cat:]), dim=0)

        s_KG_cat = entities_cat.mean(dim=0)
        s_KG_loc = entities_loc.mean(dim=0)

        s_KG = (s_KG_cat + s_KG_loc) / 2

        return s_KG

    def ensemble_state(self, s_u, KG):
        s_KG = self.pooling(KG)
        return torch.cat((s_u.view(1, -1), s_KG.view(1, -1)), dim=1)

    def learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        if self.mode == 'r':
            p_score = self.priority(self.mode)
            prob = self.prob(p_score)
            sample_index = np.random.choice(a=self.MEMORY_CAPACITY, size=self.BATCH_SIZE, p=prob)
            b_memory = self.memory[sample_index, :]
            q_target, q_eval = self.TD(b_memory)
        else:
            p_score, q_target, q_eval = self.priority(self.mode)
            prob = self.prob(p_score)
            sample_index = np.random.choice(a=self.MEMORY_CAPACITY, size=self.BATCH_SIZE, p=prob.detach().numpy())
            q_target = q_target[sample_index]
            q_eval = q_eval[sample_index]

        self.loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()