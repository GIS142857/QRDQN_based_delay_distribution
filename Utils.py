import math
import random
import numpy as np
import torch

### define the replay memory


class Buffer:
    def __init__(self, buffer_size, state_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.total_size = buffer_size
        # print('state_shape', state_shape)
        # print('action_shape', action_shape)
        self.states = torch.empty(
            (self.total_size, state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size), dtype=torch.float, device=device)
        self.action_idx = torch.empty(
            (self.total_size), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, state_shape), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size), dtype=torch.float, device=device)

    def add(self, state, action, action_idx, reward, next_state,
            done):  # state, action, action_idx, reward, next_state, done
        # print('state', state)
        # print('action_state', action_state)
        # print('append', state, action)
        # print('sample', state, action, reward, done)
        self.states[self._p].copy_(state)
        self.actions[self._p] = action
        self.action_idx[self._p] = action_idx
        # print('action_set', self.actions)
        self.rewards[self._p] = reward
        self.next_states[self._p].copy_(next_state)
        self.dones[self._p] = done

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def sample(self, batch_size):
        # print('p', self._p, batch_size)
        # assert self._p % self.buffer_size == 0
        # idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        if batch_size == 0:
            return ([], [], [], [], [], [])
        #print('self_n', self._n)
        if self._n >= batch_size:
            idxes = random.sample(range(0, self._n), batch_size)
            #print('idxes', idxes)
        else:
            size = self._n
            idxes1 = random.sample(range(0, self._n), size)
            extra_size = batch_size - size
            idxes2 = np.random.randint(low=0, high=self._n, size=extra_size)
            #print('idxes', idxes1, idxes2)
            idxes1.extend(idxes2)
            idxes = idxes1
            #print('self_n', self._n, idxes1, idxes2)
            #print('idxes', idxes)
        # print('rollut_buffer sample', idxes)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.action_idx[idxes],
            self.rewards[idxes],
            self.next_states[idxes],
            self.dones[idxes]
        )


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, device):
        self._num = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.total_size = buffer_size
        self.count = {}
        self.size = {}
        self.memory = {}
        self.sub_memory = {}
        self.state_shape = state_shape
        self.device = device


    def add(self, state, action, action_idx, reward, next_state, done):  # state, action, action_idx, reward, next_state, done
        # add a routing experience
        # 一个经验回放池包含多个子buffer，用于存储在特定key(state_action)下的经验
        self._num += 1

        state_ = state.tolist()
        #next_state_ = next_state.tolist()
        key = str(state_) + '_' + str(action)
        reward = int(reward)
        key2 = str(reward)
        if key not in self.memory.keys():
            self.size[key] = 1
            self.count[key] = {key2: 1}
            self.sub_memory[key] = {key2: [state, action, action_idx, reward, next_state, done]}
            buffer = Buffer(self.buffer_size, self.state_shape, self.device)
            self.memory[key] = buffer
            self.memory[key].add(state, action, action_idx, reward, next_state, done)
            return reward
        else:
            self.size[key] += 1
            self.memory[key].add(state, action, action_idx, reward, next_state, done)
            #flag = False
            if key2 not in self.count[key].keys():
                self.count[key][key2] = 1
                self.sub_memory[key][key2] = [state, action, action_idx, reward, next_state, done]
            else:
                self.count[key][key2] += 1
            return reward



    def get_counts(self):
        counts = []
        for key in self.memory.keys():
            counts.append(self.size[key])
        return counts

    def get_sample_keys(self):
        return [key for key in self.memory.keys()]  # the number of key1

    def sample_on_key1(self, batch_size, key):
        # sample experiences according to the key

        dense_num = batch_size
        den_states, den_actions, den_action_idxs, den_rewards, den_next_states, den_dones = self.memory[key].sample(
            dense_num)
        return (den_states, den_actions, den_action_idxs, den_rewards, den_next_states, den_dones)

    def sample_on_key(self, batch_size, key):
        sparse_keys = []
        #print('self.size[key]', self.size[key])
        rand = np.random.rand()
        if rand <= 0.4 and self.size[key] > 2000:
            count = []
            delays = []
            for key2 in self.count[key]:
                delays.append(float(key2))
                count.append(int(self.count[key][key2]))
            # print('count', count)
            # print('size', self.size[key])
            pros = [c / self.size[key] for c in count]
            # print('pros', pros)
            # print('delays', delays)

            delay_pro = zip(delays, pros)
            delay_pro = list(delay_pro)
            # print('delay_pro', delay_pro)
            delay_pro_ = sorted(delay_pro, key=lambda x: x[0])
            result = zip(*delay_pro_)
            sort_delay, sort_pro = [list(x) for x in result]
            # print('sort_delays', sort_delay)
            # print('sort_pros', sort_pro)

            i = 0
            while i < 1:
                key_ = str(int(sort_delay[-1 - i]))
                if 1e-5 <= sort_pro[-1-i] <= 5e-4 and self.count[key][key_] <= 2:
                    #print('key_', key_)
                    sparse_keys.append(key_)
                i += 1


        sparse_states = []
        sparse_actions = []
        sparse_action_idxs = []
        sparse_rewards = []
        sparse_next_states = []
        sparse_dones = []
        sparse_num = 0
        if len(sparse_keys) > 0:
            t_key = sparse_keys[0]
            experience = self.sub_memory[key][t_key]
            sparse_states = torch.tensor(experience[0]).unsqueeze(0)
            sparse_actions = torch.tensor(experience[1], dtype=torch.float).unsqueeze(0)
            sparse_action_idxs = torch.tensor(experience[2], dtype=torch.float).unsqueeze(0)
            sparse_rewards = torch.tensor(experience[3], dtype=torch.float).unsqueeze(0)
            sparse_next_states = torch.tensor(experience[4]).unsqueeze(0)
            sparse_dones = torch.tensor(experience[5], dtype=torch.float).unsqueeze(0)
            # print('sparse_s1', sparse_states)
            # print('sparse_r1', sparse_rewards)
            sparse_num += 1
            for i in range(1, len(sparse_keys)):
                t_key = sparse_keys[i]
                experience = self.sub_memory[key][t_key]
                sparse_states = torch.cat((sparse_states, torch.tensor(experience[0]).unsqueeze(0)), 0)
                sparse_actions = torch.cat(
                    (sparse_actions, torch.tensor(experience[1], dtype=torch.float).unsqueeze(0)), 0)
                sparse_action_idxs = torch.cat(
                    (sparse_action_idxs, torch.tensor(experience[2], dtype=torch.float).unsqueeze(0)), 0)
                sparse_rewards = torch.cat(
                    (sparse_rewards, torch.tensor(experience[3], dtype=torch.float).unsqueeze(0)), 0)
                sparse_next_states = torch.cat((sparse_next_states, torch.tensor(experience[4]).unsqueeze(0)), 0)
                sparse_dones = torch.cat((sparse_dones, torch.tensor(experience[5], dtype=torch.float).unsqueeze(0)), 0)
                sparse_num += 1
                if sparse_num >= batch_size:
                    break
            # print('sparse_s2', sparse_states)
            # print('sparse_r2', sparse_rewards)
            # print('sparse_d2', sparse_dones)

            sparse_states = sparse_states.to(self.device)
            sparse_actions = sparse_actions.to(self.device)
            sparse_action_idxs = sparse_action_idxs.to(self.device)
            sparse_rewards = sparse_rewards.to(self.device)
            sparse_next_states = sparse_next_states.to(self.device)
            sparse_dones = sparse_dones.to(self.device)

        # print('sparse_num', sparse_num)
        dense_num = batch_size - sparse_num
        den_states, den_actions, den_action_idxs, den_rewards, den_next_states, den_dones = self.memory[key].sample(
            dense_num)
        # print('sparse_states', sparse_states)
        # print('den_states', den_states)
        # print('den_rewards', den_rewards)
        # print('den_actions', den_actions)
        if sparse_num == 0:
            return (den_states, den_actions, den_action_idxs, den_rewards, den_next_states, den_dones)
        else:
            if dense_num > 0:
                states = torch.cat((sparse_states, den_states), dim=0)
                # print('states', states)
                actions = torch.cat((sparse_actions, den_actions), dim=0)
                # print('actions', actions)
                action_idxs = torch.cat((sparse_action_idxs, den_action_idxs), dim=0)
                rewards = torch.cat((sparse_rewards, den_rewards), dim=0)
                # print('rewards', rewards)
                next_states = torch.cat((sparse_next_states, den_next_states), dim=0)
                dones = torch.cat((sparse_dones, den_dones), dim=0)
                # print('dones', dones)
                return (states, actions, action_idxs, rewards, next_states, dones)
            else:
                return (
                sparse_states, sparse_actions, sparse_action_idxs, sparse_rewards, sparse_next_states, sparse_dones)