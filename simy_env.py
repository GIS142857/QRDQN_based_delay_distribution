import copy
import json
from collections import deque
import bisect
import inspect
import random
import simpy
from matplotlib import pyplot as plt
from simpy.util import start_delayed
from agent import Agent
from config import *
from tensorboardX import SummaryWriter
from Utils import *

BROADCAST_ADDR = 0xFFFF

writer1 = SummaryWriter(log_dir='log999_11')
writer2 = SummaryWriter(log_dir='log999_22')
writer3 = SummaryWriter(log_dir='log999_33')

def ensure_generator(env,func,*args,**kwargs):
    '''
    Make sure that func is a generator function.  If it is not, return a
    generator wrapper
    '''
    if inspect.isgeneratorfunction(func):
        return func(*args,**kwargs)
    else:
        def _wrapper():
            func(*args,**kwargs)
            yield env.timeout(0)
        return _wrapper()

###########################################################
def distance(pos1,pos2):
    '''
        calculate the distance between two nodes
    '''
    return ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)**0.5


###########################################################
class PDU:
    def __init__(self, layer, nbits, **fields):
        self.layer = layer
        self.nbits = nbits
        for f in fields:
            setattr(self, f, fields[f])

###########################################################
class DefaultPhyLayer:
    '''
        Setting up the physical layer
    '''

    LAYER_NAME = 'phy'

    def __init__(self, node, bitrate=4e6):
        self.node = node
        self.bitrate = bitrate
        self._current_rx_count = 0
        self._channel_busy_start = 0
        self.total_tx = 0
        self.total_rx = 0
        self.total_collision = 0
        self.total_error = 0
        self.total_bits_tx = 0
        self.total_bits_rx = 0
        self.total_channel_busy = 0
        self.total_channel_tx = 0

    def send_pdu(self, pdu):
        '''
            transmit a pdu
        '''

        tx_time = pdu.nbits/self.bitrate
        next_node = self.node.sim.nodes[pdu.dst]
        self.node.delayed_exec(
            1e-8, next_node.phy.on_rx_start, pdu)
        self.node.delayed_exec(
            1e-8 + tx_time, next_node.phy.on_rx_end, pdu)

    def on_tx_start(self, pdu):
        pass

    def on_tx_end(self, pdu):
        pass

    def on_rx_start(self, pdu):
        pass


    def on_rx_end(self, pdu):
        '''
            receive a pdu
        '''
        source = pdu.src
        src_node = self.node.sim.nodes[source]
        per = 0.2
        if self.node.sim.random.random() > per:
            # print('phy transmit sucesslly to', pdu.dst)
            self.node.mac.on_receive_pdu(pdu)
            self.total_rx += 1
            self.total_bits_rx += pdu.nbits
        else:
            self.total_error += 1


    def on_collision(self, pdu):
        pass

    def cca(self):
        """Return True if the channel is clear"""
        return self._current_rx_count == 0


###########################################################
class DefaultMacLayer:
    '''
        Setting up the MAC layer
    '''

    LAYER_NAME = 'mac'
    HEADER_BITS = 64

    def __init__(self, node):
        self.node = node
        self.queues = []
        self.flow1_queue = []
        self.flow2_queue = []
        self.flow3_queue = []
        self.ack_event = None
        #self.stat = Stat()
        self.total_tx_broadcast = 0
        self.total_tx_unicast = 0
        self.total_rx_broadcast = 0
        self.total_rx_unicast = 0
        self.total_retransmit = 0
        self.total_ack = 0
        self.backoff = 0 #random.randint(0, parameters.CW_MIN)
        self.retries = 0
        #self.window = parameters.CW_MIN
        self.has_reces = []
        #self.header_bits = 64

    def addPacket(self, packet, pri):
        #print('node_bf', self.node.id, 'adds packets', len(self.queues))
        queue_len = 0
        if len(self.queues) <= 1:
            packet.payload.queue_of_next_node[str(self.node.id)] = queue_len
            self.queues.append(packet)
            return

        temp = [self.queues[0]]
        for i in range(1, len(self.queues)):
            if pri < self.queues[i].payload.priority:
                packet.payload.queue_of_next_node[str(self.node.id)] = queue_len
                temp.append(packet)
                for j in range(i, len(self.queues)):
                    temp.append(self.queues[j])
                self.queues = temp
                #print('node_af', self.node.id, 'adds packets', len(self.queues))
                return
            else:
                queue_len += 1
                temp.append(self.queues[i])
        packet.payload.queue_of_next_node[str(self.node.id)] = queue_len
        self.queues.append(packet)
        #print('node_af', self.node.id, 'adds packets', len(self.queues))

    def process_queue(self):
        '''
            Send the packets in the queue in sequence
        '''
        while len(self.queues) > 0:

            send_slot = self.node.sim.frame_slot[self.node.id]
            wait_time = (self.node.sim.env.now - self.node.start_time) % (0.001 * self.node.sim.each_slot * send_slot)
            interval = 0.001 * self.node.sim.each_slot * send_slot - wait_time

            #print('slot', send_slot)

            yield self.node.sim.env.timeout(interval)

            if len(self.queues) == 0:
                return

            packet = self.queues[0]
            self.node.phy.send_pdu(packet)
            yield self.node.sim.env.timeout(0.001 * self.node.sim.each_slot)



    def send_pdu(self, packet, dst):
        '''
            Send a packet to the next node
        '''
        key = str(self.node.id)
        packet.arrival_time[key] = self.node.now
        mac_pdu = PDU(self.LAYER_NAME, packet.nbits + self.HEADER_BITS,
                      type='data',
                      src=self.node.id,
                      src_node=self.node,
                      dst=dst,
                      dst_node=self.node.sim.nodes[dst],
                      payload=packet)
        #print('node mac send_pu', self.node.id)
        self.addPacket(mac_pdu, packet.priority)
        if len(self.queues) == 1:
            self.node.sim.env.process(self.process_queue())


    def on_receive_pdu(self, pdu):
        '''
            Receive a packet from the last node
        '''
        # print(pdu.dst, self.node.id, pdu.src)
        if pdu.dst == self.node.id:
            if pdu.payload.id in self.has_reces:
                return
            self.has_reces.append(pdu.payload.id)
            # print('node mac', self.node.id, 'receive packet', pdu.payload.id)
            # TODO: need to get rid of duplications
            #print('node_on_receive', self.node.id, pdu.payload.id, 'now', self.node.sim.env.now)
            self.node.on_receive_pdu(pdu.payload)

            last_node = pdu.src
            #print('last_node', last_node)
            if len(self.node.sim.nodes[last_node].mac.queues) > 0:
                self.node.sim.nodes[last_node].mac.queues.pop(0)


class Packet:
    def __init__(self, sim, source, id, priority, nbits=8*1000):
        self.source = source
        self.id = id
        self.priority = priority
        self.nbits = nbits
        self.start_time = sim.env.now
        self.node_arrival = dict()
        self.one_hop_delay = dict()
        self.travase_delay = {}
        self.arrival_time = {}
        self.queue_of_next_node = {}
        self.end_to_end_delay = 0
        self.travese_path = [source]
        self.des = FLOW_DICT[source]



###########################################################
class Node(object):
    '''
        set the parameters of a node, including its physical layer, mac layer, simulator,,,
    '''

    DEFAULT_MSG_NBITS = 1000*8

    def __init__(self, sim, id, src, dst, arrival_rates, pos, device):
        self.sim = sim
        self.id = id
        self.isSource = False
        if self.id in src:
            self.isSource = True
            self.packet_pri = src.index(self.id)
            self.arrival_rate = arrival_rates[FLOW_MAP[src.index(self.id)]]
        self.pos = pos
        self.phy = DefaultPhyLayer(self)
        self.mac = DefaultMacLayer(self)
        self.neighbors = ADJ_TABLE[self.id]
        self.sends = 0
        self.reces_for_me = []
        self.has_reces = []
        self.timeout = self.sim.timeout
        self.start_time = 0
        self.device = device
        self.loss_history = []
        self.loss_num1 = 0
        self.loss_num2 = 0
        self.path_num = 0
        self.e2ed = []
        self.episode = 0
        self.sum_sources = len(SRC)
        self.sum_nodes = SUM_NODES
        self.state_dim = self.sum_nodes + self.sum_sources
        self.end_to_end_delay = {}


    @property
    def now(self):
        return self.sim.env.now

    def setAgent(self):
        '''
            Set an agent for the node to learn the delay distribution
        '''
        self.action_dim = 1
        self.agent = Agent(self.state_dim, self.action_dim, self.neighbors, self.id, self.device, 500)

    def create_event(self):
        return self.sim.env.event()

    def delayed_exec(self, delay, func, *args, **kwargs):
        return self.sim.delayed_exec(delay, func, *args, **kwargs)

    ############################
    def set_layers(self, phy=None, mac=None):
        if phy is not None:
            self.phy = phy(self)
        if mac is not None:
            self.mac = mac(self)

    def get_nextnode(self, packet):
        '''
            select the next node
        '''
        for nb in self.neighbors:
            if nb == packet.des:
                return nb
        #print('len(self.neighbors)', len(self.neighbors), self.neighbors)
        # print(packet.des, self.id, self.neighbors)
        # print(self.id, packet.des, self.dfs_find_all_paths(ADJ_TABLE, self.id, packet.des))

        next_node = random.choices(self.dfs_find_all_paths(ADJ_TABLE, self.id, packet.des))[0]

        return next_node

    def dfs_find_all_paths(self, adj_table, start, target, path=None, all_paths=None):
        if path is None:
            path = []  # 当前路径
        if all_paths is None:
            all_paths = []  # 所有路径

        # 将当前节点添加到路径
        path.append(start)

        # 如果到达目标节点，保存当前路径
        if start == target:
            if path[1] not in all_paths:
                all_paths.append(path[1])
        else:
            # 遍历当前节点的所有邻居
            for neighbor in adj_table.get(start, []):
                if neighbor not in path:  # 防止循环
                    self.dfs_find_all_paths(adj_table, neighbor, target, path, all_paths)

        # 回溯，移除当前节点
        path.pop()

        return all_paths


    def get_best_action(self, packet):
        # calculate worst delay of each path
        # print('neighbor', neighbor)
        txtpath = 'E2ED_' + str(self.id) + '.txt'
        with open(txtpath, 'r') as fl:
            js = fl.read()
            dis_set = json.loads(js)

        worstD_list = []
        aveD_list = []
        cand_neighbors = []
        # print('dis_set', dis_set)
        # print('keys', dis_set[str(packet.priority)].keys())
        path_set = []
        for key in dis_set[str(packet.priority)].keys():
            arry = key.split('-')
            path_ = arry[0]
            temp = path_[1:len(path_) - 1]
            # print('temp', temp, len(temp))
            path = []
            for l in range(len(temp)):
                # print('l', l, temp[l])
                if l % 3 == 0:
                    path.append(int(temp[l]))
            path_set.append(path)
        # print('paths', path_set)
        for path in path_set:
            # path_list.append(path)
            state_flow = np.zeros(len(self.sim.sources))
            # print(state_flow, packet.priority)
            state_flow[packet.priority] = 1
            # print('state_flow', state_flow)
            state_node = np.zeros(self.sim.N)
            neighbor = path[1]
            # print('i', i, 'sub_traverse_nodes', traverse_nodes[i:])
            # print('pri', packet.priority)
            for j in path:
                state_node[j] = 1
            # print('state_node', state_node)
            state = np.concatenate((state_flow, state_node))
            state = torch.tensor(state, dtype=torch.float32)
            qu = self.sim.graph.node_list[neighbor].agent.model.forward(1, state)
            qu = qu.reshape(1, -1).squeeze().detach()
            # print('qu', qu)
            # print('worst_delay', qu[-1])
            worst_delay = qu[-1]
            l = int(len(qu) / 2)
            average_delay = qu.mean()
            # print('aved', average_delay)
            if worst_delay <= packet.deadline:
                cand_neighbors.append(neighbor)
                aveD_list.append(average_delay)

        if len(aveD_list) > 0:
            idx = aveD_list.index(min(aveD_list))
            next_node = cand_neighbors[idx]
            return next_node
        # else:
        #     return None


    def run(self):
        duration = 1
        while self.now < self.start_time + duration:
            if self.id == 0:
                lamda = 1.0 / self.arrival_rate
                interval = random.expovariate(lamda)
            elif self.id == 1:
                rand = np.random.random()
                seed = 1
                if rand < 0.8:
                    seed = 2
                # print('source1', step, self.arrival_rate[source], step % self.arrival_rate[source])
                if seed == 2:
                    lamda = 1.0 / self.sim.arrival_rates[FLOW_MAP[self.id]]
                    #print('lameda', lameda)
                    interval = random.expovariate(lamda)
                    #print('interval', interval)
                else:
                    lamda = 1.0 / (self.sim.arrival_rates[FLOW_MAP[self.id]]/2)
                    # print('lameda', lameda)
                    interval = random.expovariate(lamda)
            else:
                interval = self.sim.arrival_rates[FLOW_MAP[self.id]]

                # print('interval', interval, random.expovariate(lameda))
            yield self.sim.env.timeout(interval)
            # NB: inter-message time start after mac has served previous message, to make sure that mac does not handle multiple messages concurrently
            self.sends += 1
            packet_id = str(self.id) + '_' + str(self.sends)
            packet = Packet(self.sim, self.id, packet_id, self.packet_pri)
            next_node = self.get_nextnode(packet)
            self.mac.send_pdu(packet, next_node)



    def addsamples(self, packet, isuUdate):
        '''
            add routing experience (path, delay)
        '''
        #print('addsamples')
        traverse_nodes = packet.travese_path  # the path of the packet
        rewards = [] ### calculate the cumulative rewards
        for i in range(len(traverse_nodes) - 1):
            node = traverse_nodes[i]
            next_node = traverse_nodes[i + 1]
            key = str(node) + '_' + str(next_node)
            reward = packet.travase_delay[key]
            rewards.append(reward)

        for i in range(len(traverse_nodes) - 1):
            node = traverse_nodes[i]
            next_node = traverse_nodes[i + 1]
            # print('node', node, 'next_node', next_node)
            state_flow = np.zeros(self.sum_sources)
            # print(state_flow, packet.priority)
            state_flow[packet.priority] = 1
            state_node = np.zeros(self.sum_nodes)
            # print('i', i, 'sub_traverse_nodes', traverse_nodes[i:])
            # print('pri', packet.priority)
            for j in traverse_nodes[i:]:
                state_node[j] = 1

            state = np.concatenate((state_flow, state_node))
            # print('state', state)
            action = next_node
            action_idx = 0
            next_state_node = np.zeros(self.sum_nodes)
            for j in traverse_nodes[i + 1:]:
                next_state_node[j] = 1

            next_state = np.concatenate((state_flow, next_state_node))
            # print('next_state', next_state)
            done = 1
            l = len(rewards)
            reward = np.sum(rewards[i:l])

            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            self.sim.nodes[node].agent.replay_memory.add(state, action, action_idx, reward, next_state, done)

        if self.episode > 500 and isuUdate:
            temp = copy.deepcopy(traverse_nodes)
            temp.reverse()
            loss = self.train(temp)

            self.path_num += 1
            writer2.add_scalar('loss', loss, self.path_num)

            print('update_loss', loss)


    def train(self, nodes):
        '''
            train the qrdqn of each node
        '''
        sumloss = 0
        #print('nodes', nodes)
        for i in range(1, len(nodes)):  # nodes_list
            node = nodes[i]
            # How much of the replay-memory should be used.
            # print('node train', node)
            replay_memory = self.sim.nodes[node].agent.replay_memory
            # When the replay-memory is sufficiently full.
            #print('replay_memory_num_used', replay_memory._num)
            # print('node_train', node, 'memory_len', replay_memory.num_used)
            if replay_memory._num >= 1:  # batch_size:
                loss = self.optimize_adam(node)
                #self.sim.nodes[node].agent.save_model('E2E')
                sumloss += loss
        sumloss = sumloss / (len(nodes) - 1)
        return sumloss

    def calculate_huber_loss(self, td_errors, kappa=1.0):
        return torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors.pow(2),
            kappa * (td_errors.abs() - 0.5 * kappa))

    def calculate_quantile_huber_loss(self, td_errors, taus, kappa=1.0, weights=None):
        assert not taus.requires_grad
        batch_size, N, N_dash = td_errors.shape

        # Calculate huber loss element-wisely.
        element_wise_huber_loss = self.calculate_huber_loss(td_errors, kappa)
        assert element_wise_huber_loss.shape == (
            batch_size, N, N_dash)

        element_wise_quantile_huber_loss = torch.abs(
            taus[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
        assert element_wise_quantile_huber_loss.shape == (
            batch_size, N, N_dash)
        # print('element_wise_quantile_huber_loss', element_wise_quantile_huber_loss)

        # Quantile huber loss.
        batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(
            dim=1).mean(dim=1, keepdim=True)
        # print('batch_quantile_huber_loss', batch_quantile_huber_loss)
        assert batch_quantile_huber_loss.shape == (batch_size, 1)

        if weights is not None:
            quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
        else:
            quantile_huber_loss = batch_quantile_huber_loss.mean()

        return quantile_huber_loss


    def calculate_loss(self, batch_size, states, actions, action_idxs, rewards, next_states, dones, node):  ### batch_size, states, actions, rewards, next_states, dones
        # Calculate quantile values of current states and actions at taus.

        var_s = torch.tensor(states, dtype=torch.float)
        var_s = var_s.to(self.device)

        # print('node', node)
        #print('var_s', var_s)

        current_sa_quantiles = self.sim.nodes[node].agent.model.forward(batch_size, var_s)

        assert current_sa_quantiles.shape == (batch_size, self.sim.nodes[node].agent.tau_N, 1)

        with torch.no_grad():

            var_next_s = torch.tensor(next_states[0], dtype=torch.float)
            var_next_s = var_next_s.to(self.device)
            rewards = torch.tensor(rewards).to(self.device)

            next_node = int(actions[0])

            if dones[0] == 1:
                unit = [0.0]
                next_qu = []
                for m in range(self.sim.nodes[node].agent.tau_N):
                    next_qu.append(unit)
                next_qu = torch.tensor(next_qu).unsqueeze(0).to(self.device)
            else:
                next_qu = self.sim.nodes[next_node].agent.model.forward(1, var_next_s)

            target_sa_quantiles = next_qu
            for j in range(next_qu.shape[1]):
                target_sa_quantiles[0][j][0] = rewards[0] + (1.0 - dones[0]) * next_qu[0][j][0]

            for i in range(1, batch_size):
                var_next_s = torch.tensor(next_states[i], dtype=torch.float)
                var_next_s = var_next_s.to(self.device)
                next_node = int(actions[i])
                if dones[i] == 1:
                    unit = [0.0]
                    next_qu_ = []
                    for m in range(self.sim.nodes[node].agent.tau_N):
                        next_qu_.append(unit)
                    next_qu_ = torch.tensor(next_qu_).unsqueeze(0).to(self.device)

                else:
                    next_qu_ = self.sim.nodes[next_node].agent.model.forward(1, var_next_s)

                target_sa_quantiles_ = next_qu_
                for j in range(next_qu_.shape[1]):
                    target_sa_quantiles_[0][j][0] = rewards[i] + (1.0 - dones[i]) * next_qu_[0][j][0]

                target_sa_quantiles = torch.cat((target_sa_quantiles, target_sa_quantiles_), 0)


            target_qu = target_sa_quantiles
            target_qu = target_qu.transpose(1, 2)
            target_qu = target_qu.to(self.device)

            assert target_qu.shape == (batch_size, 1, self.sim.nodes[node].agent.tau_N)

        td_errors = target_qu - current_sa_quantiles
        assert td_errors.shape == (batch_size, self.sim.nodes[node].agent.tau_N, self.sim.nodes[node].agent.tau_N)
        tau_N = self.sim.nodes[node].agent.tau_N
        taus = torch.arange(
            0, tau_N + 1, device=self.device, dtype=torch.float32) / tau_N
        tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, tau_N)

        kappa = 1
        quantile_huber_loss = self.calculate_quantile_huber_loss(
            td_errors, tau_hats, kappa)

        return quantile_huber_loss, \
               td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)


    def optimize_adam(self, node):  ### 优化QRDQN参数
        # print('node optimize', node)
        batch_size = self.sim.batch_size
        # Buffer for storing the loss-values of the most recent batches.
        sum_loss = 0
        sum_train = 0

        loss_sum = 0

        sample_keys = self.sim.nodes[node].agent.replay_memory.get_sample_keys()
        # print('sample_keys', sample_keys)
        for key in sample_keys:
            size = self.sim.batch_size
            epochs = int(self.sim.nodes[node].agent.replay_memory.memory[key]._n / batch_size)
            if epochs == 0:
                epochs = 1
            if epochs > 3:
                epochs = 3

            for _ in range(epochs):
                # print('memory_key', key, size)
                states, actions, action_idxs, rewards, next_states, dones = self.sim.nodes[
                    node].agent.replay_memory.sample_on_key(size, key)

                quantile_loss, errors = self.calculate_loss(
                    size, states, actions, action_idxs, rewards, next_states, dones, node)
                assert errors.shape == (size, 1)

                # print('errors', errors)

                loss = quantile_loss
                loss_sum += loss

                sum_loss += loss.data
                sum_train += 1
                self.sim.nodes[node].agent.model.lr = 0.0005

                print('loss', loss, 'node', node)

                # sum_loss += objective_loss
                # self.graph.node_list[node].agent.model.set_lr(learning_rate)

                opt = self.sim.nodes[node].agent.model.opt
                opt.zero_grad()
                loss.backward()
                opt.step()

        if node == 0:
            avg_loss = sum_loss / sum_train
            self.loss_history.append(avg_loss)
            self.loss_num1 += 1
            writer1.add_scalar('loss', avg_loss, self.loss_num1)
        elif node == 2:
            avg_loss = sum_loss / sum_train
            # self.loss_history.append(avg_loss)
            self.loss_num2 += 1
            writer3.add_scalar('loss', avg_loss, self.loss_num2)
        loss_sum /= sum_train
        return loss_sum


    def on_receive_pdu(self, packet):  # receive_pdu(pdu.src, pdu.payload)
        # if packet.priority == 0:
        #     print('node', self.id, 'receives packet', packet.id, packet.des)
        if packet.id in self.has_reces:
            return
        self.has_reces.append(packet.id)
        src = packet.travese_path[-1]
        packet.travese_path.append(self.id)
        key1 = str(src) + '_' + str(self.id)
        key2 = str(src)

        # calculate the one hop delay from the last node to the current node
        one_hop_delay = self.now - packet.arrival_time[key2]

        packet.travase_delay[key1] = int(1000 * one_hop_delay)
        packet.end_to_end_delay += 1000 * one_hop_delay
        # print(packet.des, self.id)
        if packet.des != self.id:
            next_node = self.get_nextnode(packet)
            self.mac.send_pdu(packet, next_node)
        else:
            self.reces_for_me.append(packet.id)
            self.e2ed.append(packet.end_to_end_delay)
            print(packet.end_to_end_delay)
            update = False
            if len(self.reces_for_me) % 10 == 0:
                update = True
            self.addsamples(packet, update)

            # Record the actual delay experienced by the packet
            for l in range(len(packet.travese_path) - 1):
                node_ = packet.travese_path[l]
                sub_path = packet.travese_path[l:]
                sum_delay = 0
                path = [str(x) for x in sub_path]
                path = "_".join(path)
                for j in range(len(sub_path) - 1):
                    key = str(sub_path[j]) + '_' + str(sub_path[j + 1])
                    sum_delay += packet.travase_delay[key]

                if str(packet.priority) not in self.sim.nodes[node_].end_to_end_delay.keys():
                    self.sim.nodes[node_].end_to_end_delay[str(packet.priority)] = {
                        path: {str(sum_delay): 1}}
                else:
                    if path not in self.sim.nodes[node_].end_to_end_delay[str(packet.priority)].keys():
                        # print('path not in')
                        self.sim.nodes[node_].end_to_end_delay[str(packet.priority)][path] = {
                            str(sum_delay): 1}
                        # print('bfnode2', node_, self.graph.node_list[node_].end_to_end_delay)
                    else:
                        if str(sum_delay) not in self.sim.nodes[node_].end_to_end_delay[str(packet.priority)][
                            path].keys():
                            self.sim.nodes[node_].end_to_end_delay[str(packet.priority)][path][
                                str(sum_delay)] = 1
                        else:
                            self.sim.nodes[node_].end_to_end_delay[str(packet.priority)][path][
                                str(sum_delay)] += 1


###########################################################
class Simulator:

    ############################
    def __init__(self, until, src, dst, arrival_rates, sum_nodes, adj_T,  frame_slot, num_nodes, device, seed=0):
        self.env = simpy.Environment()
        self.nodes = []
        self.until = until
        self.arrival_rates = arrival_rates
        self.random = random.Random(seed)
        self.src = src
        self.dst = dst
        self.range = range
        self.timeout = self.env.timeout
        self.sum_nodes = sum_nodes
        self.adj_T = adj_T
        self.device = device
        self.batch_size = 64
        self.frame_slot = frame_slot
        self.each_slot = 3
        self.num_nodes = num_nodes
        self.cumulative_reward = [[], [], []]
        self.current_episode = 500

    ############################
    def init(self):
        pass

    ############################
    @property
    def now(self):
        return self.env.now

    def delayed_exec(self, delay, func, *args, **kwargs):
        func = ensure_generator(self.env, func, *args, **kwargs)
        start_delayed(self.env, func, delay=delay)

    ############################
    def init_agent(self):
        for id in range(SUM_NODES):
            me = self.nodes[id]
            me.setAgent()

    ############################
    def run(self, episode):
        self.init()
        # for n in self.nodes:
        #     n.init()
        for id in self.src:
            node = self.nodes[id]
            node.start_time = self.env.now
            self.env.process(node.run())
        for n in self.nodes:
            n.episode = episode
        self.env.run(until=self.env.now + self.until)
        # for n in self.nodes:
        #     n.finish()
