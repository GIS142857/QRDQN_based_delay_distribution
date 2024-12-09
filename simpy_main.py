import os
import numpy as np
import torch
from config import *
from simy_env import Simulator, Node

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tau_N = 5000

packet_len = 8 * 1000

sim_time = 10*UNIT

# Build a Simulator
sim = Simulator(sim_time, SRC, DST, ARRIVAL_RATE, SUM_NODES, ADJ_TABLE, FRAME_SLOT, SUM_NODES, device)
for id in range(SUM_NODES):
    sim.nodes.append(Node(sim, id, SRC, DST, ARRIVAL_RATE, NODE_POSITION[id], device))

sim.init_agent()

# start the simulation
episodes = 2000
# sim.episode = episodes
for e in range(episodes):
    print('episode', e)
    if e > 0:
        for n in sim.nodes:
            n.mac.queues = []
    sim.run(e)

    # if (e + 1) % 100 == 0:
    #     store_path = 'E2ED'
    #     for n in sim.nodes:
    #         sub_path = store_path + '_' + str(n.id) + '.txt'
    #         with open(sub_path, 'w') as fl:
    #             fl.write(json.dumps(n.end_to_end_delay))

# for i in range(3):
#     print('src', sources[i], sim.nodes[sources[i]].sends)
#     print('des', dess[i], 'receives', len(sim.nodes[dess[i]].reces_for_me))
#     print('des', dess[i], sim.nodes[dess[i]].e2ed)
