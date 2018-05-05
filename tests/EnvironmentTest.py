

from Environment import Environment
from agents.AgentRandom import AgentRandom
from agents.AgentLogistic import AgentLogistic
import numpy as np

rnd = np.random.RandomState(1)

# Create the players, without the parameters
players = []
players.append(AgentRandom(rnd=rnd))
players.append(AgentRandom(rnd=rnd))
players.append(AgentLogistic())

e = Environment(players = players, num_squares = 6, rnd = rnd)
print("Typical state space = {}".format(e.monopoly.get_state_info(0)))
dim_state = len(e.monopoly.get_state_info(0))
rep_size = dim_state*2
dim_action = 2 # Buy and do nothing.
players[2].init_agent(dim_state, rep_size, dim_action)

num_episodes=1
maximum_steps_per_episode = 1000
for episode in range(num_episodes):
    print("episode #={}".format(episode))
    for step in range(maximum_steps_per_episode):
        is_finish = e.run(1)
        if is_finish:
            print("is_finish={}".format(is_finish))
            break
