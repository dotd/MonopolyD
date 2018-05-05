

from Environment import Environment
from agents.AgentRandom import AgentRandom
from agents.AgentLogistic import AgentLogistic
import numpy as np

rnd = np.random.RandomState(1)

# Create the players, without the parameters
players = []
players.append(AgentRandom(rnd=rnd))
players.append(AgentRandom(rnd=rnd))
players.append(AgentLogistic(rnd=rnd))

env = Environment(players = players, num_squares = 6, rnd = rnd)
print("Typical state space = {}".format(env.monopoly.get_state_info_as_dict(0)))
dim_state = len(env.monopoly.get_state_info_as_dict(0))
rep_size = dim_state*2
dim_action = 2 # The actinos are {buy, do nothing}.
players[0].init_agent(dim_state, dim_action)
players[1].init_agent(dim_state, dim_action)
players[2].init_agent(dim_state, dim_action, rep_size=rep_size)

num_episodes=1
maximum_steps_per_episode = 1000
for episode in range(num_episodes):
    print("episode #={}".format(episode))
    for step in range(maximum_steps_per_episode):
        is_finish = env.step()
        if is_finish:
            print("is_finish={}".format(is_finish))
            break
