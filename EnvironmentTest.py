

from Environment import Environment
from AgentRandom import AgentRandom
import numpy as np

rnd = np.random.RandomState(1)

players = []
players.append(AgentRandom(rnd=rnd))
players.append(AgentRandom(rnd=rnd))
players.append(AgentRandom(rnd=rnd))

e = Environment(players = players, num_squares = 6, rnd = rnd)

is_finish = e.run(100)
print("is_finish={}".format(is_finish))