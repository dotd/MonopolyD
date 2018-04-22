
from MonopolyD import MonopolyD
import numpy as np
rnd = np.random.RandomState(1)
m = MonopolyD(num_squares=6, num_players=2, rnd=rnd)
m.run()