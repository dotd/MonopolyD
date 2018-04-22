import numpy as np
from MonopolyD import MonopolyD

class Environment:
    def __init__(self, players, num_squares = 6,  rnd = np.random.RandomState(1)):
        self.monopoly = MonopolyD(num_squares=6, num_players=len(players), rnd=rnd)
        self.players = players

    def run(self, num_steps):
        for i in range(num_steps):
            is_finish = self.step()
            if is_finish:
                return True
        return False

    def step(self):
        strs = self.monopoly.step_pre()
        print("\n".join(strs))
        player = self.players[self.monopoly.cur_player]

        state = self.monopoly.get_state_vector(self.monopoly.cur_player)
        valid_actions = self.monopoly.valid_actions
        action_idx = player.choose_action(state, valid_actions)

        strs = self.monopoly.step_post(action_idx)
        print("\n".join(strs))

        return self.monopoly.is_finish()






