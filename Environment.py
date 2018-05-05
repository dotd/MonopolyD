import numpy as np
from MonopolyD import MonopolyD

class Environment:
    def __init__(self, players, num_squares = 6,  rnd = np.random.RandomState(1)):
        self.monopoly = MonopolyD(num_squares=6, num_players=len(players), rnd=rnd)
        self.state_size = len(self.monopoly.get_state_info_as_dict(0))
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
        cur_player = self.players[self.monopoly.cur_player]

        state = self.monopoly.get_state_as_vec(self.monopoly.cur_player)
        valid_actions = self.monopoly.valid_actions
        action_idx = cur_player.choose_action(state["vec"], valid_actions)

        strs = self.monopoly.step_post(action_idx)
        print("\n".join(strs))

        return self.monopoly.is_finish()






