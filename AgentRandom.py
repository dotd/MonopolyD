

class AgentRandom:
    def __init__(self, rnd):
        self.rnd = rnd

    def choose_action(self, state, valid_actions):
        action = self.rnd.choice(len(valid_actions))
        return action

