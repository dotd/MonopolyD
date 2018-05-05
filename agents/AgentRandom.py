

class AgentRandom:
    def __init__(self, rnd = None):
        self.rnd = rnd

    def init_agent(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

    def choose_action(self, state, valid_actions=None):
        action = self.rnd.choice(self.num_actions)
        return action

