

class AgentBase():
    '''
    Here we define the interface of the agent
    '''

    def __init__(self):
        '''
        This is a setup without state space and action. Sometimes we need to wait for the environment
        in order to understand what is the state space
        '''
        pass

    def init_agent(self, dim_state, dim_action):
        '''
        Here we explicitly define the state space and action space
        '''
        pass

    def choose_action(self, state, valid_actions = None):
        '''
        Given state the agent chooses the action
        Also, optionally, we have valid actions so we won't choose forbidden action
        '''
        pass

