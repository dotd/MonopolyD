import numpy as np
from Square import Square
from collections import OrderedDict

global rnd
BROKE_PLAYERS = "BROKE_PLAYERS"

class MonopolyD():
    actions = ["nothing", "buy"]

    def __init__(self, num_squares=40, num_players=4, money_start=1500, **kwargs):
        self.num_squares = num_squares
        self.num_players = num_players
        self.position = [0] * num_players
        self.money = [money_start] * num_players
        # minimal and maximal steps for a player to advance
        self.min_advance = 1
        self.max_advance = 3
        # what is the current player
        self.cur_player = 0
        # hold the squares
        self.squares = []
        self.rnd = kwargs.get("rnd")
        for i in range(self.num_squares):
            self.squares.append(Square(id=i, rnd=self.rnd))
        self.rounds = 100
        self.turn = 0

        state_vec = self.get_state_info_as_dict(0)
        self.state_size = len(state_vec)
        self.action_size = len(MonopolyD.actions)
        self.reward = np.zeros(shape=(self.num_players))
        self.broke_players = set()
        self.feature_map = {}

    def step(self):
        strs = self.step_pre()
        print("\n".join(strs))
        # random action
        action_idx = self.rnd.choice(len(self.valid_actions))
        strs = self.step_post(action_idx)
        print("\n".join(strs))

    def step_pre(self):
        self.reward[self.cur_player]=0
        strs = []

        strs.append("---------------")
        strs.append("Pre section")
        strs.append("board:")
        strs.append(self.show_board())
        strs.append("Before:\n" + self.__str__())

        # Advance the player on the board
        self.advance(self.cur_player)

        # get the current square that the player is on it
        self.player_square = self.squares[self.position[self.cur_player]]
        # get the valid actions for this player and this square
        self.valid_actions = self.get_valid_actions(self.cur_player, self.player_square)
        strs.append("valid_actions=" + self.show_actions(self.valid_actions))
        return strs

    def step_post(self, action_idx):
        strs = []
        strs.append("Post section")
        #action = self.valid_actions[action_idx]
        strs.append("Chosen action: " + str(action_idx))
        # Now, if the agent chose forbidden action for him, we override his choice with "do_nothing"
        if len(self.valid_actions)==1:
            action_idx=0

        self.apply_pays(self.cur_player, self.player_square, action_idx)

        strs.append("After:\n" + self.__str__())
        self.update_broke_players()
        self.next_player()
        return strs

    def is_finish(self):
        return len(self.broke_players) == self.num_players - 1

    def run(self):
        for self.round in range(self.rounds):
            self.step()
            is_finish = self.is_finish()
            if is_finish:
                print("The game was finished!")
                break

    def next_player(self):
        '''
        look for the next player
        '''
        for i in range(1,self.num_players):
            self.cur_player = (self.cur_player + i) % self.num_players
            if self.cur_player not in self.broke_players:
                return
        print("You have problem with the while in next player")

    def advance(self, player):
        self.position[player] = (self.position[player] + self.rnd.randint(self.min_advance,
                                                                     self.max_advance)) % self.num_squares
        self.turn += 1

    def __str__(self, lines_seperator="\n"):
        s = []
        s.append("turn={}".format(self.turn))
        s.append("player={}, players are in {}".format(self.cur_player, ",".join([str(x) for x in self.position])))
        s.append("money status: " + str(self.money))
        return lines_seperator.join(s)

    def show_board(self, lines_seperator="\n"):
        s = []
        s.append(lines_seperator.join(x.__str__() for x in self.squares))
        return lines_seperator.join(s)

    def show_actions(self, actions, separator="; "):
        s = []
        for action, cost in actions:
            s.append("{}->{} ".format(action, cost))
        return separator.join(s)

    def get_valid_actions(self, player_idx, square):
        if square.type == "asset" and square.owner == None:
            '''The square is free'''
            actions = [("nothing", 0), ("buy", square.price_ladder[0])]
        elif square.type == "asset" and square.owner == player_idx and square.ownership_level < len(
                square.price_ladder) - 1:
            ''' The square belongs to the player player_idx and it can buy it.'''
            actions = [("nothing", 0), ("buy", square.price_ladder[square.ownership_level + 1])]
        else:
            ''' No action can be done.'''
            actions = [("nothing", 0)]
        return actions

    def apply_pays(self, player, square, action_idx):
        action_vec = self.valid_actions[action_idx]
        if (square.type == "asset") and (square.owner == None) and action_vec[0] == "buy" and self.money[player] >= action_vec[1]:
            square.owner = player
            diff = -square.price_ladder[square.ownership_level]
            self.money[player] += diff
        elif (square.type == "asset") and (square.owner != None) and (square.owner != player):
            diff = -square.price_ladder[square.ownership_level]
            print("Player {} pays to Player {} ${}".format(player, square.owner, diff))
            self.money[player] += diff
            self.reward[square.owner] += square.price_ladder[square.ownership_level]
        elif square.type == "money_change":
            self.money[player] += square.money_change
            self.reward[square.owner] -= square.money_change
        else:
            print("apply_pays did nothing")

    def update_broke_players(self):
        for p in range(self.num_players):
            if self.money[p] < 0 and p not in self.broke_players:
                print("player {} is broke!!!".format(p))
                self.broke_players.add(p)

    def get_state_info_as_dict(self, player_idx):
        # dictionary with the state variables
        player_info_dict = {}

        # going over all the players
        for player in range(self.num_players):
            # Locations
            # index 0 is the current player_idx. The rest are the rest of the players in modulo order
            player_num = (player_idx + player) % self.num_players
            key = "pos{}".format(player_num)
            value = self.position[player_num]
            player_info_dict[key] = value
            # Money for each player
            key = "money{}".format(player_num)
            value = self.money[player_num]
            player_info_dict[key] = value

        # Ownerships - common to all squares and players.
        # (1) If a square is owned by the player_idx, then we put there the values of the asset with a positive sign.
        # (2) If this square is owned by other user, we put the asset value with a minus sign.
        # (3) Otherwise, we put 0.
        for square_idx, square in enumerate(self.squares):
            key = "sq{}".format(square_idx)
            value = 0
            if square.type == "asset":
                if square.owner==player_idx:
                    value = square.price_ladder[square.ownership_level]
                elif square.owner!=None and square.owner != player_idx:
                    value = -square.price_ladder[square.ownership_level]
            player_info_dict[key] = value

        return player_info_dict

    def get_state_as_vec(self, player_idx):
        dict = OrderedDict(sorted(self.get_state_info_as_dict(player_idx).items()))
        vec = [0 for x in range(len(dict))]
        for idx, key_value in enumerate(dict.items()):
            vec[idx] = key_value[1]
        return {"dict":dict, "vec":vec}

    def get_state_dim(self):
        # get the state dictionary
        state_dict = self.get_state_info_as_dict(0)
        return len(state_dict)

    def get_state_vector(self, player_info_dict):
        for key, value in player_info_dict.items():
            if key not in self.feature_map:
                self.feature_map[key] = len(self.feature_map)

        vec = np.zeros(shape=(len(self.feature_map)))
        for key, value in player_info_dict.items():
            vec[self.feature_map[key]] = value
        return value



