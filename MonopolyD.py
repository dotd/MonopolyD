import numpy as np

rnd = np.random.RandomState(1)

class Square():
    typs = ['money_change', 'asset']

    def __init__(self):
        self.type = Square.typs[rnd.randint(len(Square.typs))]
        self.owner = None
        if self.type == "asset":
            self.price_ladder = rnd.randint(1, 10) * np.array(list(range(100, 600, 100)))
            self.ownership_level = 0
        elif self.type == "money_change":
            self.money_change = rnd.randint(-100, 100)

    def tostr(self):
        sb = "type={} ".format(self.type)
        if self.type == "asset":
            sb += "owner={} ".format(self.owner)
            sb += ",".join(str(x) for x in self.price_ladder)
        else:
            sb += "money_change={}".format(self.money_change)
        return sb

class MonopolyD():
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
        for i in range(self.num_squares):
            self.squares.append(Square())
        self.rounds = 100

        state_vec = self.get_state_vector(0)
        self.state_size = len(state_vec)
        self.action_size = len(MonopolyD.actions)
        self.reward = np.zeros(shape=(self.num_players))


    actions = ["nothing", "buy"]

    def step(self):
        round_reward = self.reward[self.cur_player]
        self.reward[self.cur_player]=0

        print("---------------")
        print("Board")
        print(self.show_board())
        print("Before:\n" + self.__str__())


        # Advance the player on the board
        self.advance(self.cur_player)

        # get the current square that the player is on it
        player_square = self.squares[self.position[self.cur_player]]
        # get the valid actions for this player and this square
        actions = self.get_valid_actions(self.cur_player, player_square)
        print(">> Possible actions: " + self.show_actions(actions))

        # random action
        action_idx = rnd.choice(len(actions))

        action = actions[action_idx]
        print(">> Chosen action: " + str(action))
        self.apply_pays(self.cur_player, player_square, action)

        print(">> After:\n" + self.__str__())
        status = self.check_player_broke()
        if status != None:
            return
        self.next_player()

    def run(self):
        for self.round in range(self.rounds):
            self.step()

    def next_player(self):
        self.cur_player = (self.cur_player + 1) % self.num_players

    def advance(self, player):
        self.position[player] = (self.position[player] + rnd.randint(self.min_advance,
                                                                     self.max_advance)) % self.num_squares

    def __str__(self, lines_seperator="\n"):
        s = []
        s.append("round={}".format(self.round))
        s.append("player={}, players are in {}".format(self.cur_player, ",".join([str(x) for x in self.position])))
        s.append("money status: " + str(self.money))
        return lines_seperator.join(s)

    def show_board(self, lines_seperator="\n"):
        s = []
        s.append(lines_seperator.join(x.tostr() for x in self.squares))
        return lines_seperator.join(s)

    actions = ["nothing", "buy"]

    def show_actions(self, actions, separator="; "):
        s = []
        for action, cost in actions:
            s.append("{}->{} ".format(action, cost))
        return separator.join(s)

    def get_valid_actions(self, player, square):
        if square.type == "asset" and square.owner == None:
            actions = [("nothing", 0), ("buy", square.price_ladder[0])]
        elif square.type == "asset" and square.owner == player and square.ownership_level < len(
                square.price_ladder) - 1:
            actions = [("nothing", 0), ("buy", square.price_ladder[square.ownership_level + 1])]
        else:
            actions = [("nothing", 0)]
        return actions

    def apply_pays(self, player, square, action):
        if (square.type == "asset") and (square.owner == None) and action[0] == "buy" and self.money[player] >= action[1]:
            square.owner = player
            diff = -square.price_ladder[square.ownership_level]
            self.money[player] += diff
        elif (square.type == "asset") and (square.owner != None) and (square.owner != player):
            print("Player PAYS")
            diff = -square.price_ladder[square.ownership_level]
            self.money[player] += diff
            self.reward[square.owner] += square.price_ladder[square.ownership_level]
        elif square.type == "money_change":
            self.money[player] += square.money_change
            self.reward[square.owner] -= square.money_change
        else:
            print("apply_pays did nothing")

    def check_player_broke(self):
        for p in range(self.num_players):
            if self.money[p] < 0:
                print("player {} is broke!!!".format(p))
                return p
        return None

    def get_state_vector(self, player_idx):
        d = {}
        # Locations
        for player in range(self.num_players):
            # Locations
            #index 0 is the current player_idx. The rest are the rest of the players
            player_num = (player_idx + player)%self.num_players
            key = "pos{}".format(player_num)
            value = self.position[player_num]
            d[key] = value
            # Money for each player
            key = "money{}".format(player_num)
            value = self.money[player_num]
            d[key] = value

        # Ownerships - common to all squares and players.
        # (1) If a square is owned by the player_idx, then we put there the values of the asset with a positive sign.
        # (2) If this square is owned by other user, we put the asset value with a minus sign.
        # (3) Otherwise, we put 0.
        for square_idx, square in enumerate(self.squares):
            key = "sq{}".format(square_idx)
            value = None
            if square.type == "asset":
                if square.owner==player_idx:
                    value = square.price_ladder[square.ownership_level]
                elif square.owner!=None and square.owner != player_idx:
                    value = -square.price_ladder[square.ownership_level]
            d[key] = value

        return d



m = MonopolyD(num_squares=6, num_players=2)
m.run()

