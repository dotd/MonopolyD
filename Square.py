
#from MonopolyD import rnd
import numpy as np

global rnd

class Square():
    typs = ['money_change', 'asset']

    def __init__(self, rnd):
        self.rnd = rnd
        self.type = Square.typs[rnd.randint(len(Square.typs))]
        self.owner = None
        if self.type == "asset":
            self.price_ladder = self.rnd.randint(1, 10) * np.array(list(range(100, 600, 100)))
            self.ownership_level = 0
        elif self.type == "money_change":
            self.money_change = self.rnd.randint(-100, 100)

    def tostr(self):
        sb = "type={} ".format(self.type)
        if self.type == "asset":
            sb += "owner={} ".format(self.owner)
            sb += ",".join(str(x) for x in self.price_ladder)
        else:
            sb += "money_change={}".format(self.money_change)
        return sb