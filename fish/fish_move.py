class FishMove:
    def __init__(self, penguin_location, destination=None, player=None, type="move"):
        self.start = penguin_location
        self.end = destination
        self.type = type
        self.player = player

    def __repr__(self):
        if self.type == "move":
            return 'MOVE from {0} to {1}'.format(self.start, self.end)
        elif self.type == "place":
            return 'PLACE at {0}'.format(self.start)
        else:
            raise Exception("WRONG MOVE TYPE")