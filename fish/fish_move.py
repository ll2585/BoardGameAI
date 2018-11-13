class FishMove:
    def __init__(self, penguin_location, destination=None, player_id=None, type="move"):
        self.start = penguin_location
        self.end = destination
        self.type = type
        self.player_id = player_id

    def __repr__(self):
        if self.type == "move":
            return '{2} MOVE from {0} to {1}'.format(self.start, self.end, self.player_id)
        elif self.type == "place":
            return '{1} PLACE at {0}'.format(self.start, self.player_id)
        else:
            raise Exception("WRONG MOVE TYPE")