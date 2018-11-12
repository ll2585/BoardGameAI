class FoodChainEmployee:
    def __init__(self, title, slots=None):
        self.title = title
        self.slots = slots


CEO_CARD = FoodChainEmployee(slots=3, title='CEO')