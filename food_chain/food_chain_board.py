import numpy as np
import random

class FoodChainTile:
    def __init__(self, tile_id):
        self.tile_id = tile_id
        self.tiles = [[[],[],[],[],[]],
                        [[], [], [], [], []],
                        [[], [], [], [], []],
                        [[], [], [], [], []],
                        [[], [], [], [], []]]
        self.houses = []
        self.roads = []

    def add_road_at(self, tiles=None, road_id=None):
        new_road = FoodChainRoad(road_id)
        self.roads.append(new_road)
        for tile in tiles:
            self.tiles[tile[0]][tile[1]].append(new_road)
        return self

    def add_house_at(self, house_num=None, top_left_tile=None):
        new_house = FoodChainHouse(number=house_num)
        self.houses.append(new_house)
        self.tiles[top_left_tile[0]][top_left_tile[1]].append(new_house)
        self.tiles[top_left_tile[0]][top_left_tile[1]+1].append(new_house)
        self.tiles[top_left_tile[0]+1][top_left_tile[1]].append(new_house)
        self.tiles[top_left_tile[0]+1][top_left_tile[1]+1].append(new_house)
        return self

    def add_beer_at(self, tile):
        self.tiles[tile[0]][tile[1]].append("Beer")
        return self

    def add_coke_at(self, tile):
        self.tiles[tile[0]][tile[1]].append("Coke")
        return self

    def add_lemonade_at(self, tile):
        self.tiles[tile[0]][tile[1]].append("Lmde")
        return self

    def display_tiles(self):
        print('\n'.join(str(v) for v in self.tiles))

    def rotate(self, times):
        tiles = self.tiles
        for i in range(times):
            tiles = [[x[0] for x in tiles][::-1],
             [x[1] for x in tiles][::-1],
             [x[2] for x in tiles][::-1],
             [x[3] for x in tiles][::-1],
             [x[4] for x in tiles][::-1]]
        self.tiles = tiles
        return self

class FoodChainHouse:
    def __init__(self, number):
        self.number = number

    def __repr__(self):
        return 'Hs{0:02d}'.format(self.number)

class FoodChainRoad:
    def __init__(self, road_id):
        self.road_id = road_id

    def __repr__(self):
        return 'Rd{0:02d}'.format(self.road_id)


FOOD_CHAIN_TILES = [
    FoodChainTile("A").add_road_at(tiles=[[0,2],[1,2],[2,0],[2,1],[2,2],[2,3],[2,4],[3,2],[4,2]],road_id=0).add_house_at(house_num=2, top_left_tile=[3,0]),
    FoodChainTile("B").add_road_at(tiles=[[0,2],[1,2],[2,0],[2,1],[2,2],[2,3],[2,4],[3,2],[4,2]],road_id=1).add_house_at(house_num=4, top_left_tile=[0,3]),
    FoodChainTile("C").add_road_at(tiles=[[0,0],[0,1],[0,2],[0,3],[0,4],[1,0],[2,0],[3,0],[4,0],[4,1],[4,2],[4,3],[4,4],[3,4],[2,4],[1,4]],road_id=2).add_house_at(house_num=5, top_left_tile=[1,2]),
    FoodChainTile("D").add_road_at(tiles=[[0,0],[0,1],[0,2],[0,3],[0,4],[1,0],[2,0],[1,4],[2,4]],road_id=3).add_house_at(house_num=7,top_left_tile=[1,1]),
    FoodChainTile("E").add_road_at(tiles=[[0,0],[0,1],[0,2],[1,0],[2,0]],road_id=4).add_road_at([[2,4],[3,4],[4,2],[4,3],[4,4]],road_id=5).add_house_at(house_num=8,top_left_tile=[2,2]).add_beer_at([1,1]),
    FoodChainTile("F").add_road_at(tiles=[[0,2],[1,2],[2,0],[2,1],[2,2],[2,3],[2,4]],road_id=6).add_house_at(house_num=10, top_left_tile=[0,0]),
    FoodChainTile("G").add_road_at(tiles=[[0,2],[1,2],[2,2],[3,2],[4,2]],road_id=7).add_road_at(tiles=[[2,0],[2,1],[2,2],[2,3],[2,4]],road_id=8).add_house_at(house_num=12, top_left_tile=[0,0]),
    FoodChainTile("H").add_road_at(tiles=[[0,2],[1,2],[2,0],[2,1],[2,2],[2,3],[2,4]],road_id=9).add_house_at(house_num=13, top_left_tile=[3,1]),
    FoodChainTile("I").add_road_at(tiles=[[0,2],[1,2],[2,0],[2,1],[2,2],[2,3],[2,4]],road_id=10).add_house_at(house_num=15, top_left_tile=[3,3]),
    FoodChainTile("J").add_road_at(tiles=[[0,2],[0,3],[0,4],[1,4],[2,4]],road_id=11).add_road_at([[2,0],[3,0],[4,0],[4,1],[4,2]],road_id=12).add_house_at(house_num=16, top_left_tile=[1,1]),
    FoodChainTile("K").add_road_at(tiles=[[0,0],[0,1],[0,2],[0,3],[0,4],[1,0],[1,4],[2,0],[2,4]],road_id=13).add_house_at(house_num=18, top_left_tile=[1,2]),
    FoodChainTile("L").add_road_at(tiles=[[0,2],[1,2],[2,0],[2,1],[2,2],[2,3],[2,4]],road_id=14).add_lemonade_at(tile=[3,3]),
    FoodChainTile("M").add_road_at(tiles=[[0,2],[0,3],[0,4],[1,4],[2,4]],road_id=15).add_road_at([[2,0],[3,0],[4,0],[4,1],[4,2]],road_id=16).add_lemonade_at(tile=[1,3]).add_coke_at(tile=[3,1]),
    FoodChainTile("N").add_road_at(tiles=[[0,2],[1,2],[2,0],[2,1],[2,2],[2,3],[2,4]],road_id=17).add_beer_at(tile=[1,1]),
    FoodChainTile("O").add_road_at(tiles=[[0,2],[1,2],[2,0],[2,1],[2,2],[2,3],[2,4],[3,2],[4,2]],road_id=18).add_beer_at(tile=[0,1]),
    FoodChainTile("P").add_road_at(tiles=[[0,2],[1,2],[2,2],[3,2],[4,2]],road_id=19).add_road_at(tiles=[[2,0],[2,1],[2,2],[2,3],[2,4]],road_id=20).add_lemonade_at(tile=[1,0]).add_beer_at(tile=[4,3]),
    FoodChainTile("Q").add_road_at(tiles=[[0,2],[1,2],[2,0],[2,1],[2,2],[2,3],[2,4]],road_id=21).add_coke_at(tile=[3,1]),
    FoodChainTile("R").add_road_at(tiles=[[0,2],[1,2],[2,0],[2,1],[2,2],[2,3],[2,4],[3,2],[4,2]],road_id=22).add_coke_at(tile=[1,1]),
    FoodChainTile("S").add_road_at(tiles=[[0,2],[1,2],[2,0],[2,1],[2,2],[2,3],[2,4],[3,2],[4,2]],road_id=23).add_coke_at(tile=[0,3]).add_beer_at(tile=[3,0]),
    FoodChainTile("T").add_road_at(tiles=[[0,2],[1,2],[2,0],[2,1],[2,2],[2,3],[2,4],[3,2],[4,2]],road_id=24).add_lemonade_at(tile=[1,1]),

]


class FoodChainBoard:
    def __init__(self):
        self.board = [FOOD_CHAIN_TILES[:3],
                      FOOD_CHAIN_TILES[3:6],
                      FOOD_CHAIN_TILES[6:9]]
        for col in range(len(self.board)):
            for tile in self.board[col]:
                tile = tile.rotate(random.randint(0,3))

    def display(self):
        row_strings = []
        for col in range(len(self.board)): #3 rows X 3 columns X5 rows in tile X5 columns in tile
            string_rows = [[], [], [], [], []]
            for tile in self.board[col]:
                for i in range(5): #cus its 5
                    for t in tile.tiles[i]:
                        if len(t) == 0:
                            val = '    '
                        elif len(t) > 1:
                            val = str(t[0])
                        else:
                            val = str(t[0])
                        string_rows[i].append(val)
            first_row = [''.join(string_rows[i]) for i in range(5)]
            row_strings.append('\n'.join(first_row))
        return '\n'.join(row_strings)

    def get_possible_restaurant_entrances(self):
        for col in range(len(self.board)):
            for row in range(len(self.board[col])):
                tile = self.board[col][row]
                squares = tile.tiles
                for x in range(5):
                    for y in range(5):
                        left_square = None
                        right_square = None
                        up_square = None
                        right_square = None
                        if col != 0 or x != 0:
                            if x != 0: #same tile
                                left_square = squares[x-1][y]
                            else:
