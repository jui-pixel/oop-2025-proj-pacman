#ghosts/ghost2.py
from ghosts.basic_ghost import BasicGhost
from game.entities import PacMan

class Ghost2(BasicGhost):
    def __init__(self, x: int, y: int, name="Ghost2"):
        super().__init__(x, y, name)