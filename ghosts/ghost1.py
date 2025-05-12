# ghosts/ghost1.py
from ghosts.basic_ghost import BasicGhost
from game.entities import PacMan

class Ghost1(BasicGhost):
    def __init__(self, x: int, y: int, name="Ghost1"):
        super().__init__(x, y, name)