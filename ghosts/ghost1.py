# ghosts/ghost1.py
from ghosts.basic_ghost import BasicGhost
from game.entities import PacMan

class Ghost1(BasicGhost):
    def __init__(self, x: int, y: int, name="Ghost1"):
        super().__init__(x, y, name)

    def chase_pacman(self, pacman: PacMan, maze):
        dx = pacman.x - self.x
        dy = pacman.y - self.y
        directions = [(1 if dx > 0 else -1, 0), (0, 1 if dy > 0 else -1)]
        for dir_x, dir_y in directions:
            if self.set_new_target(dir_x, dir_y, maze):
                return
        self.move_random(maze)