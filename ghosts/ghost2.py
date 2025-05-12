#ghosts/ghost2.py
from ghosts.basic_ghost import BasicGhost
from game.entities import PacMan

class Ghost2(BasicGhost):
    def __init__(self, x: int, y: int, name="Ghost2"):
        super().__init__(x, y, name)

    def chase_pacman(self, pacman: PacMan, maze):
        dx = self.x - pacman.x
        dy = self.y - pacman.y
        directions = [(1 if dx > 0 else -1, 0), (0, 1 if dy > 0 else -1)]
        for dir_x, dir_y in directions:
            if self.set_new_target(dir_x, dir_y, maze):
                return
        self.move_random(maze)