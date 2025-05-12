#ghosts/ghost3.py
from ghosts.basic_ghost import BasicGhost
from game.entities import PacMan

class Ghost3(BasicGhost):
    def __init__(self, x: int, y: int, name="Ghost3"):
        super().__init__(x, y, name)

    def chase_pacman(self, pacman: PacMan, maze):
        """
        Ghost3: Moves randomly half the time, aggressively chases Pac-Man the other half.
        """
        import random
        if random.random() < 0.5:
            self.move_random(maze)
        else:
            dx = pacman.x - self.x
            dy = pacman.y - self.y
            directions = [(1 if dx > 0 else -1, 0), (0, 1 if dy > 0 else -1)]
            for dir_x, dir_y in directions:
                if self.set_new_target(dir_x, dir_y, maze):
                    print(f"{self.name} moving to ({self.target_x}, {self.target_y}) from ({self.x}, {self.y})")
                    return