#ghosts/ghost4.py
from ghosts.basic_ghost import BasicGhost
from game.entities import PacMan

class Ghost4(BasicGhost):
    def __init__(self, x: int, y: int, name="Ghost4"):
        super().__init__(x, y, name)

    def chase_pacman(self, pacman: PacMan, maze):
        """
        Ghost4: Attempts to predict Pac-Man's next position based on movement direction.
        """
        dx = pacman.target_x - self.x
        dy = pacman.target_y - self.y
        directions = [(1 if dx > 0 else -1, 0), (0, 1 if dy > 0 else -1)]
        for dir_x, dir_y in directions:
            if self.set_new_target(dir_x, dir_y, maze):
                print(f"{self.name} moving to ({self.target_x}, {self.target_y}) from ({self.x}, {self.y})")
                return
        self.move_random(maze)