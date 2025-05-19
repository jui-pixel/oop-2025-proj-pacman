# ghosts/ghost1.py
from ghosts.basic_ghost import BasicGhost
from game.entities import PacMan
from config import RED

class Ghost1(BasicGhost):
    def __init__(self, x: int, y: int, name: str = "Ghost1"):
        super().__init__(x, y, name, color=RED)

    def chase_pacman(self, pacman: PacMan, maze):
        """直接追逐 Pac-Man，選擇縮短距離的方向。"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        best_direction = None
        min_distance = float('inf')

        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'A', 'o', 's', 'S']:
                distance = ((new_x - pacman.x) ** 2 + (new_y - pacman.y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    best_direction = (dx, dy)

        if best_direction and self.set_new_target(best_direction[0], best_direction[1], maze):
            return
        self.move_random(maze)