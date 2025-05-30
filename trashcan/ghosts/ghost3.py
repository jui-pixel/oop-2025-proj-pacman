# ghosts/ghost3.py
"""
定義 Ghost3，繼承自 BasicGhost，與 Ghost1 協同，瞄準 Pac-Man 前方與 Ghost1 的對稱點。
"""
from ghosts.basic_ghost import BasicGhost
from game.entities import PacMan
from config import CYAN

class Ghost3(BasicGhost):
    def __init__(self, x: int, y: int, name: str = "Ghost3"):
        """
        初始化 Ghost3，設置位置、名稱和顏色。
        
        Args:
            x (int): x 座標。
            y (int): y 座標。
            name (str): 鬼魂名稱，預設為 "Ghost3"。
        """
        super().__init__(x, y, name, color=CYAN)

    def chase_pacman(self, pacman: PacMan, maze):
        """
        與 Ghost1 協同，瞄準 Pac-Man 前方 2 格與 Ghost1 的對稱點。
        
        Args:
            pacman (PacMan): Pac-Man 物件。
            maze (Map): 迷宮物件。
        """
        from game.entities import initialize_entities
        _, ghosts, _, _ = initialize_entities(maze)
        ghost1 = next((g for g in ghosts if g.name == "Ghost1"), None)
        if not ghost1:
            self.move_random(maze)
            return
        
        dx, dy = 0, 0
        if pacman.target_x > pacman.x:
            dx = 1
        elif pacman.target_x < pacman.x:
            dx = -1
        elif pacman.target_y > pacman.y:
            dy = 1
        elif pacman.target_y < pacman.y:
            dy = -1
        
        mid_x = pacman.x + dx * 2
        mid_y = pacman.y + dy * 2
        target_x = ghost1.x + 2 * (mid_x - ghost1.x)
        target_y = ghost1.y + 2 * (mid_y - ghost1.y)
        target_x = max(0, min(maze.w - 1, target_x))
        target_y = max(0, min(maze.h - 1, target_y))
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        best_direction = None
        min_distance = float('inf')
        
        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if maze.xy_valid(new_x, new_y) and maze.get_tile(new_x, new_y) in ['.', 'D', 'E', 'S']:
                distance = ((new_x - target_x) ** 2 + (new_y - target_y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    best_direction = (dx, dy)
        
        if best_direction and self.set_new_target(best_direction[0], best_direction[1], maze):
            return
        self.move_random(maze)