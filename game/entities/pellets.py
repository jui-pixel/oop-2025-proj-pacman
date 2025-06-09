# game/entities/pellets.py
"""
定義遊戲中的能量球（PowerPellet）和分數球（ScorePellet），作為 Pac-Man 遊戲中的可收集物體。
"""

from .entity_base import Entity

class PowerPellet(Entity):
    def __init__(self, x: int, y: int, value: int = 10):
        """
        初始化能量球，設置其位置和分數值。

        原理：
        - 能量球（Power Pellet）是 Pac-Man 遊戲中的特殊道具，吃掉後會使鬼魂進入可食用狀態。
        - 每個能量球有固定的分數值（預設為 10），並繼承 Entity 基類的坐標和符號屬性。
        - 符號 'o' 用於表示能量球在迷宮中的位置。

        Args:
            x (int): 迷宮中的 x 坐標（格子坐標）。
            y (int): 迷宮中的 y 坐標（格子坐標）。
            value (int): 能量球的分數值，預設為 10。
        """
        super().__init__(x, y, 'o')  # 調用基類 Entity 初始化，設置坐標和符號
        self.value = value  # 設置能量球的分數值

class ScorePellet(Entity):
    def __init__(self, x: int, y: int, value: int = 2):
        """
        初始化分數球，設置其位置和分數值。

        原理：
        - 分數球（Score Pellet）是 Pac-Man 遊戲中的主要收集目標，吃掉後增加分數。
        - 每個分數球有固定的分數值（預設為 2），並繼承 Entity 基類的坐標和符號屬性。
        - 符號 's' 用於表示分數球在迷宮中的位置。

        Args:
            x (int): 迷宮中的 x 坐標（格子坐標）。
            y (int): 迷宮中的 y 坐標（格子坐標）。
            value (int): 分數球的分數值，預設為 2。
        """
        super().__init__(x, y, 's')  # 調用基類 Entity 初始化，設置坐標和符號
        self.value = value  # 設置分數球的分數值