# game/entities/pellets.py
"""
定義遊戲中的能量球和分數球。
"""
from .entity_base import Entity

class PowerPellet(Entity):
    def __init__(self, x: int, y: int, value: int = 10):
        """
        初始化能量球，設置分數值。
        
        Args:
            x (int): x 座標。
            y (int): y 座標。
            value (int): 分數值，預設為 10。
        """
        super().__init__(x, y, 'o')
        self.value = value

class ScorePellet(Entity):
    def __init__(self, x: int, y: int, value: int = 2):
        """
        初始化分數球，設置分數值。
        
        Args:
            x (int): x 座標。
            y (int): y 座標。
            value (int): 分數值，預設為 2。
        """
        super().__init__(x, y, 's')
        self.value = value