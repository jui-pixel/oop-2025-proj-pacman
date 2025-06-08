# test_entity_initializer.py
import pytest
from unittest.mock import Mock, patch
from game.entities.entity_initializer import initialize_entities
from game.entities.pacman import PacMan
from game.entities.ghost import Ghost
from game.entities.pellets import PowerPellet, ScorePellet

@pytest.fixture
def mock_maze():
    maze = Mock()
    maze.height = 21
    maze.width = 21
    maze.get_tile.side_effect = lambda x, y: '.' if (x == 1 and y == 1) or (x == 5 and y == 5) else 'S' if x == 10 and y == 10 else '#'
    maze.xy_valid.return_value = True
    return maze

def test_initialize_entities(mock_maze):
    with patch('game.entities.entity_initializer.random.choice', side_effect=[(1, 1), (10, 10)]):
        pacman, ghosts, power_pellets, score_pellets = initialize_entities(mock_maze)
        assert isinstance(pacman, PacMan)
        assert len(ghosts) == 4
        assert all(isinstance(ghost, Ghost) for ghost in ghosts)
        assert len(power_pellets) == 0  # 依賴 maze.get_tile 返回 'E'
        assert len(score_pellets) > 0