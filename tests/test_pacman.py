# test_pacman.py
import pytest
from unittest.mock import Mock, patch
from game.entities.pacman import PacMan
from game.entities.pellets import PowerPellet, ScorePellet
from game.entities.ghost import Ghost

@pytest.fixture
def mock_maze():
    maze = Mock()
    maze.xy_valid.return_value = True
    maze.get_tile.return_value = '.'
    return maze

def test_pacman_eat_pellet(mock_maze):
    pacman = PacMan(1, 1)
    pellets = [PowerPellet(1, 1)]
    score = pacman.eat_pellet(pellets)
    assert score == 10
    assert len(pellets) == 0
    assert pacman.score == 10

def test_pacman_rule_based_ai_move(mock_maze):
    pacman = PacMan(1, 1)
    pacman.set_new_target = Mock(return_value=True)
    with patch('game.entities.pacman.PacMan.find_path', return_value=(1, 0)):
        result = pacman.rule_based_ai_move(mock_maze, [], [ScorePellet(2, 1)], [])
        assert result
        assert pacman.last_direction == (1, 0)