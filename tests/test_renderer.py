# test_renderer.py
import pytest
import pygame
from unittest.mock import Mock, patch
from game.renderer import Renderer
from game.game import Game
from config import CELL_SIZE, MAZE_WIDTH, MAZE_HEIGHT

@pytest.fixture
def setup_renderer():
    pygame.init()
    screen = pygame.display.set_mode((MAZE_WIDTH * CELL_SIZE, MAZE_HEIGHT * CELL_SIZE))
    font = pygame.font.SysFont(None, 36)
    yield Renderer(screen, font, MAZE_WIDTH * CELL_SIZE, MAZE_HEIGHT * CELL_SIZE)
    pygame.quit()

def test_render_initialization(setup_renderer):
    renderer = setup_renderer
    game = Mock(spec=Game)
    game.get_maze.return_value = Mock()
    game.get_maze().height = MAZE_HEIGHT
    game.get_maze().width = MAZE_WIDTH
    game.get_maze().get_tile.return_value = None
    game.get_power_pellets.return_value = []
    game.get_score_pellets.return_value = []
    game.get_pacman.return_value = Mock(current_x=1.0, current_y=1.0)
    game.get_ghosts.return_value = []
    game.is_death_animation_playing.return_value = False
    game.get_lives.return_value = 3
    renderer.render(game, "TestMode", 0)
    assert True  # 檢查無異常