import pytest
import pygame
from unittest.mock import Mock, patch, MagicMock, ANY
from game.renderer import Renderer
from game.game import Game
from config import CELL_SIZE, MAZE_WIDTH, MAZE_HEIGHT, BLACK


@pytest.fixture
def setup_renderer():
    """
    設置 Pygame 環境並創建 Renderer 實例。
    """
    pygame.init()
    screen = pygame.display.set_mode((MAZE_WIDTH * CELL_SIZE, MAZE_HEIGHT * CELL_SIZE))
    font = pygame.font.SysFont(None, 36)
    renderer = Renderer(screen, font, MAZE_WIDTH * CELL_SIZE, MAZE_HEIGHT * CELL_SIZE)
    yield renderer
    pygame.quit()


def test_render_initialization(setup_renderer):
    """
    測試 Renderer 的初始化和渲染功能，確保無異常並正確繪製遊戲元素。
    """
    renderer = setup_renderer
    game = Mock(spec=Game)

    # 模擬迷宮
    maze_mock = Mock()
    maze_mock.height = MAZE_HEIGHT
    maze_mock.width = MAZE_WIDTH
    maze_mock.get_tile.return_value = None
    game.get_maze.return_value = maze_mock

    # 模擬能量球和分數球
    game.get_power_pellets.return_value = []
    game.get_score_pellets.return_value = []

    # 模擬 Pac-Man
    pacman = Mock()
    pacman.configure_mock(
        current_x=50.0,
        current_y=50.0,
        x=2,
        y=2,
        target_x=3,
        target_y=2,
        score=100
    )
    game.get_pacman.return_value = pacman

    # 模擬鬼魂與狀態
    game.get_ghosts.return_value = []
    game.is_death_animation_playing.return_value = False
    game.get_lives.return_value = 3

    # 替代整個 screen 為 MagicMock 並 mock 所需方法
    mock_screen = MagicMock()
    renderer.screen = mock_screen

    # 模擬其他 pygame 函式
    with patch('pygame.draw.ellipse') as mock_ellipse, \
         patch('pygame.draw.rect') as mock_rect, \
         patch('pygame.image.load') as mock_load:

        mock_img = MagicMock()
        mock_load.return_value.convert_alpha.return_value = mock_img

        # 執行渲染
        renderer.render(game, "TestMode", frame_count=0)

        # 驗證行為
        mock_screen.fill.assert_called_once_with(BLACK)
        mock_screen.blit.assert_any_call(mock_img, ANY)
        assert mock_load.called
        assert mock_screen.blit.call_count >= 1
