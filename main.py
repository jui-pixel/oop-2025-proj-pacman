# main.py
"""
Pac-Man 遊戲的主程式，負責初始化遊戲、處理事件和運行主迴圈。
使用 Pygame 作為遊戲引擎，整合遊戲邏輯與 AI 控制策略。
"""
import sys
import pygame
from game.game import Game
from game.renderer import Renderer
from game.strategies import ControlManager
from config import MAZE_WIDTH, MAZE_HEIGHT, CELL_SIZE, FPS
from game.menu import show_menu, get_player_name, show_loading_screen, show_leaderboard, show_settings, show_pause_menu, show_game_result

# 初始化 Pygame
pygame.init()

def main():
    """
    主遊戲入口，負責設置遊戲環境、運行主迴圈和處理遊戲結束。
    添加初始選單介面，讓使用者選擇遊戲模式。
    AI 模式跳過名稱輸入，直接開始遊戲。
    支持暫停功能和重新開始遊戲。
    """
    # 設置螢幕尺寸
    screen_width = MAZE_WIDTH * CELL_SIZE
    screen_height = MAZE_HEIGHT * CELL_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pac-Man Game")

    # 設置遊戲時鐘和字體
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    # 顯示選單並獲取選擇
    mode = show_menu(screen, font, screen_width, screen_height)

    if mode == "exit":
        pygame.quit()
        sys.exit()
    elif mode == "leaderboard":
        show_leaderboard(screen, font, screen_width, screen_height)
        main()  # 返回選單
        return
    elif mode == "settings":
        show_settings(screen, font, screen_width, screen_height)
        main()  # 返回選單
        return

    # 根據模式設置名稱
    default_name = "Player" if mode == "player" else "RuleAI" if mode == "rule_ai" else "DQNAI"
    player_name = get_player_name(screen, font, screen_width, screen_height, default_name)

    # 顯示加載畫面
    show_loading_screen(screen, font, screen_width, screen_height)

    # 初始化遊戲實例和渲染器
    game = Game(player_name)
    renderer = Renderer(screen, font, screen_width, screen_height)

    # 初始化操控管理器，根據選單選擇設置初始模式
    control_manager = ControlManager(MAZE_WIDTH, MAZE_HEIGHT)
    if mode == "rule_ai":
        control_manager.current_strategy = control_manager.rule_based_ai
        print("Starting in Rule AI Mode")
    elif mode == "dqn_ai":
        if control_manager.dqn_ai:
            control_manager.current_strategy = control_manager.dqn_ai
            print("Starting in DQN AI Mode")
        else:
            control_manager.current_strategy = control_manager.rule_based_ai
            print("DQN AI unavailable, falling back to Rule AI Mode")

    frame_count = 0  # 用於動畫效果的計數器
    paused = False  # 暫停狀態標誌

    # 主遊戲迴圈
    while True:
        if not paused and game.is_running():
            frame_count += 1

            # 處理 Pygame 事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.end_game()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        paused = True
                control_manager.handle_event(event)  # 處理鍵盤輸入或切換模式

            # 更新遊戲狀態
            if not paused:
                game.update(FPS, lambda: control_manager.move(
                    game.get_pacman(), game.get_maze(), game.get_power_pellets(),
                    game.get_score_pellets(), game.get_ghosts()))

            # 渲染遊戲畫面
            renderer.render(game, control_manager.get_mode_name(), frame_count)

        elif paused:
            # 顯示暫停選單
            result = show_pause_menu(screen, font, screen_width, screen_height)
            if result == "continue":
                paused = False
            elif result == "menu":
                main()  # 返回主選單
            elif result == "exit":
                pygame.quit()
                sys.exit()

        pygame.display.flip()  # 更新畫面
        clock.tick(FPS)  # 控制幀率

        if not paused and not game.is_running():
            # 遊戲結束，儲存數據並顯示結果
            final_score = game.get_final_score()
            won = game.did_player_win()

            # 顯示輸贏畫面
            result = show_game_result(screen, font, screen_width, screen_height, won, final_score)

            if result == "restart":
                game = Game(player_name)  # 重新初始化遊戲
            elif result == "exit":
                pygame.quit()
                sys.exit()
            else:
                main()  # 返回主選單

if __name__ == "__main__":
    main()