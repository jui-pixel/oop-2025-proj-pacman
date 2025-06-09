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

    原理：
    - 初始化 Pygame 螢幕、時鐘和字體，設置遊戲視窗。
    - 顯示初始選單，讓使用者選擇遊戲模式（玩家、規則 AI、DQN AI、排行榜、設定、退出）。
    - 根據模式設置玩家名稱，顯示加載畫面，初始化遊戲實例、渲染器和控制管理器。
    - 運行主迴圈，處理事件、更新遊戲狀態、渲染畫面，支援暫停功能和重新開始遊戲。
    - 遊戲結束後儲存分數並顯示結果，提供返回選單、重啟或退出選項。
    - 螢幕尺寸計算公式：screen_width = MAZE_WIDTH * CELL_SIZE, screen_height = MAZE_HEIGHT * CELL_SIZE。
    """
    # 設置螢幕尺寸
    screen_width = MAZE_WIDTH * CELL_SIZE  # 螢幕寬度（像素）
    screen_height = MAZE_HEIGHT * CELL_SIZE  # 螢幕高度（像素）
    screen = pygame.display.set_mode((screen_width, screen_height))  # 創建遊戲視窗
    pygame.display.set_caption("Pac-Man Game")  # 設置視窗標題

    # 設置遊戲時鐘和字體
    clock = pygame.time.Clock()  # 控制遊戲幀率
    font = pygame.font.SysFont(None, 36)  # 使用系統字體，字號 36

    # 顯示選單並獲取選擇
    mode = show_menu(screen, font, screen_width, screen_height)  # 顯示主選單，返回選擇的模式

    if mode == "exit":
        pygame.quit()  # 退出 Pygame
        sys.exit()  # 終止程式
    elif mode == "leaderboard":
        show_leaderboard(screen, font, screen_width, screen_height)  # 顯示排行榜
        main()  # 遞迴調用，返回選單
        return
    elif mode == "settings":
        show_settings(screen, font, screen_width, screen_height)  # 顯示設定頁面
        main()  # 遞迴調用，返回選單
        return

    # 根據模式設置名稱
    default_name = "Player" if mode == "player" else "RuleAI" if mode == "rule_ai" else "DQNAI"  # 設置預設名稱
    player_name = get_player_name(screen, font, screen_width, screen_height, default_name)  # 獲取玩家名稱

    # 顯示加載畫面
    show_loading_screen(screen, font, screen_width, screen_height)  # 顯示加載畫面，延遲 1 秒

    # 初始化遊戲實例和渲染器
    game = Game(player_name)  # 創建遊戲實例，傳入玩家名稱
    renderer = Renderer(screen, font, screen_width, screen_height)  # 創建渲染器

    # 初始化操控管理器，根據選單選擇設置初始模式
    control_manager = ControlManager(MAZE_WIDTH, MAZE_HEIGHT)  # 創建控制管理器
    if mode == "rule_ai":
        control_manager.current_strategy = control_manager.rule_based_ai  # 設置為規則 AI 模式
        print("Starting in Rule AI Mode")
    elif mode == "dqn_ai":
        if control_manager.dqn_ai:
            control_manager.current_strategy = control_manager.dqn_ai  # 設置為 DQN AI 模式
            print("Starting in DQN AI Mode")
        else:
            control_manager.current_strategy = control_manager.rule_based_ai  # 若 DQN AI 不可用，回退到規則 AI
            print("DQN AI unavailable, falling back to Rule AI Mode")

    frame_count = 0  # 用於動畫效果的計數器（例如鬼魂閃爍）
    paused = False  # 暫停狀態標誌

    # 主遊戲迴圈
    while True:
        if not paused and (game.is_running() or game.is_death_animation_playing()):
            frame_count += 1  # 更新動畫計數器

            # 處理 Pygame 事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.end_game()  # 結束遊戲
                    pygame.quit()  # 退出 Pygame
                    sys.exit()  # 終止程式
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        paused = True  # 進入暫停狀態
                if not game.is_death_animation_playing():
                    control_manager.handle_event(event)  # 處理鍵盤輸入（僅玩家模式）或切換模式

            # 更新遊戲狀態
            if not paused:
                game.update(FPS, lambda: control_manager.move(
                    game.get_pacman(), game.get_maze(), game.get_power_pellets(),
                    game.get_score_pellets(), game.get_ghosts()))  # 更新遊戲狀態，執行移動

            # 渲染遊戲畫面
            renderer.render(game, control_manager.get_mode_name(), frame_count)  # 渲染當前畫面

        elif paused:
            # 顯示暫停選單
            result = show_pause_menu(screen, font, screen_width, screen_height)  # 顯示暫停選單
            if result == "continue":
                paused = False  # 繼續遊戲
            elif result == "menu":
                main()  # 返回主選單
            elif result == "exit":
                pygame.quit()  # 退出 Pygame
                sys.exit()  # 終止程式

        pygame.display.flip()  # 更新螢幕顯示
        clock.tick(FPS)  # 控制幀率為 FPS（每秒幀數）

        if not paused and not game.is_running() and not game.is_death_animation_playing():
            # 遊戲結束，儲存數據並顯示結果
            final_score = game.get_final_score()  # 獲取最終分數
            won = game.did_player_win()  # 判斷是否獲勝

            # 顯示輸贏畫面
            result = show_game_result(screen, font, screen_width, screen_height, won, final_score)  # 顯示遊戲結果

            if result == "restart":
                game = Game(player_name)  # 重新初始化遊戲，使用相同玩家名稱
            elif result == "exit":
                pygame.quit()  # 退出 Pygame
                sys.exit()  # 終止程式
            else:
                main()  # 返回主選單

if __name__ == "__main__":
    main()  # 執行主程式