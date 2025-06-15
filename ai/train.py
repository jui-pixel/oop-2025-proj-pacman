# ai/train.py
# 導入必要的 Python 模組
import os  # 用於處理檔案路徑和檔案操作
import argparse  # 用於解析命令行參數
import numpy as np  # 用於數值計算和陣列操作
import torch  # PyTorch 深度學習框架
import json  # 用於儲存和讀取 JSON 格式的數據
from torch.utils.tensorboard import SummaryWriter  # 用於記錄訓練過程並可視化
from environment import PacManEnv  # 匯入 Pac-Man 遊戲環境
from agent import DQNAgent  # 匯入 DQN 代理類
from config import *  # 匯入所有設定參數（如學習率、迷宮大小等）
import random  # 用於隨機選擇動作或生成隨機數
import optuna  # 用於超參數優化

def collect_expert_data(env, agent, num_episodes=EXPERT_EPISODES, max_steps_per_episode=EXPERT_MAX_STEPS_PER_EPISODE, expert_random_prob=EXPERT_RANDOM_PROB, max_expert_data=MAX_EXPERT_DATA):
    """
    收集專家數據用於預訓練。

    這個函數模擬一個基於規則的專家 AI，收集其在遊戲中的狀態和動作，用於初始化 DQN 代理的學習。

    Args:
        env (PacManEnv): Pac-Man 遊戲環境。
        agent (DQNAgent): DQN 代理，用於儲存轉換數據。
        num_episodes (int): 收集專家數據的回合數，預設從 config 取得。
        max_steps_per_episode (int): 每個回合的最大步數，預設從 config 取得。
        expert_random_prob (float): 專家採取隨機動作的概率，預設從 config 取得。
        max_expert_data (int): 最大專家數據量，預設從 config 取得。

    Returns:
        list: 包含 (狀態, 動作) 元組的專家數據列表。
    """
    # 初始化專家數據列表
    expert_data = []
    # 遍歷指定的回合數
    for episode in range(num_episodes):
        # 如果數據量已達上限，則停止收集
        if len(expert_data) >= max_expert_data:
            break
        # 重置環境，使用隨機出生點種與隨機地圖以增加多樣性
        state, _ = env.reset(random_spawn_seed=episode, random_maze_seed=episode)
        # 初始化回合結束標誌和步數計數器
        done = False
        steps = 0
        # 在回合未結束且步數未達上限時繼續
        while not done and steps < max_steps_per_episode:
            # 以一定概率選擇隨機動作，增加數據多樣性
            if random.random() < expert_random_prob:
                action = np.random.randint(0, env.action_space.n)
                # print("random")
            else:
                # 否則使用環境提供的專家動作（基於規則的 AI）
                action = env.get_expert_action()
                # print("expert")
            # 執行動作，獲取下一個狀態、獎勵、結束標誌和資訊
            next_state, reward, done, info = env.step(action)
            # print(info["valid_step"])
            # print(done)
            # 如果這一步是有效的（即成功移動且未終結）
            if info.get('valid_step', False):
                # 將狀態和動作記錄到專家數據中
                expert_data.append((state, action))
                # 將轉換數據儲存到代理的回放緩衝區
                agent.store_transition(state, action, reward, next_state, done)
            # 更新當前狀態
            state = next_state
            # 增加步數
            steps += 1
        # 印出回合資訊，方便追蹤進度
        print(f"專家回合 {episode + 1}/{num_episodes}，步數：{steps}，數據量：{len(expert_data)}")
    # 返回最多 max_expert_data 筆數據
    return expert_data[:max_expert_data]

def train(trial=None, resume=False,
    model_path=MODEL_PATH, memory_path=MEMORY_PATH, episodes=TRAIN_EPISODES,
    early_stop_reward=EARLY_STOP_REWARD, pretrain_episodes=PRETRAIN_EPISODES,
    lr=LEARNING_RATE, batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
    sigma=SIGMA, n_step=N_STEP, gamma=GAMMA, alpha=ALPHA, beta=BETA,
    beta_increment=BETA_INCREMENT, expert_prob_start=EXPERT_PROB_START,
    expert_prob_end=EXPERT_PROB_END, expert_prob_decay_steps=EXPERT_PROB_DECAY_STEPS,
    expert_random_prob=EXPERT_RANDOM_PROB, max_expert_data=MAX_EXPERT_DATA, ghost_penalty_weight=GHOST_PENALTY_WEIGHT):
    """
    訓練 DQN 代理，支援 Optuna 超參數優化。

    這個函數負責執行完整的 DQN 訓練流程，包括專家數據預訓練、環境交互、模型學習和結果記錄。

    Args:
        trial (optuna.Trial, optional): Optuna 試驗對象，用於超參數優化。
        resume (bool): 是否從現有模型繼續訓練。
        model_path (str): 模型儲存/載入路徑。
        memory_path (str): 回放緩衝區儲存/載入路徑。
        episodes (int): 訓練回合數。
        early_stop_reward (float): 早期停止的獎勵閾值。
        pretrain_episodes (int): 預訓練回合數。
        lr (float): 學習率。
        batch_size (int): 批量大小。
        target_update_freq (int): 目標網絡更新頻率。
        sigma (float): Noisy DQN 的噪聲因子。
        n_step (int): N 步回報的步數。
        gamma (float): 折扣因子。
        alpha (float): 優先回放的 alpha 參數。
        beta (float): 優先回放的 beta 參數。
        beta_increment (float): beta 每次增加的值。
        expert_prob_start (float): 專家動作概率的起始值。
        expert_prob_end (float): 專家動作概率的終止值。
        expert_prob_decay_steps (int): 專家概率衰減的步數。
        expert_random_prob (float): 專家隨機動作概率。
        max_expert_data (int): 最大專家數據量。
        ghost_penalty_weight (float): 鬼魂懲罰權重。

    Returns:
        float: 訓練的最終評估分數（平均獎勵 + 鬼魂距離加成 - 生命損失懲罰）。
    """
    # 如果使用 Optuna，則從試驗中建議超參數
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) if trial else lr
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32) if trial else batch_size
    target_update_freq = trial.suggest_int("target_update_freq", 5, 50) if trial else target_update_freq
    sigma = trial.suggest_float("sigma", 0.1, 2.0) if trial else sigma
    n_step = trial.suggest_int("n_step", 3, 10) if trial else n_step
    gamma = trial.suggest_float("gamma", 0.9, 0.99) if trial else gamma
    alpha = trial.suggest_float("alpha", 0.6, 1.0) if trial else alpha
    beta = trial.suggest_float("beta", 0.4, 0.8) if trial else beta
    beta_increment = trial.suggest_float("beta_increment", 1e-5, 1e-2, log=True) if trial else beta_increment
    expert_prob_start = trial.suggest_float("expert_prob_start", 0.2, 0.5) if trial else expert_prob_start
    expert_prob_end = trial.suggest_float("expert_prob_end", 0.01, 0.1) if trial else expert_prob_end
    expert_prob_decay_steps = trial.suggest_int("expert_prob_decay_steps", 100000, 1000000) if trial else expert_prob_decay_steps
    expert_random_prob = trial.suggest_float("expert_random_prob", 0.05, 0.2) if trial else expert_random_prob
    max_expert_data = trial.suggest_int("max_expert_data", 5000, 20000) if trial else max_expert_data
    ghost_penalty_weight = trial.suggest_float("ghost_penalty_weight", 2.0, 10.0) if trial else ghost_penalty_weight

    # 驗證參數的有效性，確保所有參數在合理範圍內
    for param, valid, name, desc in [
        (lr, lr > 0, "學習率 (lr)", "大於 0"),
        (batch_size, batch_size > 0, "批量大小 (batch_size)", "大於 0"),
        (target_update_freq, target_update_freq > 0, "目標更新頻率", "大於 0"),
        (sigma, sigma > 0, "Noisy因子 (sigma)", "大於 0"),
        (n_step, n_step > 0, "n-step 數", "大於 0"),
        (gamma, 0 <= gamma < 1, "折扣因子 (gamma)", "[0, 1)"),
        (expert_prob_start, 0 <= expert_prob_start <= 1, "專家概率起點 (expert_prob_start)", "[0, 1]"),
        (expert_prob_end, 0 <= expert_prob_end <= 1, "專家概率終點 (expert_prob_end)", "[0, 1]"),
        (expert_prob_decay_steps, expert_prob_decay_steps > 0, "專家概率衰減步數", "大於 0"),
        (expert_random_prob, 0 <= expert_random_prob <= 1, "專家隨機概率", "[0, 1]"),
        (ghost_penalty_weight, ghost_penalty_weight > 0, "鬼魂懲罰權重", "大於 0"),
    ]:
        if not valid:
            raise ValueError(f"{name} 無效，必須 {desc}")

    # 選擇訓練設備（GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"訓練設備：{device}")
    # 印出所有訓練參數，方便除錯和追蹤
    print(f"訓練參數：lr={lr:.6e}, batch_size={batch_size}, target_update_freq={target_update_freq}, "
          f"sigma={sigma:.2f}, n_step={n_step}, gamma={gamma:.2f}, alpha={alpha:.2f}, beta={beta:.2f}, "
          f"beta_increment={beta_increment:.6e}, expert_prob_start={expert_prob_start:.2f}, "
          f"expert_prob_end={expert_prob_end:.2f}, expert_prob_decay_steps={expert_prob_decay_steps:,}, "
          f"expert_random_prob={expert_random_prob:.2f}, max_expert_data={max_expert_data:,}, "
          f"ghost_penalty_weight={ghost_penalty_weight:.2f}")

    # 初始化 Pac-Man 環境
    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED, 
                    ghost_penalty_weight=ghost_penalty_weight)
    # 獲取狀態和動作維度
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    # 初始化 DQN 代理
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        target_update_freq=target_update_freq,
        n_step=n_step,
        alpha=alpha,
        beta=beta,
        beta_increment=beta_increment,
        expert_prob_start=expert_prob_start,
        expert_prob_end=expert_prob_end,
        expert_prob_decay_steps=expert_prob_decay_steps,
        sigma=sigma
    )

    # 如果選擇繼續訓練且模型檔案存在，則載入模型和回放緩衝區
    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        print(f"從 {model_path} 載入模型")

    # 如果不是繼續訓練，則進行專家數據預訓練
    if not resume:
        print(f"收集 {pretrain_episodes} 回合的專家數據...")
        expert_data = collect_expert_data(
            env, agent, pretrain_episodes, max_steps_per_episode=2000,
            expert_random_prob=expert_random_prob, max_expert_data=max_expert_data)
        # 使用專家數據進行預訓練
        agent.pretrain(expert_data, pretrain_steps=100000)

    # 初始化 TensorBoard 記錄器
    writer = SummaryWriter()
    # 初始化用於記錄訓練結果的列表
    episode_rewards = []
    recent_rewards = []
    avg_ghost_distances = []
    ghost_encounters = []
    lives_lost_list = []

    # 開始訓練循環
    for episode in range(episodes):
        # 初始化回合數據
        total_reward = 0
        steps = 0
        lives_lost = 0
        total_ghost_dist = 0
        encounter_count = 0
        done = False
        # 重置環境，使用隨機出生點種子+隨機地圖
        state, _ = env.reset(random_spawn_seed=episode, random_maze_seed=episode)
        # 重置模型的噪聲（Noisy DQN）
        agent.model.reset_noise()
        # 初始化動作計數器
        action_counts = np.zeros(action_dim)
        # 初始化 Q 值記錄列表
        q_values_list = []
        # 在回合未結束時繼續
        while not done:
            # 以專家概率選擇專家動作，否則使用代理選擇動作
            if random.random() < agent.expert_prob:
                action = env.get_expert_action()
                expert_action = True
            else:
                action = agent.choose_action(state)
                expert_action = False
            # 記錄動作使用次數
            action_counts[action] += 1
            # 計算當前狀態的 Q 值
            q_values, noise_metrics = agent.model(torch.FloatTensor(state).unsqueeze(0).to(device))
            q_values_list.append(q_values.detach().cpu().numpy().mean())
            # 計算與鬼魂的距離
            ghost_distances = []
            for i in range(3, 5):  # 通道 3 和 4 分別表示可食用鬼魂和普通鬼魂
                if i < len(state) and np.any(state[i]):
                    # 找出 Pac-Man 的坐標
                    pacman_x = np.argmax(state[0].max(axis=1))
                    pacman_y = np.argmax(state[0].max(axis=0))
                    # 找出鬼魂的坐標
                    ghost_x = np.argmax(state[i].max(axis=1))
                    ghost_y = np.argmax(state[i].max(axis=0))
                    # 計算歐氏距離
                    dist = np.sqrt((pacman_x - ghost_x)**2 + (pacman_y - ghost_y)**2)
                    ghost_distances.append(dist)
            # 如果有鬼魂，則取最小距離；否則使用迷宮最大距離
            min_ghost_dist = min(ghost_distances) if ghost_distances else MAZE_WIDTH + MAZE_HEIGHT
            total_ghost_dist += min_ghost_dist
            # 如果與鬼魂距離小於 2，記錄為一次遭遇
            if min_ghost_dist < 2.0:
                encounter_count += 1
            # 記錄 Noisy DQN 層的噪聲指標
            writer.add_scalar('FC1_Weight_Sigma', noise_metrics['fc1_weight_sigma_mean'], agent.steps)
            writer.add_scalar('FC1_Bias_Sigma', noise_metrics['fc1_bias_sigma_mean'], agent.steps)
            writer.add_scalar('Value_Weight_Sigma', noise_metrics['value_weight_sigma_mean'], agent.steps)
            writer.add_scalar('Value_Bias_Sigma', noise_metrics['value_bias_sigma_mean'], agent.steps)
            writer.add_scalar('Advantage_Weight_Sigma', noise_metrics['advantage_weight_sigma_mean'], agent.steps)
            writer.add_scalar('Advantage_Bias_Sigma', noise_metrics['advantage_bias_sigma_mean'], agent.steps)
            # 執行動作
            next_state, reward, done, info = env.step(action)
            # 如果這一步有效，則儲存轉換並進行學習
            if info.get('valid_step', False):
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.learn(expert_action=expert_action)
                if loss is not None:
                    # 記錄損失值
                    writer.add_scalar('Loss', loss, agent.steps)
                total_reward += reward
                steps += 1
            # 如果失去生命，記錄一次
            if info.get('lives_lost', False):
                lives_lost += 1
            # 更新狀態
            state = next_state
        # 記錄回合數據
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        avg_ghost_distances.append(total_ghost_dist / max(steps, 1))
        ghost_encounters.append(encounter_count)
        lives_lost_list.append(lives_lost)
        # 保持最近 100 回合的數據
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            avg_ghost_distances.pop(0)
            ghost_encounters.pop(0)
            lives_lost_list.pop(0)
        # 記錄各種指標到 TensorBoard
        writer.add_scalar('Mean_Q_Value', np.mean(q_values_list) if q_values_list else 0, episode)
        writer.add_scalar('Lives_Lost', lives_lost, episode)
        writer.add_scalar('Avg_Ghost_Distance', avg_ghost_distances[-1], episode)
        writer.add_scalar('Ghost_Encounters', encounter_count, episode)
        for i in range(action_dim):
            writer.add_scalar(f'Action_{i}_Ratio', action_counts[i] / max(steps, 1), episode)
        # 印出回合資訊
        print(f"回合 {episode + 1}/{episodes}, 獎勵：{total_reward:.2f}, 步數：{steps}, "
              f"專家概率：{agent.expert_prob:.2f}, 平均 Q 值：{np.mean(q_values_list):.2f}, "
              f"平均鬼距離：{avg_ghost_distances[-1]:.2f}, 鬼遭遇：{encounter_count}, "
              f"生命損失：{lives_lost}")
        writer.add_scalar('Reward', total_reward, episode)
        writer.add_scalar('Expert_Probability', agent.expert_prob, episode)
        # 每 5 回合儲存模型
        if (episode + 1) % 1 == 0:
            agent.save(model_path, memory_path)
            print(f"回合 {episode + 1} 保存模型")
        # 如果使用 Optuna 且回合數超過 50，則報告中間結果並檢查是否需要剪枝
        if trial and episode >= 50:
            avg_reward = np.mean(recent_rewards[-100:])
            avg_ghost_dist = np.mean(avg_ghost_distances[-100:])
            avg_lives_lost = np.mean(lives_lost_list[-100:])
            trial.report(avg_reward + 10 * avg_ghost_dist - 50 * avg_lives_lost, episode)
            if trial.should_prune():
                writer.close()
                env.close()
                raise optuna.TrialPruned()
        # 如果最近 100 回合平均獎勵達到早期停止閾值，則停止訓練
        if len(recent_rewards) >= 100 and np.mean(recent_rewards[-100:]) >= early_stop_reward:
            print(f"早期停止：最近 100 回合平均獎勵 {np.mean(recent_rewards[-100:]):.2f} >= {early_stop_reward:.2f}")
            break
    # 儲存最終模型和回放緩衝區
    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    # 將回合獎勵儲存為 JSON 檔案
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)
    # 關閉 TensorBoard 記錄器和環境
    writer.close()
    env.close()
    print("訓練完成")
    # 返回最終評估分數
    return np.mean(recent_rewards[-100:]) + np.mean(avg_ghost_distances[-100:]) * 10 - np.mean(lives_lost_list[-100:]) * 50 if recent_rewards else total_reward

def objective(trial):
    """
    Optuna 優化目標函數。

    這個函數定義了 Optuna 的優化目標，通過調用 train 函數進行試驗。

    Args:
        trial (optuna.Trial): Optuna 試驗對象。

    Returns:
        float: 試驗的評估分數。
    """
    return train(trial=trial, episodes=500, early_stop_reward=10000)

if __name__ == "__main__":
    # 定義命令行參數解析器
    parser = argparse.ArgumentParser(description="Train Pac-Man DQN Agent")
    parser.add_argument('--resume', action='store_true', help='Resume training from previous model')
    parser.add_argument('--optuna', action='store_true', help='Use Optuna for hyperparameter optimization')
    # 訓練設置
    parser.add_argument('--episodes', type=int, default=TRAIN_EPISODES, help='Number of training episodes')
    parser.add_argument('--pretrain_episodes', type=int, default=PRETRAIN_EPISODES, help='Number of pretraining episodes')
    parser.add_argument('--early_stop_reward', type=float, default=EARLY_STOP_REWARD, help='Early stopping reward threshold')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to save/load model')
    parser.add_argument('--memory_path', type=str, default=MEMORY_PATH, help='Path to save/load replay buffer')
    # DQN 模型參數
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--target_update_freq', type=int, default=TARGET_UPDATE_FREQ, help='Target network update frequency')
    parser.add_argument('--sigma', type=float, default=SIGMA, help='Noisy DQN sigma factor')
    parser.add_argument('--n_step', type=int, default=N_STEP, help='N-step return')
    parser.add_argument('--gamma', type=float, default=GAMMA, help='Discount factor')
    parser.add_argument('--alpha', type=float, default=ALPHA, help='Prioritized replay alpha')
    parser.add_argument('--beta', type=float, default=BETA, help='Prioritized replay beta')
    parser.add_argument('--beta_increment', type=float, default=BETA_INCREMENT, help='Beta increment per step')
    parser.add_argument('--expert_prob_start', type=float, default=EXPERT_PROB_START, help='Starting expert probability')
    parser.add_argument('--expert_prob_end', type=float, default=EXPERT_PROB_END, help='Ending expert probability')
    parser.add_argument('--expert_prob_decay_steps', type=int, default=EXPERT_PROB_DECAY_STEPS, help='Expert probability decay steps')
    parser.add_argument('--ghost_penalty_weight', type=float, default=GHOST_PENALTY_WEIGHT, help='Ghost penalty weight')
    # 專家數據收集參數
    parser.add_argument('--expert_episodes', type=int, default=EXPERT_EPISODES, help='Number of expert data collection episodes')
    parser.add_argument('--expert_max_steps_per_episode', type=int, default=EXPERT_MAX_STEPS_PER_EPISODE, help='Max steps per expert episode')
    parser.add_argument('--expert_random_prob', type=float, default=EXPERT_RANDOM_PROB, help='Expert random action probability')
    parser.add_argument('--max_expert_data', type=int, default=MAX_EXPERT_DATA, help='Maximum expert data size')

    # 解析命令行參數
    args = parser.parse_args()
    # 如果啟用 Optuna，則執行超參數優化
    if args.optuna:
        # 創建 Optuna 研究，目標是最大化評估分數
        study = optuna.create_study(direction="maximize", storage="sqlite:///optuna.db")
        # 執行 50 次試驗
        study.optimize(objective, n_trials=50)
        # 印出最佳試驗的參數和分數
        print("最佳試驗：", study.best_trial.params)
        print(f"最佳值：{study.best_value:.2f}")
    else:
        # 否則執行單次訓練
        train(
            resume=args.resume,
            model_path=args.model_path,
            memory_path=args.memory_path,
            episodes=args.episodes,
            early_stop_reward=args.early_stop_reward,
            pretrain_episodes=args.pretrain_episodes,
            lr=args.lr,
            batch_size=args.batch_size,
            target_update_freq=args.target_update_freq,
            sigma=args.sigma,
            n_step=args.n_step,
            gamma=args.gamma,
            alpha=args.alpha,
            beta=args.beta,
            beta_increment=args.beta_increment,
            expert_prob_start=args.expert_prob_start,
            expert_prob_end=args.expert_prob_end,
            expert_prob_decay_steps=args.expert_prob_decay_steps,
            expert_random_prob=args.expert_random_prob,
            max_expert_data=args.max_expert_data,
            ghost_penalty_weight=args.ghost_penalty_weight
        )