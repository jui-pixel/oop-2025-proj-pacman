# ai/train.py
import os
import argparse
import numpy as np
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from environment import PacManEnv
from agent import DQNAgent
from config import *
import random
import optuna

def collect_expert_data(env, agent, num_episodes=EXPERT_EPISODES, max_steps_per_episode=EXPERT_MAX_STEPS_PER_EPISODE, expert_random_prob=EXPERT_RANDOM_PROB, max_expert_data=MAX_EXPERT_DATA):
    """
    收集專家數據用於預訓練。
    """
    expert_data = []
    for episode in range(num_episodes):
        if len(expert_data) >= max_expert_data:
            break
        state, _ = env.reset(random_spawn_seed=episode)
        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            if random.random() < expert_random_prob:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = env.get_expert_action()
            next_state, reward, done, info = env.step(action)
            # done = terminated or truncated
            if info.get('valid_step', False):
                expert_data.append((state, action))
                agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            steps += 1
        print(f"專家回合 {episode + 1}/{num_episodes}，步數：{steps}，數據量：{len(expert_data)}")
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
    """
    # Optuna 超參數建議（若啟用）
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

    # 參數驗證
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"訓練設備：{device}")
    print(f"訓練參數：lr={lr:.6e}, batch_size={batch_size}, target_update_freq={target_update_freq}, "
          f"sigma={sigma:.2f}, n_step={n_step}, gamma={gamma:.2f}, alpha={alpha:.2f}, beta={beta:.2f}, "
          f"beta_increment={beta_increment:.6e}, expert_prob_start={expert_prob_start:.2f}, "
          f"expert_prob_end={expert_prob_end:.2f}, expert_prob_decay_steps={expert_prob_decay_steps:,}, "
          f"expert_random_prob={expert_random_prob:.2f}, max_expert_data={max_expert_data:,}, "
          f"ghost_penalty_weight={ghost_penalty_weight:.2f}")

    env = PacManEnv(width=MAZE_WIDTH, height=MAZE_HEIGHT, seed=MAZE_SEED, 
                    ghost_penalty_weight=ghost_penalty_weight)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
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

    if resume and os.path.exists(model_path):
        agent.load(model_path, memory_path)
        print(f"從 {model_path} 載入模型")

    if not resume:
        print(f"收集 {pretrain_episodes} 回合的專家數據...")
        expert_data = collect_expert_data(
            env, agent, pretrain_episodes, max_steps_per_episode=200,
            expert_random_prob=expert_random_prob, max_expert_data=max_expert_data)
        agent.pretrain(expert_data, pretrain_steps=1000)

    writer = SummaryWriter()
    episode_rewards = []
    recent_rewards = []
    avg_ghost_distances = []
    ghost_encounters = []
    lives_lost_list = []

    for episode in range(episodes):
        total_reward = 0
        steps = 0
        lives_lost = 0
        total_ghost_dist = 0
        encounter_count = 0
        done = False
        state, _ = env.reset(random_spawn_seed=episode)
        agent.model.reset_noise()
        action_counts = np.zeros(action_dim)
        q_values_list = []
        while not done:
            if random.random() < agent.expert_prob:
                action = env.get_expert_action()
                expert_action = True
            else:
                action = agent.choose_action(state)
                expert_action = False
            action_counts[action] += 1
            q_values, noise_metrics = agent.model(torch.FloatTensor(state).unsqueeze(0).to(device))
            q_values_list.append(q_values.detach().cpu().numpy().mean())
            # 計算鬼魂距離
            min_ghost_dist = np.min([
                np.sqrt(
                    (np.argmax(state[0].max(axis=1)) - np.argmax(g, axis=1))**2 +
                    (np.argmax(state[0].max(axis=0)) - np.argmax(g, axis=0))**2
                )
                for g in state[3:5] if np.any(g)
            ])
            total_ghost_dist += min_ghost_dist
            if min_ghost_dist < 2.0:
                encounter_count += 1
            writer.add_scalar('FC1_Weight_Sigma', noise_metrics['fc1_weight_sigma_mean'], agent.steps)
            writer.add_scalar('FC1_Bias_Sigma', noise_metrics['fc1_bias_sigma_mean'], agent.steps)
            writer.add_scalar('Value_Weight_Sigma', noise_metrics['value_weight_sigma_mean'], agent.steps)
            writer.add_scalar('Value_Bias_Sigma', noise_metrics['value_bias_sigma_mean'], agent.steps)
            writer.add_scalar('Advantage_Weight_Sigma', noise_metrics['advantage_weight_sigma_mean'], agent.steps)
            writer.add_scalar('Advantage_Bias_Sigma', noise_metrics['advantage_bias_sigma_mean'], agent.steps)
            next_state, reward, done, info = env.step(action)
            # done = terminated or truncated
            if info.get('valid_step', False):
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.learn(expert_action=expert_action)
                if loss is not None:
                    writer.add_scalar('Loss', loss, agent.steps)
                total_reward += reward
                steps += 1
            if info.get('lives_lost', False):
                lives_lost += 1
            state = next_state
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        avg_ghost_distances.append(total_ghost_dist / max(steps, 1))
        ghost_encounters.append(encounter_count)
        lives_lost_list.append(lives_lost)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            avg_ghost_distances.pop(0)
            ghost_encounters.pop(0)
            lives_lost_list.pop(0)
        writer.add_scalar('Mean_Q_Value', np.mean(q_values_list) if q_values_list else 0, episode)
        writer.add_scalar('Lives_Lost', lives_lost, episode)
        writer.add_scalar('Avg_Ghost_Distance', avg_ghost_distances[-1], episode)
        writer.add_scalar('Ghost_Encounters', encounter_count, episode)
        for i in range(action_dim):
            writer.add_scalar(f'Action_{i}_Ratio', action_counts[i] / max(steps, 1), episode)
        print(f"回合 {episode + 1}/{episodes}, 獎勵：{total_reward:.2f}, 步數：{steps}, "
              f"專家概率：{agent.expert_prob:.2f}, 平均 Q 值：{np.mean(q_values_list):.2f}, "
              f"平均鬼距離：{avg_ghost_distances[-1]:.2f}, 鬼遭遇：{encounter_count}, "
              f"生命損失：{lives_lost}")
        writer.add_scalar('Reward', total_reward, episode)
        writer.add_scalar('Expert_Probability', agent.expert_prob, episode)
        if (episode + 1) % 5 == 0:
            agent.save(model_path, memory_path)
            print(f"回合 {episode + 1} 保存模型")
        if trial and episode >= 50:
            avg_reward = np.mean(recent_rewards[-100:])
            avg_ghost_dist = np.mean(avg_ghost_distances[-100:])
            avg_lives_lost = np.mean(lives_lost_list[-100:])
            trial.report(avg_reward + 10 * avg_ghost_dist - 50 * avg_lives_lost, episode)
            if trial.should_prune():
                writer.close()
                env.close()
                raise optuna.TrialPruned()
        if len(recent_rewards) >= 100 and np.mean(recent_rewards[-100:]) >= early_stop_reward:
            print( f"早期停止：最近 100 回合平均獎勵 {np.mean(recent_rewards[-100:]):.2f} >= {early_stop_reward:.2f}" )
            break
    agent.save("pacman_dqn_final.pth", "replay_buffer_final.pkl")
    with open("episode_rewards.json", "w") as f:
        json.dump(episode_rewards, f)
    writer.close()
    env.close()
    print("訓練完成")
    return np.mean(recent_rewards[-100:]) + np.mean(avg_ghost_distances[-100:]) * 10 - np.mean(lives_lost_list[-100:]) * 50 if recent_rewards else total_reward

def objective(trial):
    """
    Optuna 優化目標函數。
    """
    return train(trial=trial, episodes=500, early_stop_reward=10000)

if __name__ == "__main__":
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

    args = parser.parse_args()
    if args.optuna:
        study = optuna.create_study(direction="maximize", storage="sqlite:///optuna.db")
        study.optimize(objective, n_trials=50)
        print("最佳試驗：", study.best_trial.params)
        print(f"最佳值：{study.best_value:.2f}")
    else:
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