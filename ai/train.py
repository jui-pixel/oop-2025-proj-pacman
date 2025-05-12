# ai/train.py
import torch
from game.environment import PacManEnv
from ai.agent import DQNAgent

def train():
    env = PacManEnv()
    state_dim = (env.maze.w, env.maze.h, 5)
    action_dim = len(env.action_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim, action_dim, device)

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train(env)
            state = next_state
            total_reward += reward

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
            agent.save(f"pacman_dqn_{episode}.pth")

    agent.save("pacman_dqn_final.pth")

if __name__ == "__main__":
    train()