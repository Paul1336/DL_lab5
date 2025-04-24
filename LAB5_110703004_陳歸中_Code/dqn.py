# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

from gymnasium.wrappers import TimeLimit
from gymnasium.envs.classic_control import CartPoleEnv
from gym.vector import AsyncVectorEnv

from model import DQN_task1, DQN_task2, init_weights

gym.register_envs(ale_py)

class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class ReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, prioritized=False, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.prioritized = prioritized

    def add(self, transition, error = 1):

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        if self.prioritized:
            self.priorities[self.pos] = (error + 1e-5) ** self.alpha
        self.pos = (self.pos + 1) % self.capacity
        return 
    def sample(self, batch_size):
        if self.prioritized:
            probs = self.priorities[:len(self.buffer)]
            probs /= probs.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        else:
            indices = np.random.choice(len(self.buffer), batch_size)

        samples = [self.buffer[idx] for idx in indices]

        if self.prioritized:
            total = len(self.buffer)
            probs = self.priorities[indices] / self.priorities[:total].sum()
            weights = (total * probs) ** (-self.beta)
            weights /= weights.max()
            # should maybe use globel IS_Weight max for normalization?
            weights = np.array(weights, dtype=np.float32)
        else:
            weights = np.ones(batch_size, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for id, error in zip(indices, errors):
            self.priorities[id] = (error + 1e-5) ** self.alpha
        return
    def __len__(self):
        return len(self.buffer)
        
def make_env(env_name):
    def _thunk():
        return gym.make(env_name, render_mode="rgb_array")
    return _thunk

class DQNAgent:
    def __init__(self, args=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        if args.num_envs > 1:
            self.env = AsyncVectorEnv([make_env(args.env_name) for _ in range(args.num_envs)])
            self.vector_env = True
        else:
            self.env = gym.make(args.env_name, render_mode="rgb_array")
            self.vector_env = False

        # self.env = TimeLimit(CartPoleEnv(render_mode="rgb_array"), max_episode_steps=1000)
        # self.test_env = TimeLimit(CartPoleEnv(render_mode="rgb_array"), max_episode_steps=1000)
        # self.env = gym.make(args.env_name, render_mode="rgb_array")
        self.test_env = gym.make(args.env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        # self.preprocessor = AtariPreprocessor()
        self.input_dim = self.env.observation_space.shape[0]

        if args.task == 1:
            self.q_net = DQN_task1(self.num_actions, self.input_dim).to(self.device)
            self.target_net = DQN_task1(self.num_actions, self.input_dim).to(self.device)
            self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
            self.preprocessor = None
        if args.task == 2:
            self.q_net = DQN_task2(self.num_actions).to(self.device)
            self.target_net = DQN_task2(self.num_actions).to(self.device)
            self.preprocessor = AtariPreprocessor()
            self.best_reward = -21
        
        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.memory = ReplayBuffer(args.memory_size)
        #self.memory = deque(maxlen = args.memory_size)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def preprocess(self, obs):
        if self.preprocessor:
            return self.preprocessor.reset(obs)
        return obs

    def run(self, episodes=10000):
        for ep in range(episodes):
            start_time = time.time()
            obs, _ = self.env.reset()
            if self.vector_env:
                state = np.stack([self.preprocess(o) for o in obs])
            else:
                state = self.preprocess(obs)
            
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if self.preprocessor:
                    next_state = self.preprocessor.step(next_obs)
                    #raise RuntimeError("err")
                else:
                    next_state = next_obs

                self.memory.add((state, action, reward, next_state, done))
                #self.memory.append((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            end_time = time.time()
            episode_time = end_time - start_time
            if ep % 100 == 0:
                print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f} Time: {episode_time:.2f}s")
            
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
                "Episode Time (sec)": episode_time
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 20 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")
                eval_reward = 0
                for i in range(args.eval_batch_size):
                    eval_reward += self.evaluate()
                eval_reward /= args.eval_batch_size
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        if self.preprocessor:
            state = self.preprocessor.reset(obs)
        else:
            state = obs
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if self.preprocessor:
                state = self.preprocessor.step(next_obs)
            else:
                state = next_obs

        return total_reward


    def train(self):
        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        batch, indices, weights = self.memory.sample(self.batch_size)
        #batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
      
            
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)
        #print(weights)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)
        
        td_errors = target_q - q_values
        loss = (weights * td_errors.pow(2)).mean()
        #loss = (td_errors.pow(2)).mean()
        wandb.log({
                    "TD errors": td_errors.abs().mean().item(),
                    "Loss": loss,
                })
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        #self.memory.update_priorities(indices, td_errors.abs().detach().cpu().numpy())
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--num-envs", type=int, default=4)
    #task1
    # parser.add_argument("--wandb-run-name", type=str, default="CartPole-run")
    # parser.add_argument("--task", type=int, default=1)
    # parser.add_argument("--env-name", type=str, default="CartPole-v1")
    
    # parser.add_argument("--train-per-step", type=int, default=1)
    # parser.add_argument("--batch-size", type=int, default=64)
    # parser.add_argument("--memory-size", type=int, default=5000)
    # parser.add_argument("--lr", type=float, default=0.00005)
    # parser.add_argument("--discount-factor", type=float, default=0.99)
    # parser.add_argument("--epsilon-start", type=float, default=1.0)
    # parser.add_argument("--epsilon-decay", type=float, default=0.99995)
    # parser.add_argument("--epsilon-min", type=float, default=0.01)
    # parser.add_argument("--target-update-frequency", type=int, default=1500)
    # parser.add_argument("--replay-start-size", type=int, default=1000)
    # parser.add_argument("--max-episode-steps", type=int, default=1000)
    # parser.add_argument("--eval-batch-size", type=int, default=10)

    #task2
    parser.add_argument("--wandb-run-name", type=str, default="ALE/Pong-run")
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5")

    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.000025)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999974)
    parser.add_argument("--epsilon-min", type=float, default=0.1)
    parser.add_argument("--target-update-frequency", type=int, default=10000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=float('inf'))
    parser.add_argument("--eval-batch-size", type=int, default=10)
    


    args = parser.parse_args()
    if args.task == 1:
        wandb.init(project="DLP-Lab5-DQN-CartPole", save_code=True)
    if args.task == 2:
        wandb.init(project="DLP-Lab5-DQN-Pong-v5-task2", save_code=True)
    agent = DQNAgent(args=args)
    agent.run(10000)