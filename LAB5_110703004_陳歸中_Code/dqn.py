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
from gymnasium.vector import AsyncVectorEnv

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
        probs = self.priorities[:len(self.buffer)]
        probs /= probs.sum()
        if self.prioritized:
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            weights = (len(self.buffer) *probs[indices]) ** (-self.beta)
            weights /= weights.max()
            weights = np.array(weights, dtype=np.float32)
        else:
            indices = np.random.choice(len(self.buffer), batch_size)
            weights = np.ones(batch_size, dtype=np.float32)

        samples = [self.buffer[idx] for idx in indices]

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for id, error in zip(indices, errors):
            self.priorities[id] = (abs(error) + 1e-5) ** self.alpha
        return
    def __len__(self):
        return  len(self.buffer)
        
def make_env(env_name):
    def _thunk():
        return gym.make(env_name, render_mode="rgb_array")
    return _thunk

class DQNAgent:
    def __init__(self, args=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        # self.env = TimeLimit(CartPoleEnv(render_mode="rgb_array"), max_episode_steps=1000)
        # self.test_env = TimeLimit(CartPoleEnv(render_mode="rgb_array"), max_episode_steps=1000)

        if args.num_envs > 1:
            self.env = AsyncVectorEnv([make_env(args.env_name) for _ in range(args.num_envs)])
            self.vector_env = True
            self.num_actions = self.env.single_action_space.n
        else:
            self.env = gym.make(args.env_name, render_mode="rgb_array")
            self.vector_env = False
            self.num_actions = self.env.action_space.n

        # self.env = gym.make(args.env_name, render_mode="rgb_array")
        self.test_env = gym.make(args.env_name, render_mode="rgb_array")
        
        # self.preprocessor = AtariPreprocessor()
        

        if args.task == 1:
            self.input_dim = self.env.observation_space.shape[0]
            self.q_net = DQN_task1(self.num_actions, self.input_dim).to(self.device)
            self.target_net = DQN_task1(self.num_actions, self.input_dim).to(self.device)
            self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
            self.preprocessor = None
        if args.task == 2:
            self.q_net = DQN_task2(self.num_actions).to(self.device)
            self.target_net = DQN_task2(self.num_actions).to(self.device)
            if args.num_envs > 1:
                self.preprocessor = [AtariPreprocessor(args.frame_stack) for _ in range(args.num_envs)]
            else: 
                self.preprocessor = AtariPreprocessor(args.frame_stack)
            self.best_reward = -21

        self.preprocessor_eval = AtariPreprocessor(args.frame_stack)
        
        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.scheduler_step, gamma = args.scheduler_gamma)
        self.memory = ReplayBuffer(args.memory_size)
        #self.memory = deque(maxlen = args.memory_size)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_step_count = 0
        self.train_count = 0
        
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        if args.clip>0:
            self.clip = True
        else:
            self.clip = False

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()


    def vec_run(self, episodes=10000, total_steps_target=2000):
        total_steps_target *= self.env.num_envs

        for ep in range(episodes):
            start_time = time.time()
            obs, _ = self.env.reset()

            if self.preprocessor:
                states = np.stack([self.preprocessor[i].reset(o) for i, o in enumerate(obs)])
            else:
                states = obs

            episode_rewards = np.zeros(self.env.num_envs)
            episode_step_count = 0

            while episode_step_count < total_steps_target:
                state_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    q_values = self.q_net(state_tensor)
                    actions = q_values.argmax(dim=1).cpu().numpy()

                next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
                dones = np.logical_or(terminated, truncated)

                if self.preprocessor:
                    next_states = np.stack([self.preprocessor[i].step(o) for i, o in enumerate(next_obs)])
                else:
                    next_states = next_obs

                for i in range(self.env.num_envs):
                    self.memory.add((states[i], actions[i], rewards[i], next_states[i], dones[i]))
                    episode_rewards[i] += rewards[i]

                    if dones[i]:
                        obs_reset, _ = self.env.reset()
                        if self.preprocessor:
                            next_states[i] = self.preprocessor[i].reset(obs_reset[i])
                        else:
                            next_states[i] = obs_reset[i]

                for _ in range(self.train_per_step):
                    self.train()

                states = next_states
                self.env_step_count += self.env.num_envs
                episode_step_count += self.env.num_envs

                if self.env_step_count % 1000 == 0:
                    wandb.log({
                        "Episode": ep,
                        "Episode Step Count": episode_step_count,
                        "Env Step Count": self.env_step_count,
                        "Update Count": self.train_count,
                    })

            # 單回合結束紀錄
            episode_time = time.time() - start_time
            avg_reward = episode_rewards.mean()

            print(f"[Eval] Ep: {ep} Avg Reward: {avg_reward:.2f} SC: {self.env_step_count} UC: {self.train_count} Eps: {self.epsilon:.4f} Time: {episode_time:.2f}s")

            wandb.log({
                "Episode": ep,
                "Avg Episode Reward": avg_reward,
                "Env Step Count": self.env_step_count,
                "Max Reward": episode_rewards.max(),
                "Min Reward": episode_rewards.min(),
                "Episode Time (sec)": episode_time,
                "Epsilon": self.epsilon,
                "Update Count": self.train_count,
            })

            if ep % 1 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

                eval_reward = sum(self.evaluate() for _ in range(args.eval_batch_size)) / args.eval_batch_size
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    best_model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), best_model_path)
                    print(f"Saved new best model to {best_model_path} with reward {eval_reward}")

                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_step_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_step_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })
        
    def run(self, episodes=10000):
        for ep in range(episodes):
            start_time = time.time()
            obs, _ = self.env.reset()
            if self.preprocessor:
                state = self.preprocessor.reset(obs)
            else:
                state = obs
            
            done = False
            episode_reward = 0
            episode_step_count = 0

            while not done and episode_step_count < self.max_episode_steps:
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
                episode_reward += reward
                self.env_step_count += 1
                episode_step_count += 1

                if self.env_step_count % 1000 == 0:                 
                    #print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Episode Step Count": episode_step_count,
                        "Env Step Count": self.env_step_count,
                        "Update Count": self.train_count,
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            end_time = time.time()
            episode_time = end_time - start_time
            print(f"[Eval] Ep: {ep} Episode Reward: {episode_reward} SC: {self.env_step_count} UC: {self.train_count} Eps: {self.epsilon:.4f} Time: {episode_time:.2f}s")
            
            wandb.log({
                "Episode": ep,
                "Episode Reward": episode_reward,
                "Env Step Count": self.env_step_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
                "Episode Time (sec)": episode_time
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 1 == 0:
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
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_step_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_step_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        if self.preprocessor_eval:
            state = self.preprocessor_eval.reset(obs)
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
            if self.preprocessor_eval:
                state = self.preprocessor_eval.step(next_obs)
            else:
                state = next_obs

        return total_reward


    def train(self):
        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
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
        # DDQN
        
        td_errors = target_q - q_values
        loss = (weights * td_errors.pow(2)).mean()
        #loss = (td_errors.pow(2)).mean()
        wandb.log({
                    "TD errors": td_errors.abs().mean().item(),
                    "Loss": loss,
                })
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        #self.memory.update_priorities(indices, td_errors.abs().detach().cpu().numpy())
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        wandb.log({
                "Q mean":q_values.mean().item(),
                "std": q_values.std().item(),
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--save-dir", type=str, default="./results/042501")
    # parser.add_argument("--num-envs", type=int, default=1)
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

    # parser.add_argument("--train-per-step", type=int, default=8)
    # parser.add_argument("--batch-size", type=int, default=64)
    # parser.add_argument("--memory-size", type=int, default=100000)
    # parser.add_argument("--lr", type=float, default=0.0001)
    # parser.add_argument("--discount-factor", type=float, default=0.95)
    # parser.add_argument("--epsilon-start", type=float, default=1.0)
    # parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    # parser.add_argument("--epsilon-min", type=float, default=0.1)
    # parser.add_argument("--target-update-frequency", type=int, default=2000)
    # parser.add_argument("--replay-start-size", type=int, default=50000)
    # parser.add_argument("--max-episode-steps", type=int, default=float('inf'))
    # parser.add_argument("--eval-batch-size", type=int, default=10)
    # parser.add_argument("--clip", type=int, default=1)
    # parser.add_argument("--scheduler", type=int, default=1)
    # parser.add_argument("--scheduler-step", type=int, default=100000)
    # parser.add_argument("--scheduler-gamma", type=float, default=0.9)
    # parser.add_argument("--frame-stack", type=int, default=4)


    parser.add_argument("--save-dir", type=str, default="./results/0425_1_03")
    parser.add_argument("--num-envs", type=int, default=8)

    parser.add_argument("--train-per-step", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.1)
    parser.add_argument("--target-update-frequency", type=int, default=2000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=float('inf'))
    parser.add_argument("--eval-batch-size", type=int, default=10)
    parser.add_argument("--clip", type=int, default=0)
    parser.add_argument("--scheduler", type=int, default=1)
    parser.add_argument("--scheduler-step", type=int, default=100000)
    parser.add_argument("--scheduler-gamma", type=float, default=0.98)
    parser.add_argument("--frame-stack", type=int, default=4)


    # parser.add_argument("--save-dir", type=str, default="./results/0425_1_02")
    # parser.add_argument("--num-envs", type=int, default=8)

    # parser.add_argument("--train-per-step", type=int, default=16)
    # parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--memory-size", type=int, default=200000)
    # parser.add_argument("--lr", type=float, default=0.0001)
    # parser.add_argument("--discount-factor", type=float, default=0.98)
    # parser.add_argument("--epsilon-start", type=float, default=1.0)
    # parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    # parser.add_argument("--epsilon-min", type=float, default=0.1)
    # parser.add_argument("--target-update-frequency", type=int, default=2000)
    # parser.add_argument("--replay-start-size", type=int, default=50000)
    # parser.add_argument("--max-episode-steps", type=int, default=float('inf'))
    # parser.add_argument("--eval-batch-size", type=int, default=10)
    # parser.add_argument("--clip", type=int, default=1)
    # parser.add_argument("--scheduler", type=int, default=0)
    # parser.add_argument("--scheduler-step", type=int, default=100000)
    # parser.add_argument("--scheduler-gamma", type=float, default=0.98)
    # parser.add_argument("--frame-stack", type=int, default=4)


    # parser.add_argument("--save-dir", type=str, default="./results/0425_1_01")
    # parser.add_argument("--num-envs", type=int, default=8)

    # parser.add_argument("--train-per-step", type=int, default=16)
    # parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--memory-size", type=int, default=200000)
    # parser.add_argument("--lr", type=float, default=0.0005)
    # parser.add_argument("--discount-factor", type=float, default=0.95)
    # parser.add_argument("--epsilon-start", type=float, default=1.0)
    # parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    # parser.add_argument("--epsilon-min", type=float, default=0.1)
    # parser.add_argument("--target-update-frequency", type=int, default=2000)
    # parser.add_argument("--replay-start-size", type=int, default=50000)
    # parser.add_argument("--max-episode-steps", type=int, default=float('inf'))
    # parser.add_argument("--eval-batch-size", type=int, default=10)
    # parser.add_argument("--clip", type=int, default=1)
    # parser.add_argument("--scheduler", type=int, default=1)
    # parser.add_argument("--scheduler-step", type=int, default=100000)
    # parser.add_argument("--scheduler-gamma", type=float, default=0.98)
    # parser.add_argument("--frame-stack", type=int, default=4)

    args = parser.parse_args()

    if args.task == 1:
        wandb.init(
            project="DLP-Lab5-DQN-CartPole",
            name=args.wandb_run_name,
            save_code=True
        )
    if args.task == 2:
        wandb.init(
            project="DLP-Lab5-DQN-Pong-v5-task2",
            name=args.wandb_run_name,
            save_code=True
        )

    wandb.config.update(vars(args))

    agent = DQNAgent(args=args)
    if(args.num_envs > 1):
        agent.vec_run(episodes=10000)
    else:
        agent.run(10000)
