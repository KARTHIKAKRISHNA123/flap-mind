import random

import flappy_bird_gymnasium 
import gymnasium as gym

from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random
import time







if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)



class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]
        self.mini_batch_size = params["mini_batch_size"]
        self.max_episodes = params.get("max_episodes", None)  # None = run forever

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")
        self.CHECKPOINT_FILE = os.path.join(RUNS_DIR, f"{self.param_set}_checkpoint.pt")
        self.checkpoint_interval = params.get("checkpoint_interval", 500)  # episodes between checkpoints

    def run(self, is_training=True, render=False):
        render_mode = "human" if render else None
        env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)

        num_states = env.observation_space.shape[0] # input dimension of the state space
        num_actions = env.action_space.n # output dimension of the action space

        policy_dqn = DQN(num_states, num_actions).to(device) # Initialize the DQN model and move it to the appropriate device (CPU, GPU, or MPS)

        state, _ = env.reset()

        start_episode = 0

        if is_training:
            memory = ReplayMemory(self.replay_memory_size) # Initialize the replay memory for experience replay
            epsilon = self.epsilon_init # Initialize epsilon for epsilon-greedy action selection
            target_dqn = DQN(num_states, num_actions).to(device)
            # Copy the weights & biases values from the policy DQN to the target DQN
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha) 

            best_reward = float("-inf")

            # Resume from checkpoint if one exists
            if os.path.exists(self.CHECKPOINT_FILE):
                checkpoint = torch.load(self.CHECKPOINT_FILE, map_location=device)
                policy_dqn.load_state_dict(checkpoint["policy_state"])
                target_dqn.load_state_dict(checkpoint["target_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                epsilon = checkpoint["epsilon"]
                best_reward = checkpoint["best_reward"]
                start_episode = checkpoint["episode"] + 1
                print(f"Resumed from checkpoint: episode {start_episode}, epsilon={epsilon:.4f}, best_reward={best_reward}")

        else:
            # Best Model & Best Policy Load
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        start_time = time.time()

        try:
            for episode in itertools.count(start=start_episode):

                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float, device=device )
                episode_reward = 0
                terminated = False

                while (not terminated and episode_reward < self.reward_threshold):
                    if is_training and random.random() < epsilon:
                    # Next Action
                    # (feed the observation to your agent here)
                        action = env.action_space.sample() # explore
                        action = torch.tensor(action, dtype=torch.long, device=device)
                    else:
                        with torch.no_grad():
                            action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() # exploit

                    #Processing => terminated is True if the player has died, truncated is True if the game has been closed, Terminated => done
                    next_state, reward, terminated, _, info = env.step(action.item())

                    episode_reward += reward


                    #Create Tensors
                    reward = torch.tensor(reward, dtype=torch.float, device=device)
                    next_state = torch.tensor(next_state, dtype=torch.float, device=device)

                    if is_training:
                        memory.append((state, action, next_state, reward, terminated))
                        steps += 1


                    # #Checking if the player is still alive
                    # if terminated:
                    #     break

                    state = next_state

                elapsed = time.time() - start_time
                episodes_done_this_run = episode - start_episode + 1
                eps_per_sec = episodes_done_this_run / elapsed if elapsed > 0 else 0.0
                print(f"Episode: {episode + 1} with Total Reward: {episode_reward} & Epsilon: {epsilon:.4f} | {eps_per_sec:.2f} ep/s | elapsed {elapsed/60:.1f} min")

                if is_training and (episode + 1) % 100 == 0:
                    remaining = None
                    if self.max_episodes is not None and eps_per_sec > 0:
                        remaining_eps = self.max_episodes - (episode + 1)
                        remaining = remaining_eps / eps_per_sec / 3600  # hours
                    log_line = f"[timing] episode={episode+1} eps_per_sec={eps_per_sec:.2f} elapsed_min={elapsed/60:.1f}"
                    if remaining is not None:
                        log_line += f" est_hours_remaining={remaining:.1f}"
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_line + "\n")

                if is_training and self.max_episodes is not None and (episode + 1) >= self.max_episodes:
                    print(f"Reached max_episodes={self.max_episodes}, stopping training.")
                    break

                if is_training:
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                    if episode_reward > best_reward:
                        log_message = f"Best Reward: {episode_reward} at Episode: {episode + 1} with Epsilon: {epsilon:.4f}"

                        with open(self.LOG_FILE, "a") as f:
                            f.write(log_message +"\n") 

                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        best_reward = episode_reward

                if is_training and len(memory) > self.mini_batch_size:
                    # get sample
                    mini_batch = memory.sample(self.mini_batch_size)

                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    #Sync the network
                    if steps > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

                if is_training and (episode + 1) % self.checkpoint_interval == 0:
                    torch.save({
                        "policy_state": policy_dqn.state_dict(),
                        "target_state": target_dqn.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "epsilon": epsilon,
                        "best_reward": best_reward,
                        "episode": episode,
                    }, self.CHECKPOINT_FILE)
                    print(f"Checkpoint saved at episode {episode + 1}")

        except KeyboardInterrupt:
            if is_training:
                torch.save({
                    "policy_state": policy_dqn.state_dict(),
                    "target_state": target_dqn.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "epsilon": epsilon,
                    "best_reward": best_reward,
                    "episode": episode,
                }, self.CHECKPOINT_FILE)
                print(f"\nInterrupted. Checkpoint saved at episode {episode + 1}. Re-run the same command to resume.")
            raise

        # env.close() - manually stop

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get experience
        # for state, action, next_state, reward, terminated in mini_batch:
        #     if terminated:
        #         target = reward
        #     else:
        #         with torch.no_grad():
        #             target_q = reward + self.gamma * target_dqn(next_state).max()

        #     current_q = policy_dqn(state)


        #     # loss
        #     loss = self.loss_fn(current_q, target_q)

        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()

        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)


        # Calculate target Q-values - if terminated = true => zero
        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]


        # Calculate y-pred Q_value from current policy DQN
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        #Compute Loss
        loss = self.loss_fn(current_q, target_q)

        #optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
        #Parse Command Line Inputs
        parser = argparse.ArgumentParser(description="Train or Test Model")
        parser.add_argument("hyperparameters", help='')
        parser.add_argument("--train", help="Training Mode", action="store_true")
        args = parser.parse_args()

        dql = Agent(param_set=args.hyperparameters)

        if args.train:
            dql.run(is_training=True)
        else:
            dql.run(is_training=False, render=True)


