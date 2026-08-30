import flappy_bird_gymnasium 
import gymnasium as gym

from dqn import DQN
from experience_replay import ReplayMemory



if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



def run(self, is_training=True, render=False):
    env = gym.make("FlappyBird-v0", render_mode="human", use_Lidar=True if render else None)

    num_states = env.observation_space.shape[0] # input dimension of the state space
    num_actions = env.action_space.n # output dimension of the action space

    policy_dqn = DQN(num_states, num_actions).to(device) # Initialize the DQN model and move it to the appropriate device (CPU, GPU, or MPS)

    state, _ = env.reset()

    if is_training:
        memory = ReplayMemory(10000)

    while True:
        # Next Action
        # (feed the observation to your agent here)
        action = env.action_space.sample()

        #Processing => terminated is True if the player has died, truncated is True if the game has been closed, Terminated => done
        next_state, reward, terminated, _, info = env.step(action)

        if is_training:
            memory.append((state, action, new_state, reward, terminated))

        #Checking if the player is still alive
        if terminated:
            break

    env.close()