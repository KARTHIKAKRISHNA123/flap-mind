import gymnasium as gym
import flappy_bird_gymnasium
import pygame

#Creating our Env
env = gym.make("FlappyBird-v0", render_mode="human")
state, info = env.reset()
done = False

#Initialize pyGame keyboard
pygame.init()
screen = pygame.display.get_surface() # Gym has already created a window for us, we just need to get the surface

while not done:
    action = 0 # Default action is to do nothing & 1 means to flap

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1 # Flap if space is pressed
    state, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
pygame.quit()