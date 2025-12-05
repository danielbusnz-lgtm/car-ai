import pygame
import random
from environment import CarEnvironment


env = CarEnvironment()

state = env.reset()

# Initialize pygame
pygame.init()

running =True
episode =1

print("Testing the car environment")
print("Close this window to exit")
print(" -" *40)

while running:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = random.choice([0,1,2])

    state, reward, done, info = env.step(action)

    env.render()

    print(f"Episode: {episode} | Step: {info['steps']} |  Reward:{reward:.1f} | Total: {info['total_reward']:.1f}")

    if done:
        print(f"Episode {episode} Ended! Total Steps: {info['steps']}, Total Reward: {info['total_reward']:.1f}")
        print("-" * 40)
        state = env.reset()
        episode += 1
pygame.quit()
print("Test Complete")

