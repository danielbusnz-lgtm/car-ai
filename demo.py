import pygame
import torch
import numpy as np
from environment import CarEnvironment
from train import DQN

def demo(model_path="model_final.pth"):
    env = CarEnvironment()


    model = DQN(state_size = 7, action_size = 3)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    pygame.init()

    print(f"Loading model: {model_path}")
    print("Watching trained AI")
    print("Close the window to exit")
    print("-" * 60)

    episode =1
    running =True

    while running:
        state = env.reset()
        total_reward =0
        step = 0

        episode_running = True
        while episode_running:
            for event in pygame.event.get():
                if event.type ==pygame.QUIT:
                    running = False
                    episode_running = False

            if not running:
                break

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

            next_state, reward, done, info = env.step(action)

            env.render()

            total_reward += reward
            state = next_state
            step += 1

            if done: 
                episode_running = False
        if running:
            print(f"Episode {episode} | Steps: {step} | Total Reward {total_reward:.1f}")
            episode += 1

    pygame.quit()
    print("\n Demo Complete")

if __name__ == "__main__":
    demo("model_final.pth")
