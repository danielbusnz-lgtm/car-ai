import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
from collections import deque
from environment import CarEnvironment



class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()


        #First Fully connected Layer, takes the 7 inputs and outputs 64
        self.fc1 = nn.Linear(state_size, 64)

        #Second Fully Connceted Layer, takes 64 inputs and outputs 64, hidden layer processing
        self.fc2 = nn.Linear(64, 64)
        
        #Third Fully Conncected Layer, takes the 64 inputs and gives us the 3 action states, values for the actions
        self.fc3 = nn.Linear(64, action_size)

    # x is the cars state
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(self, state_size, action_size, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device if device else torch.device("cpu")
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        self.memory = deque(maxlen = 10000)
        
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()


    #Adding to the agents memory of past actions, rewards and failures
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state,action,reward,next_state,done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()




    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_tensor =  torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

            target = reward

            if not done:
                with torch.no_grad():
                    target= reward + self.gamma * torch.max(self.model(next_state_tensor)).item()

            q_values = self.model(state_tensor)
            target_q = q_values.clone()
            target_q[0][action] = target

            loss = self.criterion(q_values, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CarEnvironment()
    agent = DQNAgent(state_size=7, action_size=3, device = device)
    agent.model.to(device)


    episodes = 1000
    max_steps = 500
    
    print(f" Device using: {device}")

    print("training began")
    print(f"Episode {episodes}, Max steps per episode: {max_steps}")
    print("-" * 60)

    for episode in range(episodes):
        state= env.reset()
        total_reward = 0

        for step in range(max_steps):

            action = agent.act(state)

            next_state, reward, done, info = env.step(action)
        
            env.render()

            agent.remember(state, action, reward, next_state, done)

            agent.replay()

            total_reward += reward
            state = next_state
            


            if done:
                break
        print(f"Episode {episode + 1}/{episodes} | Steps: {step + 1} | Total Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f}")

        if (episode +1 ) % 100 == 0:
            agent.save(f"model_episode_{episode + 1}.pth")
            print(f"model saved at episode {episode + 1}")
    agent.save("model_final.pth")
    print("\n Training completed, final model saved as 'model_final.pth'")

if __name__=="__main__":
    train()











