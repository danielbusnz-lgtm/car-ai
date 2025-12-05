1. ##AI Reinforcement Leanring Car Project.
I used Deep Q-Learning (DQN) to teach a car how to drive around a track.

2. ##Features Include:
A custom car environment with vision based sensors using rays:
GPU-accelerated training

3.## Usage
Train: `python train.py`
Demo: `python demo.py`

4. ##Installation
pip install -r requirements.txt

5. ## How it works##
this works by using 5 raycasting sensors pointed from the front of the car at the  angles [-90,-45,0,45,90]. We then calculate the distance from each rays to the edge of the track. 

6. ##Future Improvements
Adding a more complicated track for the AI agent, using more angles and inputs for the ai as well as more outputs and actions the car can take. I would also like to include a 3D version of this project at some point, this is more of a working demo just to see a very simple reinforcement learning process in action.
