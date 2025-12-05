# AI Reinforcement Learning Car Project

Put simply, I used Deep Q-Learning (DQN) to teach a car how to drive around a track.

## Features Include

A custom car environment with vision based sensors using rays:
- GPU-accelerated training
- idk its pretty simple tbh

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Train:**
```bash
python train.py
```

**Demo:**
```bash
python demo.py
```

## How It Works

This works by using 5 raycasting sensors pointed from the front of the car at the angles [-90,-45,0,45,90]. We then calculate the distance from each rays to the edge of the track.

## Future Improvements

Adding a more complicated track for the AI agent, using more angles and inputs for the ai as well as more outputs and actions the car can take. I would also like to include a 3D version of this project at some point, this is more of a working demo just to see a very simple reinforcement learning process in action.
