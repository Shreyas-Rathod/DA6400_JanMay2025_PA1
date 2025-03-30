# DA6400 - Programming Assignment 1 

This repository contains the implementation and evaluation of two fundamental reinforcement learning algorithms: SARSA and Q-Learning. The agents were trained and tested within the Gymnasium environment suite, focusing on the following tasks:

## Team Member :
Shreyas Rathod

## Environments

The following Gymnasium environments were utilized for this assignment:

* **CartPole-v1:** The classic balancing problem where the agent must learn to keep a pole upright on a moving cart by applying left or right forces.
* **MountainCar-v0:** A challenging environment where a car must learn to escape a valley and reach the top of a hill by strategically applying accelerations. This version uses discrete actions.
* **MiniGrid-Dynamic-Obstacles-5x5-v0:** A bonus environment involving navigation in a small, empty room with moving obstacles. The agent's goal is to reach a green goal square while avoiding collisions.

## Algorithms

The following reinforcement learning algorithms were implemented and compared:

* **SARSA:** An on-policy temporal difference learning algorithm utilizing $\epsilon$-greedy exploration for action selection.
* **Q-Learning:** An off-policy temporal difference learning algorithm employing Softmax exploration for action selection.

Reward shaping was considered but generally discouraged for simpler environments to assess the algorithms' performance in their basic forms.

**To run the assignment, you need to:**

Copy the code to a Python file (e.g., cs24m046_pa1.py)
Install the requirements from the requirements.txt file:
```python
pip install -r requirements.txt
```
Run the code:
```python
python rl_assignment.py
```

**The code will:**
1. Perform hyperparameter search on both environments
2. Train models with the best hyperparameters found
3. Generate plots comparing algorithm performance
4. Save results to a "results" directory

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/lDGSs7Pt)
