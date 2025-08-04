# 🎮 2048 Game AI using Deep Q-Learning

This project implements a Reinforcement Learning agent that learns to play the popular 2048 puzzle game using **Deep Q-Networks (DQN)**.  
The goal is to teach the agent to **maximize the highest tile**, not just the score — aiming for smart, long-term planning.

---

## 🧠 Project Highlights

- 🔁 Uses a custom-built game environment for 2048
- 🧠 Trained with Deep Q-Learning (DQN)
- 💾 Experience Replay and Target Networks
- 📈 Tracks performance across episodes
- 🧪 Evaluation focused on the highest tile achieved

---

## 📊 Results

After training, the agent is capable of reaching:

* 🟪 **1024** and even **2048** tiles consistently
* ⚖️ Chooses moves that optimize for long-term reward
* 📈 Learns strategies like corner-stacking and tile prioritization

> *Visualization plots and logs can be added later for clarity.*

---

## 🔮 Future Work

* Add TensorBoard logging
* Try Double DQN or Dueling DQN architectures
* Deploy using Streamlit or PyGame UI
* Try curriculum learning (start with small boards)
