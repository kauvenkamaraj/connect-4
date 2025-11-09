# 🎮 Connect 4 — Fast Reinforcement Learning (Python + Pygame)

An intelligent **Connect 4** built with **Python** and **Pygame**, powered by **Linear Q-Learning** and **self-play reinforcement learning**.  
The AI learns from scratch — improving by playing against itself — and can be challenged by a human player in a fun interactive UI.  

---

## ✨ Features

- 🧠 **Linear Q-Learning Agent** – learns directly from handcrafted features
- 🤖 **Self-Play Training** – both sides share weights for faster convergence
- 🪄 **Reward Shaping** – encourages winning, blocking, and smart 3-in-a-row plays
- ⚡ **Fast Learning** – linear model learns in minutes (no deep networks)
- 🎨 **Polished Pygame UI** – animated board, color-coded discs, and modes
- 💾 **Save/Load Weights** – store your agent’s progress
- 🧩 **Three Modes**  
  - **Train** – AI vs AI (learns)
  - **Watch** – AI vs AI (no learning)
  - **Play** – Human vs AI

---

## ⚙️ Installation

### 1️⃣ Clone the repository
git clone https://github.com/kauvenkamaraj/connect4-rl.git
cd connect4-rl

### 2️⃣ Install dependencies
pip install -r requirements.txt

---

## ▶️ Running the Game

python connect4_rl.py

---

## 🎮 Controls

| Key | Action |
|-----|---------|
| **ENTER** | Start / Close instructions |
| **I** | Show instructions |
| **T** | Train (AI vs AI) |
| **W** | Watch (AI vs AI, no training) |
| **P** | Play (Human vs AI) |
| **1/2/3** | Adjust speed |
| **F** | Toggle fast mode |
| **S** | Save model |
| **L** | Load model |
| **R** | Reset game |
| **ESC** | Quit |

---

## 🧠 How It Learns

The agent uses **Linear Q-learning** to estimate action values:

Q_θ(s, a) = θᵀ φ(s, a)

Where:
- φ(s, a): feature vector describing the move
- θ: learned weights

The update rule (TD learning):

θ ← θ + α [r + γ maxₐ′ Q_θ(s′, a′) − Q_θ(s, a)] φ(s, a)

### Features include:
- Center column control  
- Count of 2-in-a-row and 3-in-a-row opportunities  
- Blocking and winning potential  
- Immediate-win and create-3 signals  

---

## 🧩 Reward System

| Event | Reward |
|--------|--------|
| Win | +1.0 |
| Loss | −1.0 |
| Draw | 0.0 |
| Immediate Win | +0.75 |
| Block Opponent | +0.50 |
| Create 3-in-a-row | +0.20 |

---

## 📊 Results

After several thousand self-play episodes, the agent:
- Learns to **prioritize center columns**
- **Block opponent traps**
- Create **multi-turn winning opportunities**
- Achieves **consistent win rates** over random play

---

## 🌟 Star the repo if you like it!

If you found this project useful or interesting, please consider giving it a ⭐ on GitHub — it really helps!
