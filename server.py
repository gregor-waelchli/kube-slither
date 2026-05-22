# server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import torch
import random
import numpy as np
from collections import deque

from torch.xpu import device

from game import SnakeGameAI
from model import QNet, QTrainer

app = FastAPI()

# Game & Model
game = SnakeGameAI(w=640, h=480)
print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("Yes I'm running on GPU...")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model = QNet().cuda()
else:
    print("Oh no, running on CPU only... *snif*")
    device = torch.device("cpu")
    model = QNet()

trainer = QTrainer(model)

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 80
        self.scores = []
        self.avg_scores = []
        self.score_window = deque(maxlen=50)

    def get_action(self, state):
        self.epsilon = max(10, 80 - self.n_games // 4)  # slower decay
        if random.randint(0, 200) < self.epsilon:
            return random.randint(0, 2)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            pred = model(state_tensor)
        return torch.argmax(pred).item()

agent = Agent()

@app.get("/")
async def get():
    with open("index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            state_old = game.get_state()
            action = agent.get_action(state_old)

            reward, game_over, score = game.step(action)
            state_new = game.get_state()

            # Train every Xth step
            if agent.n_games % 2 == 0:
                try:
                    trainer.train_step(state_old, action, reward, state_new, game_over)
                except Exception as e:
                    print(f"Training error: {e}")

            # Show every XXth game
            delay = 0
            if agent.n_games % 50 == 0:
                delay = 0.02
                await websocket.send_json({
                    "type": "game",
                    "snake": [[p.x, p.y] for p in game.snake],
                    "food": [game.food.x, game.food.y],
                    "score": score,
                    "episode": agent.n_games
                })

            if game_over:
                agent.n_games += 1
                agent.score_window.append(score)
                agent.scores.append(score)
                avg_score = np.mean(agent.score_window)
                agent.avg_scores.append(avg_score)

                print(f"Episode {agent.n_games} | Score: {score} | Avg: {avg_score:.1f} | Epsilon: {agent.epsilon}")

                # Update chart data
                await websocket.send_json({
                    "type": "chart",
                    "episode": agent.n_games,
                    "score": score,
                    "avg_score": float(avg_score)
                })

                game.reset()

            await asyncio.sleep(delay)

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)