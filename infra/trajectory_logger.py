# coding=utf-8
"""
Trajectory Logger for Curriculum Learning & RLHF/SFT Dataset Collection.
Records (Command, Vision Context, Model Thought, Action, User Reward) to JSONL.
"""
import json
import os
from datetime import datetime, timezone

class TrajectoryLogger:
    def __init__(self, dataset_path="dataset/curriculum_trajectories.jsonl"):
        self.dataset_path = dataset_path
        dir_name = os.path.dirname(dataset_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    def log_trajectory(self, user_command: str, thought: str, action: dict, reward: float, curriculum_level: str = "level_1_basic", image_b64: str = None):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "curriculum_level": curriculum_level,
            "user_command": user_command,
            "reward": reward,
            "image_b64": image_b64,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a visuomotor Minecraft 1.8.9 agent. Observe the 1st-person screen view and command, and output the JSON action."
                },
                {
                    "role": "user",
                    "content": f"Command: {user_command}"
                },
                {
                    "role": "assistant",
                    "content": f"<think>{thought}</think>{json.dumps(action)}"
                }
            ]
        }
        with open(self.dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

if __name__ == "__main__":
    logger = TrajectoryLogger("scratch_dataset.jsonl")
    logger.log_trajectory(
        user_command="vire para a esquerda",
        thought="Girar o mouse no eixo X negativo para virar a visão para a esquerda.",
        action={"mouse": [-40, 0], "duration_ms": 150},
        reward=1.0,
        curriculum_level="level_1_basic_movement"
    )
    print("[OK] Trajectory logged to scratch_dataset.jsonl")
    if os.path.exists("scratch_dataset.jsonl"):
        os.remove("scratch_dataset.jsonl")
