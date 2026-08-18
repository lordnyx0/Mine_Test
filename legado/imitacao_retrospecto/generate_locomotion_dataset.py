# coding=utf-8
"""
Generate Locomotion Trajectories Dataset for Qwen3LoopVLA.
Creates dataset/locomotion_trajectories.jsonl containing synthetic visual scenes
and targets for 4-direction movement (WASD) and 2D camera turning (Yaw/Pitch dx/dy).
"""
import base64
import io
import json
import os
import random
import numpy as np
from PIL import Image, ImageDraw

OUTPUT_PATH = "dataset/locomotion_trajectories.jsonl"

def draw_locomotion_scene(action_type: str) -> Image.Image:
    """Generates synthetic visual scenes corresponding to navigation states."""
    img = Image.new("RGB", (224, 224), color=(135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)

    # Draw basic ground (Green grass)
    draw.rectangle([0, 112, 224, 224], fill=(34, 139, 34))

    if action_type == "W":
        # Clear path straight ahead -> Open green horizon with a central brown path
        draw.polygon([(80, 224), (144, 224), (118, 112), (106, 112)], fill=(139, 69, 19))
    elif action_type == "S":
        # Obstacle/wall directly ahead -> Large stone grey wall
        draw.rectangle([20, 40, 204, 180], fill=(100, 100, 100))
    elif action_type == "A":
        # Path turning to the left -> Brown path curving left
        draw.polygon([(0, 224), (80, 224), (0, 120)], fill=(139, 69, 19))
        draw.rectangle([140, 60, 220, 180], fill=(120, 120, 120))  # Obstacle on the right
    elif action_type == "D":
        # Path turning to the right -> Brown path curving right
        draw.polygon([(144, 224), (224, 224), (224, 120)], fill=(139, 69, 19))
        draw.rectangle([40, 60, 120, 180], fill=(120, 120, 120))   # Obstacle on the left
    elif action_type == "CAM_LEFT":
        # Target object shifted right, need to rotate camera left
        draw.ellipse([160, 100, 200, 140], fill=(255, 215, 0))  # Yellow target on right
    elif action_type == "CAM_RIGHT":
        # Target object shifted left, need to rotate camera right
        draw.ellipse([20, 100, 60, 140], fill=(255, 215, 0))   # Yellow target on left
    elif action_type == "CAM_UP":
        # Looking up -> Mostly sky
        draw.rectangle([0, 0, 224, 180], fill=(135, 206, 235))
        draw.ellipse([90, 30, 134, 74], fill=(255, 255, 255))   # Sun/Cloud
    elif action_type == "CAM_DOWN":
        # Looking down -> Mostly ground/dirt
        draw.rectangle([0, 0, 224, 224], fill=(101, 67, 33))

    # Add slight random noise to prevent identical pixel memorization
    arr = np.array(img, dtype=np.int16)
    noise = np.random.randint(-5, 6, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def generate_locomotion_dataset(num_samples_per_class=40):
    os.makedirs("dataset", exist_ok=True)

    classes = [
        ("W", {"hold": ["W"]}, [0, 0]),
        ("S", {"hold": ["S"]}, [0, 0]),
        ("A", {"hold": ["A"]}, [-30, 0]),
        ("D", {"hold": ["D"]}, [30, 0]),
        ("CAM_LEFT", {"mouse": [-40, 0]}, [-40, 0]),
        ("CAM_RIGHT", {"mouse": [40, 0]}, [40, 0]),
        ("CAM_UP", {"mouse": [0, -20]}, [0, -20]),
        ("CAM_DOWN", {"mouse": [0, 20]}, [0, 20]),
    ]

    total_written = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for action_type, action_dict, mouse_vec in classes:
            for i in range(num_samples_per_class):
                img = draw_locomotion_scene(action_type)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=90)
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                act_copy = dict(action_dict)
                act_copy["mouse"] = mouse_vec

                entry = {
                    "task": "autonomous_locomotion",
                    "action_type": action_type,
                    "reward": 1.0,
                    "image_b64": img_b64,
                    "messages": [
                        {"role": "user", "content": f"Navigate direction {action_type}"},
                        {"role": "assistant", "content": json.dumps({"action": act_copy, "completed": False})}
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"[Dataset Generator] Geradas {total_written} amostras de locomoção e rotação em '{OUTPUT_PATH}'.")

if __name__ == "__main__":
    generate_locomotion_dataset()
