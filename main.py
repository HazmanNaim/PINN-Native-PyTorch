from pathlib import Path
import random

import numpy as np
import torch

from model import Pinn
from data import get_dataset
from trainer import Trainer

def process_test_result(
    test_data: torch.Tensor,
    loss: float,
    preds: torch.Tensor,
    lambda1: float,
    lambda2: float,
):
    preds = preds.detach().cpu().numpy()
    test_arr = np.array(test_data.data)
    p = test_arr[:, 3]
    u = test_arr[:, 4]
    v = test_arr[:, 5]
    w = test_arr[:, 6]  # Added for 3D
    p_pred = preds[:, 0]
    u_pred = preds[:, 1]
    v_pred = preds[:, 2]
    w_pred = preds[:, 3]  # Added for 3D

    # Error calculations
    err_u = np.mean(np.abs((u - u_pred) / u))
    err_v = np.mean(np.abs((v - v_pred) / v))
    err_w = np.mean(np.abs((w - w_pred) / w))  # Added for 3D
    err_p = np.mean(np.abs((p - p_pred) / p))

    err_lambda1 = np.abs(lambda1 - 1.0)
    err_lambda2 = np.abs(lambda2 - 0.01) / 0.01

    print(f"Error in u: {err_u:.4e}")
    print(f"Error in v: {err_v:.4e}")
    print(f"Error in w: {err_w:.4e}")  # Added for 3D
    print(f"Error in pressure: {err_p:.4e}")
    print(f"Error in lambda 1: {err_lambda1:.2f}")
    print(f"Error in lambda 2: {err_lambda2:.2f}")

    # Plot (if needed)
    pass

def main():
    torch.random.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data
    data_path = Path("data/pinn_data_3d.jsonl")  # Adjusted for 3D
    train_data, test_data, min_x, max_x = get_dataset(data_path)

    # Model
    model = Pinn(min_x, max_x, dim=3)  # Adjusted for 3D
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Train
    output_dir = Path(
        "result/pinn-3d-tanh-bs128-lr0.005-lrstep1-lrgamma0.8-epoch20")  # Adjusted for 3D
    trainer = Trainer(model, output_dir)
    trainer.train(train_data)

    # Test
    ckpt_dir = trainer.get_last_ckpt_dir()
    trainer.load_ckpt(ckpt_dir)
    outputs = trainer.predict(test_data)

    # Test Result
    lambda1 = trainer.model.lambda1.item()
    lambda2 = trainer.model.lambda2.item()
    print("lambda 1:", lambda1)
    print("lambda 2:", lambda2)
    loss = outputs["loss"]
    preds = outputs["preds"]
    process_test_result(test_data, loss, preds, lambda1, lambda2)

if __name__ == "__main__":
    main()
