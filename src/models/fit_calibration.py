import json
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "data"))
sys.path.append(str(PROJECT_ROOT / "src" / "utils"))

from dataset import get_dataloaders
from model_factory import build_model
from config import load_config
from calibration import fit_temperature

def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    config = load_config(PROJECT_ROOT / "configs" / "vit_b_16_dropout.yaml")
    model = build_model(config["backbone"]).to(device)
    model.load_state_dict(torch.load(PROJECT_ROOT / config["checkpoint_dir"] / "best_model.pt", map_location=device))

    _, val_loader, _ = get_dataloaders(batch_size=config["data"]["batch_size"])

    temperature = fit_temperature(model, val_loader, device)
    print(f"Fitted temperature: {temperature:.4f}")

    out_path = PROJECT_ROOT / config["checkpoint_dir"] / "temperature.json"
    with open(out_path, "w") as f:
        json.dump({"temperature": temperature}, f)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()