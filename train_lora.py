import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import TimesFm2_5ModelForPrediction

from data_handle.price_lora_dataset import PriceWindowDataset, load_price_splits


REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "datasets" / "datasets.csv"
BASE_MODEL_PATH = REPO_ROOT / "timesfm" / "transformers"
ADAPTER_DIR = REPO_ROOT / "adapters" / "timesfm_sichuan_lora_r4"
LOG_PATH = REPO_ROOT / "outputs" / "timesfm_lora_train_log.jsonl"

CONTEXT_LEN = 672
HORIZON_LEN = 96
EXPECTED_MINUTES = 15
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 0

LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05


def collate_price_windows(batch):
    past_values = torch.stack([item["past_values"] for item in batch])
    future_values = torch.stack([item["future_values"] for item in batch])
    return {"past_values": past_values, "future_values": future_values}


def build_dataloaders():
    train_split, val_split, _ = load_price_splits(
        DATA_PATH,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )
    train_dataset = PriceWindowDataset(
        train_split,
        context_len=CONTEXT_LEN,
        horizon_len=HORIZON_LEN,
        expected_minutes=EXPECTED_MINUTES,
    )
    val_dataset = PriceWindowDataset(
        val_split,
        context_len=CONTEXT_LEN,
        horizon_len=HORIZON_LEN,
        expected_minutes=EXPECTED_MINUTES,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_price_windows,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_price_windows,
    )
    return train_loader, val_loader


def build_model(device):
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        str(BASE_MODEL_PATH),
        dtype=model_dtype,
    )
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model.to(device)


def forward_loss(model, batch, device):
    past_values = batch["past_values"].to(device)
    future_values = batch["future_values"].to(device)
    past_sequence = [row for row in past_values]

    outputs = model(
        past_values=past_sequence,
        future_values=future_values,
        forecast_context_len=CONTEXT_LEN,
    )
    return outputs.loss


@torch.no_grad()
def evaluate_loss(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in val_loader:
        loss = forward_loss(model, batch, device)
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def append_log(record):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_config():
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "data_path": str(DATA_PATH),
        "base_model_path": str(BASE_MODEL_PATH),
        "adapter_dir": str(ADAPTER_DIR),
        "context_len": CONTEXT_LEN,
        "horizon_len": HORIZON_LEN,
        "expected_minutes": EXPECTED_MINUTES,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
    }
    (ADAPTER_DIR / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders()
    model = build_model(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        running_batches = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            loss = forward_loss(model, batch, device)
            scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaled_loss.backward()

            running_loss += float(loss.detach().cpu())
            running_batches += 1

            if batch_index % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        if running_batches % GRADIENT_ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        train_loss = running_loss / max(running_batches, 1)
        val_loss = evaluate_loss(model, val_loader, device)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        append_log(record)
        print(record)

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    save_config()
    print(f"Saved LoRA adapter to: {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
