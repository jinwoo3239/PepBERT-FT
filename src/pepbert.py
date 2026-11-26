
import torch
import torch.nn as nn


class PepBERTRegressor(nn.Module):
    """PepBERT regression with SOS or mean pooling."""

    def __init__(
        self,
        encoder,
        hidden_size=320,
        pooling="sos",
        fc_layers=1,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = encoder
        self.pooling = pooling

        # Regression head
        dimensions = [hidden_size, 256, 128, 64]
        layers = []

        for i in range(fc_layers - 1):
            layers.extend([
                nn.Linear(dimensions[i], dimensions[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        layers.append(nn.Linear(dimensions[fc_layers - 1], 1))

        self.regressor = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):

        # PepBERT encoder
        encoder_mask = attention_mask[:, None, None, :]
        hidden = self.encoder.encode(input_ids, encoder_mask)

        # Pooling
        if self.pooling == "sos":
            features = hidden[:, 0, :]

        elif self.pooling == "mean":
            amino_mask = (
                attention_mask.bool()
                & (input_ids != 2)
                & (input_ids != 3)
            )

            mask = amino_mask.unsqueeze(-1)

            features = (
                (hidden * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1)
            )

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.regressor(features).squeeze(-1)


from torch.utils.data import Dataset


class PeptideDataset(Dataset):
    """Prepare peptide sequences for PepBERT."""

    def __init__(self, df, tokenizer, max_length=52):
        self.sequences = df["sequence"].tolist()
        self.targets = df["pMIC"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.sos_id = tokenizer.token_to_id("[SOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # Encode peptide
        tokens = (
            [self.sos_id]
            + self.tokenizer.encode(sequence).ids
            + [self.eos_id]
        )

        tokens = tokens[:self.max_length]
        length = len(tokens)

        # Padding
        tokens += [self.pad_id] * (self.max_length - length)

        attention_mask = (
            [1] * length
            + [0] * (self.max_length - length)
        )

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long
            ),
            "labels": torch.tensor(
                self.targets[idx], dtype=torch.float32
            )
        }


import pandas as pd
from tqdm.auto import tqdm
from src.seed import set_seed

def train_pepbert(
    model,
    train_loader,
    valid_loader,
    seed,
    save_path,
    history_path,
    max_epochs=100,
    patience=7,
    lr=1e-5,
    weight_decay=0.01,
    run_name="PepBERT"
):
    import pandas as pd
    import torch
    from tqdm import tqdm

    save_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0
    patience_count = 0
    history = []

    # Track training epochs
    with tqdm(
        range(1, max_epochs + 1),
        desc=run_name,
        unit="epoch",
        position=0,
        leave=True,
        ncols=150,
        mininterval=1
    ) as pbar:

        for epoch in pbar:

            # Training
            model.train()
            train_loss = 0.0
            train_samples = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()

                predictions = model(
                    input_ids,
                    attention_mask
                )

                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                train_loss += loss.item() * batch_size
                train_samples += batch_size

            train_mse = train_loss / train_samples

            # Validation
            model.eval()
            valid_loss = 0.0
            valid_samples = 0

            with torch.no_grad():
                for batch in valid_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    predictions = model(
                        input_ids,
                        attention_mask
                    )

                    loss = criterion(predictions, labels)

                    batch_size = labels.size(0)
                    valid_loss += loss.item() * batch_size
                    valid_samples += batch_size

            valid_mse = valid_loss / valid_samples

            # Save best checkpoint
            if valid_mse < best_val:
                best_val = valid_mse
                best_epoch = epoch
                patience_count = 0

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_mse": best_val,
                    "seed": seed
                }, save_path)

            else:
                patience_count += 1

            # Save training history
            history.append({
                "Epoch": epoch,
                "Train_MSE": train_mse,
                "Validation_MSE": valid_mse,
                "Best_Validation_MSE": best_val,
                "Best_Epoch": best_epoch,
                "Patience": patience_count
            })

            pd.DataFrame(history).to_csv(
                history_path,
                index=False
            )

            # Display training progress
            pbar.set_postfix({
                "Train": f"{train_mse:.4f}",
                "Val": f"{valid_mse:.4f}",
                "Best": f"{best_val:.4f}",
                "BestEp": best_epoch,
                "Patience": f"{patience_count}/{patience}"
            })

            # Early stopping
            if patience_count >= patience:
                break

    return {
        "Best_Epoch": best_epoch,
        "Best_Validation_MSE": best_val
    }
