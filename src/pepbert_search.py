
import gc
import torch
import pandas as pd

from torch.utils.data import DataLoader, Subset

from configs.common import MODEL_DIR, RESULT_DIR, RANDOM_SEEDS
from configs import pepbert as config

from src.seed import set_seed
from src.ml import load_validation_split
from src.pepbert import PepBERTRegressor, train_pepbert
from src.pepbert_pretrained import build_pretrained_encoder


def run_architecture_search(
    targets,
    train_datasets,
    model_module,
    pretrained_config,
    tokenizer,
    pretrained_weights,
    device
):
    tasks = [
        (target, pooling, fc_layers, seed)
        for target in targets
        for pooling in config.POOLING_TYPES
        for fc_layers in config.FC_LAYERS
        for seed in RANDOM_SEEDS
    ]

    records = []
    total_runs = len(tasks)

    for run_idx, (target, pooling, fc_layers, seed) in enumerate(tasks, 1):

        set_seed(seed)

        # Define current experiment
        run_name = (
            f"{run_idx}/{total_runs} | {target} | "
            f"{pooling.upper()} | FC{fc_layers} | Seed {seed}"
        )

        print(f"\\n[Run {run_idx}/{total_runs}] "
              f"{target} | {pooling.upper()} | FC{fc_layers} | Seed {seed}",
              flush=True)

        # Load existing validation split
        train_idx, valid_idx = load_validation_split(target, seed)

        train_loader = DataLoader(
            Subset(train_datasets[target], train_idx),
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )

        valid_loader = DataLoader(
            Subset(train_datasets[target], valid_idx),
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )

        # Initialize fresh pretrained encoder
        encoder = build_pretrained_encoder(
            model_module=model_module,
            pretrained_config=pretrained_config,
            tokenizer=tokenizer,
            pretrained_weights=pretrained_weights
        )

        # Initialize regression model
        model = PepBERTRegressor(
            encoder=encoder,
            hidden_size=pretrained_config["d_model"],
            pooling=pooling,
            fc_layers=fc_layers,
            dropout=config.DROPOUT
        ).to(device)

        # Define output paths
        model_dir = MODEL_DIR / "pepbert" / target
        result_dir = RESULT_DIR / "pepbert" / target

        save_path = (
            model_dir / pooling / "architecture_search"
            / f"FC{fc_layers}_seed{seed}.pt"
        )

        history_path = (
            result_dir / pooling / "architecture_search"
            / f"FC{fc_layers}_seed{seed}_history.csv"
        )

        # Train and validate
        result = train_pepbert(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            seed=seed,
            save_path=save_path,
            history_path=history_path,
            max_epochs=config.MAX_EPOCHS,
            patience=config.PATIENCE,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            run_name=run_name
        )

        # Record validation results
        records.append({
            "Dataset": target,
            "Pooling": pooling,
            "FC_Layers": fc_layers,
            "Seed": seed,
            "Dropout": config.DROPOUT,
            "Learning_Rate": config.LEARNING_RATE,
            "Weight_Decay": config.WEIGHT_DECAY,
            "Best_Epoch": result["Best_Epoch"],
            "Validation_MSE": result["Best_Validation_MSE"]
        })

        # Save completed runs
        result_dir.mkdir(parents=True, exist_ok=True)

        target_records = [
            row for row in records
            if row["Dataset"] == target
        ]

        pd.DataFrame(target_records).to_csv(
            result_dir / "architecture_search_results.csv",
            index=False
        )

        print(
            f"[Completed {run_idx}/{total_runs}] "
            f"Best Epoch={result['Best_Epoch']} | "
            f"Best Val MSE={result['Best_Validation_MSE']:.6f}",
            flush=True
        )

        # Release GPU memory
        del model, encoder
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(records)
