
import gc
import json
import torch
import pandas as pd

from torch.utils.data import DataLoader, Subset

from configs.common import MODEL_DIR, RESULT_DIR, RANDOM_SEEDS
from configs import pepbert_grid as config

from src.seed import set_seed
from src.ml import load_validation_split
from src.pepbert import PepBERTRegressor, train_pepbert
from src.pepbert_pretrained import build_pretrained_encoder


def run_grid_search(
    grid_tasks,
    train_datasets,
    model_module,
    pretrained_config,
    tokenizer,
    pretrained_weights,
    device
):
    # Define all training runs
    tasks = [
        (params, seed)
        for params in grid_tasks
        for seed in RANDOM_SEEDS
    ]

    total_runs = len(tasks)
    records = []

    # Run hyperparameter optimization
    for run_idx, (params, seed) in enumerate(tasks, 1):

        target = params["Dataset"]
        pooling = params["Pooling"]
        fc_layers = int(params["FC_Layers"])
        lr = float(params["Learning_Rate"])
        wd = float(params["Weight_Decay"])
        dropout = float(params["Dropout"])

        set_seed(seed)

        # Define experiment name
        run_name = (
            f"{run_idx}/{total_runs} | {target} | "
            f"{pooling.upper()} | FC{fc_layers} | "
            f"LR={lr:.0e} | WD={wd:.0e} | "
            f"DO={dropout:.1f} | Seed={seed}"
        )

        # Load validation split
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

        # Create regression model
        model = PepBERTRegressor(
            encoder=encoder,
            hidden_size=pretrained_config["d_model"],
            pooling=pooling,
            fc_layers=fc_layers,
            dropout=dropout
        ).to(device)

        # Define output paths
        experiment_id = (
            f"FC{fc_layers}_"
            f"LR{lr:.0e}_"
            f"WD{wd:.0e}_"
            f"DO{dropout:.1f}_"
            f"seed{seed}"
        )

        model_dir = MODEL_DIR / "pepbert" / target / pooling / "grid_search"
        result_dir = RESULT_DIR / "pepbert" / target / pooling / "grid_search"

        save_path = model_dir / f"{experiment_id}.pt"
        history_path = result_dir / f"{experiment_id}_history.csv"

        # Fine-tune and validate
        result = train_pepbert(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            seed=seed,
            save_path=save_path,
            history_path=history_path,
            max_epochs=config.MAX_EPOCHS,
            patience=config.PATIENCE,
            lr=lr,
            weight_decay=wd,
            run_name=run_name
        )

        # Record validation results
        records.append({
            "Dataset": target,
            "Pooling": pooling,
            "FC_Layers": fc_layers,
            "Learning_Rate": lr,
            "Weight_Decay": wd,
            "Dropout": dropout,
            "Seed": seed,
            "Best_Epoch": result["Best_Epoch"],
            "Validation_MSE": result["Best_Validation_MSE"]
        })

        # Save completed runs
        output_dir = RESULT_DIR / "pepbert"
        output_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(records).to_csv(
            output_dir / "grid_search_results.csv",
            index=False
        )

        print(
            f"[Completed {run_idx}/{total_runs}] "
            f"{target} | {pooling.upper()} | FC{fc_layers} | "
            f"Best Val={result['Best_Validation_MSE']:.6f} | "
            f"Best Epoch={result['Best_Epoch']}",
            flush=True
        )

        # Release GPU memory
        del model, encoder
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(records)
