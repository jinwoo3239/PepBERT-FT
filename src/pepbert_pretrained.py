
import importlib.util
import torch

from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from configs.common import MODEL_DIR


MODEL_NAME = "dzjxzyd/PepBERT-large-UniParc"
PRETRAINED_DIR = MODEL_DIR / "pretrained" / "pepbert"

REQUIRED_FILES = [
    "model.py",
    "config.py",
    "tokenizer.json",
    "tmodel_17.pt"
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pretrained_pepbert():

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    # Download missing pretrained files
    if not all(
        (PRETRAINED_DIR / name).is_file()
        for name in REQUIRED_FILES
    ):
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=str(PRETRAINED_DIR),
            allow_patterns=REQUIRED_FILES
        )

    # Load original architecture and configuration
    model_module = load_module(
        "pepbert_original_model",
        PRETRAINED_DIR / "model.py"
    )

    config_module = load_module(
        "pepbert_original_config",
        PRETRAINED_DIR / "config.py"
    )

    pretrained_config = config_module.get_config()

    # Load tokenizer
    tokenizer = Tokenizer.from_file(
        str(PRETRAINED_DIR / "tokenizer.json")
    )

    # Load pretrained weights
    checkpoint = torch.load(
        PRETRAINED_DIR / "tmodel_17.pt",
        map_location="cpu",
        weights_only=True
    )

    pretrained_weights = checkpoint["model_state_dict"]

    return (
        model_module,
        pretrained_config,
        tokenizer,
        pretrained_weights
    )


def build_pretrained_encoder(
    model_module,
    pretrained_config,
    tokenizer,
    pretrained_weights
):

    # Initialize fresh PepBERT encoder
    encoder = model_module.build_transformer(
        src_vocab_size=tokenizer.get_vocab_size(),
        src_seq_len=pretrained_config["seq_len"],
        d_model=pretrained_config["d_model"]
    )

    # Restore original pretrained weights
    encoder.load_state_dict(
        pretrained_weights,
        strict=True
    )

    return encoder
