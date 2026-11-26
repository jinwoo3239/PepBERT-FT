
# Architecture search
POOLING_TYPES = ["sos", "mean"]
FC_LAYERS = [1, 2, 3, 4]

# Training parameters
BATCH_SIZE = 32
MAX_LENGTH = 52
MAX_EPOCHS = 100
PATIENCE = 10

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.2
LOSS_FUNCTION = "MSE"
