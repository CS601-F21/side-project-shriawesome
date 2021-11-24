class ModelConfig:
    # File location for generating the dataset
    DATA_CONFIG_FILE = "../dataset/metadata/lines.txt"

    # Maximum text size for the input text
    MAX_TEXT_LEN = 100

    # Batch size for processing
    BATCH_SIZE = 64

    # Image widht and height
    IMG_WIDTH = 800
    IMG_HEIGHT = 64

    # Epochs for training
    EPOCHS = 10
    EARLY_STOP_PATIENCE = 10

    # MODEL NAME TO BE SAVED
    SAVE_MODEL = "../models/HWRModel_v1.h5"

