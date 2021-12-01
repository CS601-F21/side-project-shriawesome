class ModelConfig:
    # File location for generating the dataset
    DATA_CONFIG_FILE = "../dataset/metadata/words.txt"

    # Maximum text size for the input text
    MAX_TEXT_LEN = 21

    # Padding Token
    PAD_TOKEN = 99

    # Batch size for processing
    BATCH_SIZE = 64

    # Image widht and height
    IMG_WIDTH = 128
    IMG_HEIGHT = 32

    # Epochs for training
    EPOCHS = 10
    EARLY_STOP_PATIENCE = 10

    # MODEL NAME TO BE SAVED
    SAVE_MODEL = "../models/HWRModel_v4.h5"

