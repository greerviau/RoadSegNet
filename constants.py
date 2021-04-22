# Constants
CLASS_VALUES = {(64, 32, 32) : 0, 
                (255, 0, 0) : 1, 
                (128, 128, 96) : 2, 
                (0, 255, 102) : 3, 
                (204, 0, 255) : 4}

NUM_CLASSES = len(CLASS_VALUES)
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = NUM_CLASSES
BATCH_SIZE = 2

RESOLUTION = (400,300)

MAX_EPOCHS = 6000

LEARNING_RATE = 1e-6
MOMENTUM = 0.9
BATCH_SIZE = 6