from yacs.config import CfgNode as CN

_C = CN()
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
_C.DEVICE = True
_C.DEVICE_ID = '1'

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.BACKBONE = "VGG"
_C.MODEL.NAME = "SSD"
_C.MODEL.STRIDE = 2
_C.MODEL.RESUM = False
_C.MODEL.BACKBONE_WEIGHTS = "./backbone_weights"
_C.MODEL.SAVE_MODEL_FRE = 5
_C.MODEL.TRAINED_MODEL = ""
_C.MODEL.TOP_K = 5
_C.MODEL.CONFIDENCE_THRE = 0.01
_C.MODEL.VAL_GAP = 1000
_C.MODEL.FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
_C.MODEL.MIN_DIM = 300
_C.MODEL.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
_C.MODEL.VARIANCE = [0.1, 0.2]
_C.MODEL.CLIP = True
_C.MODEL.MIN_SIZES = [30, 60, 111, 162, 213, 264]
_C.MODEL.MAX_SIZES = [60, 111, 162, 213, 264, 315]
_C.MODEL.STEPS = [8, 16, 32, 64, 100, 300]
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
# _C.INPUT.HF_PROB = 0.5

# Values to be used for image normalization
# _C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
_C.INPUT.PIXEL_MEAN = [104, 117, 123]
# Values to be used for image normalization
# _C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# Value of padding size
# _C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
# _C.DATASETS.NAME = ''
# Setup storage directory for dataset
_C.DATASETS.ROOT = '/data/shw/dolphinDetect'
_C.DATASETS.NAME = "JT001"
_C.DATASETS.MIN_DIM = 300
_C.DATASETS.TEST_DIR = 'Test'
_C.DATASETS.NUM_CLS = 1
_C.DATASETS.MEANS = (104, 117, 123)
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# Number of instance for one batch
_C.DATALOADER.BATCH_SIZE = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Sampler for data loading
_C.SOLVER.LOSS = 'softmax'
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.MAX_EPOCHS = 12000
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY=5e-4
# SGD
# _C.SOLVER.BASE_LR = 0.01
_C.SOLVER.NESTEROV = True

# Adam
_C.SOLVER.WEIGHT_DECAY = 0.0005

_C.SOLVER.EVAL_PERIOD = 50
_C.SOLVER.PRINT_FREQ = 10

_C.SCHEDULER = CN()
_C.SCHEDULER.NAME = 'StepLR'
_C.SCHEDULER.STEP = [8000, 10000, 12000]
_C.SCHEDULER.GAMMA = 0.1

# Warm up factor
_C.SCHEDULER.WARMUP_FACTOR = 100
# iterations of warm up
_C.SCHEDULER.WARMUP_ITERS = 20

# Show train log
_C.SHOW = CN()
_C.SHOW.PRINT_FREQ = 1
_C.SHOW.VISDOM = False

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()

_C.TEST.BATCH_SIZE = 128
_C.TEST.SET_TYPE = 'test'
_C.TEST.RESULT = '/home/shw/code/ZhiXing/checkpoint/Exp-1/'