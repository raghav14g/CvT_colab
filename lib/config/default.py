from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as op
import yaml
from yacs.config import CfgNode as CN

from lib.utils.comm import comm


_C = CN()

_C.BASE = ['']
_C.NAME = ''
_C.DATA_DIR = ''
_C.DIST_BACKEND = 'nccl'
_C.GPUS = (0,)
# _C.LOG_DIR = ''
_C.MULTIPROCESSING_DISTRIBUTED = True
_C.OUTPUT_DIR = ''
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 20
_C.RANK = 0
_C.VERBOSE = True
_C.WORKERS = 4
_C.MODEL_SUMMARY = False

_C.AMP = CN()
_C.AMP.ENABLED = False
_C.AMP.MEMORY_FORMAT = 'nchw'

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'cls_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.PRETRAINED_LAYERS = ['*']
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.SPEC = CN(new_allowed=True)
_C.MODEL.SPEC.CONV_PROJ = [True, True, True] # Default CvT mode
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LABEL_SMOOTHING = 0.0
_C.LOSS.LOSS = 'softmax'

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'imagenet'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.LABELMAP = ''
_C.DATASET.TRAIN_TSV_LIST = []
_C.DATASET.TEST_TSV_LIST = []
_C.DATASET.SAMPLER = 'default'

_C.DATASET.TARGET_SIZE = -1

# training data augmentation
_C.INPUT = CN()
_C.INPUT.MEAN = [0.485, 0.456, 0.406]
_C.INPUT.STD = [0.229, 0.224, 0.225]

# data augmentation
_C.AUG = CN()
_C.AUG.SCALE = (0.08, 1.0)
_C.AUG.RATIO = (3.0/4.0, 4.0/3.0)
_C.AUG.COLOR_JITTER = [0.4, 0.4, 0.4, 0.1, 0.0]
_C.AUG.GRAY_SCALE = 0.0
_C.AUG.GAUSSIAN_BLUR = 0.0
_C.AUG.DROPBLOCK_LAYERS = [3, 4]
_C.AUG.DROPBLOCK_KEEP_PROB = 1.0
_C.AUG.DROPBLOCK_BLOCK_SIZE = 7
_C.AUG.MIXUP_PROB = 0.0
_C.AUG.MIXUP = 0.0
_C.AUG.MIXCUT = 0.0
_C.AUG.MIXCUT_MINMAX = []
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'
_C.AUG.MIXCUT_AND_MIXUP = False
_C.AUG.INTERPOLATION = 2
_C.AUG.TIMM_AUG = CN(new_allowed=True)
_C.AUG.TIMM_AUG.USE_LOADER = False
_C.AUG.TIMM_AUG.USE_TRANSFORM = False

# train
_C.TRAIN = CN()

_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.LR_SCHEDULER = CN(new_allowed=True)
_C.TRAIN.SCALE_LR = True
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.OPTIMIZER_ARGS = CN(new_allowed=True)
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.WITHOUT_WD_LIST = []
_C.TRAIN.NESTEROV = True
# for adam
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 100

_C.TRAIN.IMAGE_SIZE = [224, 224]  # width * height, ex: 192 * 256
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

_C.TRAIN.EVAL_BEGIN_EPOCH = 0

_C.TRAIN.DETECT_ANOMALY = False

_C.TRAIN.CLIP_GRAD_NORM = 0.0
_C.TRAIN.SAVE_ALL_MODELS = False

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.CENTER_CROP = True
_C.TEST.IMAGE_SIZE = [224, 224]  # width * height, ex: 192 * 256
_C.TEST.INTERPOLATION = 2
_C.TEST.MODEL_FILE = ''
_C.TEST.REAL_LABELS = False
_C.TEST.VALID_LABELS = ''

_C.FINETUNE = CN()
_C.FINETUNE.FINETUNE = False
_C.FINETUNE.USE_TRAIN_AUG = False
_C.FINETUNE.BASE_LR = 0.003
_C.FINETUNE.BATCH_SIZE = 512
_C.FINETUNE.EVAL_EVERY = 3000
_C.FINETUNE.TRAIN_MODE = True
# _C.FINETUNE.MODEL_FILE = ''
_C.FINETUNE.FROZEN_LAYERS = []
_C.FINETUNE.LR_SCHEDULER = CN(new_allowed=True)
_C.FINETUNE.LR_SCHEDULER.DECAY_TYPE = 'step'

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
# Add the SWA Configuration Block
_C.SWA = CN()
_C.SWA.ENABLED = False
_C.SWA.START_EPOCH = 50
_C.SWA.LR_FACTOR = 0.05
_C.SWA.ANNEAL_EPOCHS = 10
#CNN+
# 1. Define all missing Top-Level Nodes (Critical Fix)
_C.SYSTEM = CN() 
_C.OUTPUT = CN()
_C.DISTRIBUTED = CN()
_C.CHECKPOINT = CN() # <--- Must be defined before its sub-keys are set

# 2. CNN-Specific Configuration Node (Crucial for your new model)
_C.CNN_SPEC = CN()
_C.CNN_SPEC.DEPTHS = [2, 2, 2, 2]         # Blocks per stage (e.g., ResNet-18)
_C.CNN_SPEC.CHANNELS = [64, 64, 128, 256, 512] # Channels: Stem output, Stage 1-4

# 3. Model Configuration Updates
_C.MODEL.NAME = 'VanillaCNN' # Default name
# ... (Other model parameters) ...
_C.DATA = CN()
_C.VALIDATION = CN()
# 4. DATA Configuration Updates
_C.DATA.DATASET = 'imagenet'
_C.DATA.BATCH_SIZE = 64
_C.DATA.CROP_PCT = 0.875
_C.DATA.IMG_SIZE = 224
_C.DATA.PIN_MEMORY = True    # Fixes likely error
_C.DATA.SHUFFLE = True

# 5. TRAIN Configuration Updates
_C.TRAIN.EPOCHS = 100
_C.TRAIN.LR = 0.01
_C.TRAIN.SCHEDULER = 'cosine' # Used by the script logic
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0001
_C.TRAIN.WARMUP_EPOCHS = 5

# Nested LR_SCHEDULER structure (Fixes AttributeError: METHOD)
_C.TRAIN.LR_SCHEDULER.METHOD = 'timm'
_C.TRAIN.LR_SCHEDULER.ARGS = CN(new_allowed=True)
#_C.TRAIN.LR_SCHEDULER.ARGS.sched = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30

# 6. VALIDATION Configuration Updates
_C.VALIDATION.VALIDATION_START_EPOCH = 0
_C.VALIDATION.BATCH_SIZE = 64
_C.VALIDATION.IMG_SIZE = 224

# 7. CHECKPOINT Configuration Updates
_C.CHECKPOINT.RESUME_FROM = '' 
_C.CHECKPOINT.AUTO_RESUME = False # Fixes likely error
#CNN-
def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, op.join(op.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    config.merge_from_list(args.opts)
    if config.TRAIN.SCALE_LR:
        config.TRAIN.LR *= comm.world_size
    file_name, _ = op.splitext(op.basename(args.cfg))
    config.NAME = file_name + config.NAME
    config.RANK = comm.rank

    if 'timm' == config.TRAIN.LR_SCHEDULER.METHOD:
        config.TRAIN.LR_SCHEDULER.ARGS.epochs = config.TRAIN.END_EPOCH

    if 'timm' == config.TRAIN.OPTIMIZER:
        config.TRAIN.OPTIMIZER_ARGS.lr = config.TRAIN.LR

    aug = config.AUG
    if aug.MIXUP > 0.0 or aug.MIXCUT > 0.0 or aug.MIXCUT_MINMAX:
        aug.MIXUP_PROB = 1.0
    config.freeze()


def save_config(cfg, path):
    if comm.is_main_process():
        with open(path, 'w') as f:
            f.write(cfg.dump())


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

