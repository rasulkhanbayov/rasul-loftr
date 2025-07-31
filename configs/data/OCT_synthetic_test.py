from configs.data.base import cfg

TEST_BASE_PATH = "data/OCTsynthetic/test_new"

cfg.DATASET.TEST_DATA_SOURCE = "data/OCTsynthetic/test_new"
cfg.DATASET.TEST_DATA_ROOT = "data/OCTsynthetic/test_new"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/trainvaltest_list/val_list.txt"

cfg.DATASET.MGDPT_IMG_RESIZE = 256
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
