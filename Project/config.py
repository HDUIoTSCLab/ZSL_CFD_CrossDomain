SAMPLE_LENGTH = 2048
fft_dim = SAMPLE_LENGTH // 2
MAX_SAMPLES_PER_CLASS = 100
GENERATE_COMPOUND_FAULT_NUM = 1
SOURCE_DIR="./Data/S1000"
TARGET_DIR="./Data/S1400"

SAMPLES_PER_SIMPLE_CLASS = {
    'N':  {'dir': './Data/F0', 'num': MAX_SAMPLES_PER_CLASS},
    'IF': {'dir': './Data/F0', 'num': 10},
    'BF': {'dir': './Data/F0', 'num': 10},
    'OF': {'dir': './Data/F0', 'num': 10},
    'CF': {'dir': './Data/F0', 'num': 10}
}

TEST_DIRS = [
    #"./Data/F0", "./Data/F200", "./Data/F400","./Data/F600","./Data/F800", "./Data/F1000", "./Data/F1200"
    #"./Data/S400", "./Data/S600", "./Data/S800","./Data/S1000", "./Data/S1200", "./Data/S1400", "./Data/S1600"
    TARGET_DIR
    #"./Data/F0", "./Data/F200", "./Data/F400","./Data/F800", "./Data/F1000", "./Data/F1200"
    #"./Data/S400", "./Data/S600", "./Data/S800","./Data/S1200", "./Data/S1400", "./Data/S1600"
]

COMPOSITE_DEF = {
    'IOF':  {'parts': ['IF', 'OF'],     'num': MAX_SAMPLES_PER_CLASS},
    'IBF':  {'parts': ['IF', 'BF'],     'num': MAX_SAMPLES_PER_CLASS},
    'ICF':  {'parts': ['IF', 'CF'],     'num': MAX_SAMPLES_PER_CLASS},
    'OBF':  {'parts': ['OF', 'BF'],     'num': MAX_SAMPLES_PER_CLASS},
    'IOBF': {'parts': ['IF', 'OF', 'BF'],'num': MAX_SAMPLES_PER_CLASS}
}

TRAIN_TYPES = list(SAMPLES_PER_SIMPLE_CLASS.keys())
TEST_COMPOSITE = list(COMPOSITE_DEF.keys())