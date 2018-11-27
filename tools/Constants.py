import torch

PAD = 0
SOS = 1
EOS = 2
UNK = 3
BATCH_SIZE = 32
MAX_WORD_LENGTH = [50, 50]
EMB_DIM = 300
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_RESULT_PATH = "results/"