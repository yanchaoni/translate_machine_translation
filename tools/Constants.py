import torch

PAD = 0
SOS = 1
EOS = 2
UNK = 3
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")