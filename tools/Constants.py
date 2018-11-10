import torch

PAD = 0
SOS = 1
EOS = 2
UNK = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")