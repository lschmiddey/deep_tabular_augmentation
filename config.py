import torch

# device on which to train
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')