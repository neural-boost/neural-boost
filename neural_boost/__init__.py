import torch
import os

cwd = os.path.dirname(os.path.abspath(__file__))
torch.classes.load_library(cwd+"/libneural_boost.so")
print(torch.classes.loaded_libraries)
