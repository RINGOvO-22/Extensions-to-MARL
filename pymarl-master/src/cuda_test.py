import torch

print(torch.__version__)
print(torch.version.cuda)
# If output is "Ture", then cuda is set successfully
print(torch.cuda.is_available())
