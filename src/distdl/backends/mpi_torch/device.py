import torch

# Also pass requested_device? Why?
def set_device(rank=0):

    # Set device based on rank
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Return current device
    return torch.cuda.current_device()

def get_device():
    return torch.cuda.current_device()