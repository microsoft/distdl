import torch

# Also pass requested_device? Why?
def set_device(rank=0):

    # Set device based on rank
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

        # Return current device
        return torch.cuda.current_device()
    else:
        return 'cpu'

def get_device():
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return 'cpu'