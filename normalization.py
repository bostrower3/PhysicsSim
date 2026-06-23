import torch
import torch.nn as nn

class Normalizer(nn.Module):
    """Feature Normalizer that accumulates statistics online"""
    def __init__(self,size,max_accumulations = 10 **6, std_epsilon = 1e-8):
        super().__init__()
        self.max_accumulations = max_accumulations
        self.std_epsilon = std_epsilon

        #Use buffers so that these are saved in the state dict, but not trainable
        self.register_buffer("acc_count",torch.tensor(0.0))
        self.register_buffer("num_accumulations",torch.tensor(0.0))
        self.register_buffer("acc_sum",torch.zeros(size))
        self.register_buffer("acc_sum_squared",torch.zeros(size))

    def forward(self,batched_data,accumulate = True):
        if accumulate and self.num_accumulations < self.max_accumulations:
            self._accumulate(batched_data)
        return (batched_data-self.mean_()) / self._std_with_epsilon()
    
    def inverse(self,normalized_data):
        return normalized_data * self._std_with_epsilon() + self._mean()
    
    def _accumulate(self,batched_data):
        with torch.no_grad():
            count = batched_data.size(0)
            data_sum = batched_data.sum(dim = 0)
            squared_sum = (batched_data ** 2).sum(dim = 0)

            self.acc_sum += data_sum
            self.acc_sum_squared += squared_sum
            self.acc_count += count
            self.num_accumulations += 1

    def _mean(self):
        safe_count = torch.clamp(self.acc_count,min = 1.0)
        return self.acc_sum / safe_count
    
    def _std_with_epsilon(self):
        safe_count = torch.clamp(self.acc_count,min = 1.0)
        mean = self._mean()
        std = torch.sqrt(self.acc_sum_squared / safe_count - mean ** 2)
        std = std.nan_to_num(self.std_epsilon)
        return torch.clamp(std,min = self.std_epsilon)