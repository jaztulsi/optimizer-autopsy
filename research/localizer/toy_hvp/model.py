import torch
import torch.nn as nn

torch.manual_seed(0)

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)   # 4*8 + 8 = 40 params
        self.fc2 = nn.Linear(8, 2)   # 8*2 + 2 = 18 params
        # total: 58 params

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

def count_params(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    model = TinyModel()
    print(f"Total params: {count_params(model)}")
    x = torch.randn(5, 4)
    y = torch.randn(5, 2)
    out = model(x)
    loss = ((out - y) ** 2).mean()
    print(f"Loss: {loss.item():.4f}")
    
