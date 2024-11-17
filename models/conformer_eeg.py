import torch
from torch import nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
import pickle
import scipy
from torch.onnx.symbolic_opset9 import tensor


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, channel_size, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (channel_size, 1), (1, 1)),
            nn.BatchNorm2d(40), # 进行批量归一化
            nn.ELU(), # 激活函数
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x) # 64, 40, 1, 532
        x = self.projection(x) # 64, 532, 40
        return x


if __name__ == '__main__':
    data = torch.randn(64, 1, 32, 8064)
    model = PatchEmbedding(32)
    output = model(data)
    print(output.shape) # torch.Size([64, 532, 40])

    model2 = nn.Linear(532, 256)
    output2 = model2(output)
    print(output2.shape)

    a =  torch.randn(40, 1, 40, 8064)
    m1 = nn.Conv2d(1, 40, (1, 25), (1, 1))
    b = m1(a)
    print(b.shape) # 40, 40, 40, 8040
    m2 =  nn.Conv2d(40, 40, (40, 1), (1, 1))
    c = m2(b)
    print(c.shape) # 40, 40, 1, 8040