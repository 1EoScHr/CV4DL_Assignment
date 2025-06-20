import torch.nn as nn
import torch

"""
CNN
使用基于AlexNet的结构，原输入是224x224，要修改以适应CIFAR10的32x32
"""
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.mainbody = nn.Sequential(          # (3,32,32)  ->
            nn.Conv2d(3, 64, 3, padding=1),     # (64,32,32) ->
            nn.ReLU(),
            nn.MaxPool2d(2),                    # (64,16,16) ->

            nn.Conv2d(64, 128, 3, padding=1),   # (128,16,16)->
            nn.ReLU(),
            nn.MaxPool2d(2),                    # (128,8,8)  ->

            nn.Conv2d(128, 256, 3, padding=1),  # (256,8,8)  ->
            nn.ReLU(),
            nn.MaxPool2d(2)                     # (256,4,4)  ->

            nn.Flatten(start_dim=1)             # (256*4*4)
        )

        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.mainbody(x)
        return self.fc(x)
    

"""
RNN
使用2层LSTM(Long Short-Term Memory)单元处理图像，将每张图像看作32行、每行有32*3的向量
"""
class LSTM(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=128, num_layers=2, num_classes=10):
        super().__init__()

        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.size()       # (B,3,32,32)，需要显式Batch维度
        x = x.permute(0, 2, 3, 1)   # (B,32,32,3)
        x = x.reshape(B, H, W*C)    # (B,32,96)，重排以满足32行、32*3列
        _, (hn, _) = self.rnn(x)    # (num_layers,B,hidden_dim)
        out = hn[-1]                # 只关心最后一层、最后时刻的状态，作为总结

        return self.fc(out)


"""
Transformer
使用简化的ViT(Vision Transformer)，将图像划分为4×4的patch，总共(32/4)²=64个patch
把图像划分为多个patch，当作一个“单词”输入Transformer，就像NLP中处理句子一样。
"""
class ViT(nn.Module):
    def __init__(self, patch_size=4, emb_dim=128, num_heads=4, num_layers=4, num_classes=10):
        super().__init__()

        # patch总数计算
        self.patch_size = patch_size
        self.num_patches = (32 // patch_size) ** 2 
        
        # (3,4,4) -> (128,)，一个全连接层，将每个patch转为128维向量
        self.embed = nn.Linear(3 * patch_size * patch_size, emb_dim)

        # 位置编码，Transformer是无序的，但是需要让模型知道每个patch对应的有空间位置
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

        # 构建一个transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出fc层
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.size()
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(B, self.num_patches, -1)  # (B, N, 48)
        tokens = self.embed(patches) + self.pos_embedding
        x = self.transformer(tokens)      # (B, N, D)
        x = x.mean(dim=1)                 # average over all tokens
        return self.fc(x)