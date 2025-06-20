# main.py
from model import AlexNet, LSTM, ViT
from dataload import get_dataLoader
from train import train
import torch

if __name__ == "__main__":
    train_loader, test_loader = get_dataLoader(batch_size=64)

    model_name = 'CNN' 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'CNN':
        model = AlexNet()
    elif model_name == 'RNN':
        model = LSTM()
    elif model_name == 'ViT':
        model = ViT()
    else:
        raise ValueError("!!! can't find ur model define !!!")
    
    # 简单的超参数设置
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 1e-3


    train(model, train_loader, test_loader, epochs=EPOCHS, lr=LR, device=device)
