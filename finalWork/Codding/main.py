from model import AlexNet, LSTM, ViT
from dataload import get_dataLoader
from train import train
from torchinfo import summary
import torch


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataLoader(batch_size=64)

    model_name = 'ViT' 
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

    print("model used:" + model_name)
    summary(model, input_size=(BATCH_SIZE, 3, 32, 32), depth=1, col_names=["input_size", "output_size", "num_params"])
    train(model_name, model, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LR, device=device)
