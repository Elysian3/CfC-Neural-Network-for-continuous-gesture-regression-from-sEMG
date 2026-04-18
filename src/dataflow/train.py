import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from ncps.wirings import AutoNCP
from ncps.torch import LTC, CfC
import scipy as sp

# LightningModule for training a RNNSequence module
class SequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss  # 返回 loss tensor 本身

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)




def load_data_from_mat():
    
    pass

def main()
    

    load_data_from_mat()

    dataloader = data.DataLoader(
        data.TensorDataset(data_x, data_y), batch_size=1, shuffle=True, num_workers=0
    )
    
    out_features = 1
    in_features = 2
    

    wiring = AutoNCP(16, out_features)  # 16 units, 1 motor neuron
    # 提取自动生成的线虫邻接矩阵
    
    cfc_model = CfC(in_features, wiring, batch_first=True)
    learn = SequenceLearner(cfc_model, lr=0.01)
    
    # 关键点 2: 使用 auto 自动寻找可用的硬件，不会强行崩溃
    trainer = pl.Trainer(
        logger=pl.loggers.CSVLogger("log"),
        max_epochs=400,
        gradient_clip_val=1,  # Clip gradient to stabilize training
        accelerator="auto",
        devices="auto"
    )
    
    # Train the model for 400 epochs (= training steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.fit(learn, dataloader)
    cfc_model.to(device)
    
    with torch.no_grad():
        prediction = cfc_model(data_x)[0].cpu().numpy()  # 添加 .cpu() 防报错

if __name__ == '__main__':
    main()