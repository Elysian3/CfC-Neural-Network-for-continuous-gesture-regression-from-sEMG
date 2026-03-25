import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from ncps.wirings import AutoNCP
from ncps.torch import LTC, CfC


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

def main():
    N = 48 # Length of the time-series
    # Input feature is a sine and a cosine wave
    data_x = np.stack(
        [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))], axis=1
    )
    data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
    # Target output is a sine with double the frequency of the input signal
    data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
    print("data_x.shape: ", str(data_x.shape))
    print("data_y.shape: ", str(data_y.shape))
    data_x = torch.Tensor(data_x)
    data_y = torch.Tensor(data_y)
    
    # 关键点 1: Windows 下必须为 num_workers=0
    dataloader = data.DataLoader(
        data.TensorDataset(data_x, data_y), batch_size=1, shuffle=True, num_workers=0
    )

    # Let's visualize the training data
    sns.set_theme()
    plt.figure(figsize=(6, 4))
    plt.plot(data_x[0, :, 0], label="Input feature 1")
    plt.plot(data_x[0, :, 1], label="Input feature 2") # 修改图例
    plt.plot(data_y[0, :, 0], label="Target output")
    plt.ylim((-1, 1))
    plt.title("Training data")
    plt.legend(loc="upper right")
    plt.show()

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

    sns.set_style("white")
    plt.figure(figsize=(6, 4))
    legend_handles = wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()

    # Let's visualize how LTC initialy performs before the training
    sns.set_theme()
    with torch.no_grad():
        prediction = cfc_model(data_x)[0].numpy()
    plt.figure(figsize=(6, 4))
    plt.plot(data_y[0, :, 0], label="Target output")
    plt.plot(prediction[0, :, 0], label="NCP output")
    plt.ylim((-1, 1))
    plt.title("Before training")
    plt.legend(loc="upper right")
    plt.show()

    # Train the model for 400 epochs (= training steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.fit(learn, dataloader)
    cfc_model.to(device)

    # How does the trained model now fit to the sinusoidal function?
    sns.set_theme()
    with torch.no_grad():
        prediction = cfc_model(data_x)[0].cpu().numpy() # 添加 .cpu() 防报错
    plt.figure(figsize=(6, 4))
    plt.plot(data_y[0, :, 0], label="Target output")
    plt.plot(prediction[0, :, 0], label="NCP output")
    plt.ylim((-1, 1))
    plt.title("After training")
    plt.legend(loc="upper right")
    plt.show()

# 关键点 3: 主入口保护
if __name__ == "__main__":
    main()