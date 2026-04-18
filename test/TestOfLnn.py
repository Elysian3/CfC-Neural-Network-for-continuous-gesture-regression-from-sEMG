import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from ncps.wirings import AutoNCP
from ncps.torch import LTC, CfC
import os

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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. 获取项目根目录 (向上退一级，到达 Project Antikythera)
    project_root = os.path.dirname(current_dir)

    # 3. 拼接出数据文件夹的绝对路径
    base_path = os.path.join(project_root, "src", "data", "LowerLimbTestData")

    filenames = ["1Amar.txt", "1Apie.txt", "1Asen.txt"]

    dataset = {}

    for filename in filenames:
        file_path = os.path.join(base_path, filename)

        # 使用 numpy 加载数据。如果数据包含表头或逗号分隔，可以添加 skiprows/delimiter 参数
        data = np.loadtxt(file_path)

        # 使用切片分离数据
        # data[:, :-1] 表示提取除了最后一列之外的所有行和列 (特征数据)
        # data[:, -1] 表示只提取最后一列的所有行 (通常标签或目标值)
        features = data[:, :-1]
        last_column = data[:, -1]

        dataset[filename] = {
            "features": features,
            "last_column": last_column
        }

        print(f"文件 {filename} 加载成功:")
        print(f" -> 总数据形状: {data.shape}")
        print(f" -> 前面的列形状: {features.shape}")
        print(f" -> 最后一列形状: {last_column.shape}\n")
    
    # 关键点 1: Windows 下必须为 num_workers=0
    # dataloader = data.DataLoader(
    #     data.TensorDataset(data_x, data_y), batch_size=1, shuffle=True, num_workers=0
    # )
    #
    # # Let's visualize the training data
    # sns.set_theme()
    # plt.figure(figsize=(6, 4))
    # plt.plot(data_x[0, :, 0], label="Input feature 1")
    # plt.plot(data_x[0, :, 1], label="Input feature 2") # 修改图例
    # plt.plot(data_y[0, :, 0], label="Target output")
    # plt.ylim((-1, 1))
    # plt.title("Training data")
    # plt.legend(loc="upper right")
    # plt.show()
    #
    # out_features = 1
    # in_features = 2
    #
    # wiring = AutoNCP(16, out_features)  # 16 units, 1 motor neuron
    # # 提取自动生成的线虫邻接矩阵
    #
    # cfc_model = CfC(in_features, wiring, batch_first=True)
    # learn = SequenceLearner(cfc_model, lr=0.01)
    #
    # # 关键点 2: 使用 auto 自动寻找可用的硬件，不会强行崩溃
    # trainer = pl.Trainer(
    #     logger=pl.loggers.CSVLogger("log"),
    #     max_epochs=400,
    #     gradient_clip_val=1,  # Clip gradient to stabilize training
    #     accelerator="auto",
    #     devices="auto"
    # )
    #
    # sns.set_style("white")
    # plt.figure(figsize=(6, 4))
    # legend_handles = wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
    # plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    # sns.despine(left=True, bottom=True)
    # plt.tight_layout()
    # plt.show()
    #
    # # Let's visualize how LTC initialy performs before the training
    # sns.set_theme()
    # with torch.no_grad():
    #     prediction = cfc_model(data_x)[0].numpy()
    # plt.figure(figsize=(6, 4))
    # plt.plot(data_y[0, :, 0], label="Target output")
    # plt.plot(prediction[0, :, 0], label="NCP output")
    # plt.ylim((-1, 1))
    # plt.title("Before training")
    # plt.legend(loc="upper right")
    # plt.show()
    #
    # # Train the model for 400 epochs (= training steps)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # trainer.fit(learn, dataloader)
    # cfc_model.to(device)
    #
    # # How does the trained model now fit to the sinusoidal function?
    # sns.set_theme()
    # with torch.no_grad():
    #     prediction = cfc_model(data_x)[0].cpu().numpy() # 添加 .cpu() 防报错
    # plt.figure(figsize=(6, 4))
    # plt.plot(data_y[0, :, 0], label="Target output")
    # plt.plot(prediction[0, :, 0], label="NCP output")
    # plt.ylim((-1, 1))
    # plt.title("After training")
    # plt.legend(loc="upper right")
    # plt.show()

# 关键点 3: 主入口保护
if __name__ == "__main__":
    main()