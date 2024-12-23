import torch
from lightning.pytorch import Trainer
from model import MultiClassModel  # モデル定義
from dataModule import DataModule  # データモジュール

# モデルとデータモジュールのインスタンス化
model_path = "checkpoints/best-epoch=99-val_loss=0.24.ckpt"  # 学習時に保存された最良モデルのパス
model = MultiClassModel.load_from_checkpoint(model_path)

# データモジュールのインスタンス化
data_module = DataModule(dataset_path="data", batch_size=2)

# テストエポックの実行
trainer = Trainer(accelerator="gpu", devices=1)
trainer.test(model=model, datamodule=data_module)