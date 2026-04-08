import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.video import r2plus1d_18
from tqdm import tqdm
import logging
import torch.nn.functional as F

from decord import VideoReader, cpu
import torch

def load_video(video_path, num_frames=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    idxs = torch.linspace(0, total_frames-1, num_frames).long()
    frames = vr.get_batch(idxs).asnumpy()
    video = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    return video

# 不平衡的类误差
class WeightedOrdinalRegressionLoss(nn.Module):
    def __init__(self, class_weights=None, distance_power=2):
        """
        class_weights: Tensor[num_classes]，类别权重（不平衡补偿）
        distance_power: 距离惩罚的幂次，=1 表示L1距离, =2 表示L2距离
        """
        super().__init__()
        self.class_weights = class_weights
        self.distance_power = distance_power

    def forward(self, logits, targets):
        """
        logits: (N, C)
        targets: (N,)  int类别标签
        """
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)

        # 计算预测的“疼痛等级期望值”
        class_indices = torch.arange(num_classes, device=logits.device).float()
        pred_score = torch.sum(probs * class_indices, dim=1)  # [N]
        true_score = targets.float()

        # 距离损失
        loss = torch.abs(pred_score - true_score) ** self.distance_power

        # 类别不平衡加权
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = loss * weights

        return loss.mean()

# =======================
# 1️⃣ 自定义 VideoDataset (从 txt 文件读取)
# =======================
class VideoDataset(Dataset):
    def __init__(self, label_file, root_dir="", num_frames=16, transform=None):
        """
        label_file: txt文件路径，每行: video_path label
        root_dir:   视频文件根目录，可为空（label_file中是绝对路径时）
        num_frames: 每个视频采样帧数
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.samples = []

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(root_dir, path)
                self.samples.append((full_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        video = load_video(video_path, num_frames=16)
        if self.transform:
            # 对每一帧执行 transform
            video = torch.stack([self.transform(frame) for frame in video.permute(1, 0, 2, 3)])
            video = video.permute(1, 0, 2, 3)

        return video, label


# =======================
# 2️⃣ R(2+1)D 模型
# =======================
class R2Plus1DModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.model = r2plus1d_18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# =======================
# 3️⃣ 训练函数
# =======================
def train_model_pre(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for videos, labels in tqdm(dataloader):
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * videos.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        logging.info(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Loss: {total_loss/total:.4f} | Acc: {correct/total:.3f}")


# =======================
# 3️⃣ 训练函数（含保存）
# =======================
def train_model(model, dataloader, criterion, criterion_reg, optimizer, device, num_epochs=10, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)

    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for videos, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels) + criterion_reg(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * videos.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Acc: {acc:.3f}")

        # ✅ 保存checkpoint
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
        # torch.save({
        #     "epoch": epoch + 1,
        #     "model_state": model.state_dict(),
        #     "optimizer_state": optimizer.state_dict(),
        #     "loss": avg_loss,
        # }, ckpt_path)
        torch.save(model.state_dict(), ckpt_path)
        logging.info(f"Checkpoint saved: {ckpt_path}")

        # ✅ 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            logging.info(f"Best model updated at epoch {epoch+1}, loss={avg_loss:.4f}")


def get_class_count(label_file):
    from collections import defaultdict
    aa = defaultdict(int)
    aa['0']
    aa['1']
    aa['2']
    aa['3']
    with open(label_file,'r',encoding='utf-8') as f:
        data = f.readlines()
        for d in data:
            tag = d.strip().split()[1]
            aa[tag] += 1
    return aa
# =======================
# 4️⃣ 主程序
# =======================
if __name__ == "__main__":
    # 配置
    for seed in ['768','2104','3013','3871','4633','4901','5346','8109','8221','9883','9985']:
        for idx in range(6):
            label_file = f"./datas/exp/{seed}_6fold/train_{idx}.txt"
            root_dir = ""  # 如果 label.txt 已经是绝对路径，可以设为空 ""
            num_classes = 4
            num_frames = 16
            batch_size = 8
            num_epochs = 4
            lr = 1e-4
            device = "cuda" if torch.cuda.is_available() else "cpu"
            save_dir = f'c3d_checkpoint_1130/11_trail/{seed}_6fold/{idx}'
            os.makedirs(save_dir,exist_ok=True)

            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(f'{save_dir}/training.log'),  # 输出到文件
                    logging.StreamHandler()  # 输出到控制台
                ]
            )

            # 预处理
            transform = transforms.Compose([
                transforms.Resize((128, 171)),#128，171
                transforms.CenterCrop(112),
                transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                     std=[0.225, 0.225, 0.225]),
            ])

            # 数据加载
            dataset = VideoDataset(label_file, root_dir=root_dir,
                                   num_frames=num_frames, transform=transform)
            dataloader = DataLoader(dataset, batch_size=batch_size,pin_memory=True,
                                    shuffle=True, num_workers=4)

            # 类别数量
            num_classes = 4
            # 样本数

            class_count = get_class_count(label_file)
            counts = torch.tensor(list(class_count.values()), dtype=torch.float)
            # 权重：样本少的类别权重大
            weights = 1.0 / counts
            # 归一化（可选）
            weights = weights / weights.sum() * num_classes
            logging.info(f'cross weight:{str(weights.numpy())}')
            weights = weights.to(device)
            # 模型 & 优化器
            model = R2Plus1DModel(num_classes=num_classes, pretrained=True)
            criterion = nn.CrossEntropyLoss(weight=weights)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion_reg = WeightedOrdinalRegressionLoss(class_weights=weights, distance_power=2)

            # 训练
            train_model(model, dataloader, criterion,criterion_reg, optimizer, device, num_epochs, save_dir)
