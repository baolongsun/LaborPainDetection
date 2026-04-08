import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import logging
from collections import defaultdict
import torch.nn.functional as F

# =======================
# 1️⃣ 自定义 ImageDataset (从 txt 文件读取)
# =======================
class ImageDataset(Dataset):
    def __init__(self, label_file, root_dir="", transform=None):
        """
        label_file: txt文件路径，每行: image_path label
        root_dir:   图片文件根目录，可为空（label_file中是绝对路径时）
        """
        self.root_dir = root_dir
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
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


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
# 2️⃣ ResNet 模型
# =======================
class ResNetModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# =======================
# 3️⃣ 获取类别样本数
# =======================
def get_class_count(label_file):
    counts = defaultdict(int)
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            label = line.strip().split()[1]
            counts[label] += 1
    return counts


# =======================
# 4️⃣ 训练函数（含保存）
# =======================
def train_model(model, dataloader, criterion,criterion_reg, optimizer, device, num_epochs=10, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            # loss = criterion(outputs, labels)
            loss = criterion(outputs, labels) + criterion_reg(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Acc: {acc:.3f}")

        # 保存 checkpoint
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1+START_EPOCH}.pth")
        if epoch % 2 == 0:
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"Checkpoint saved: {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            logging.info(f"Best model updated at epoch {epoch+1}, loss={avg_loss:.4f}")


# =======================
# 5️⃣ 主程序
# =======================
if __name__ == "__main__":
    # 配置
    for seed in ['2104','3013','4901','8221','9985','768','3871','4633','5346','8109']:
        # seed = 2104
        from tqdm import tqdm
        START_EPOCH = 0
        for idx in range(6):
            label_file = f"./exp_simple/{seed}_6fold/train_{idx}.txt"
            root_dir = r'C:'
            num_classes = 4
            batch_size = 64
            num_epochs = 2
            lr = 1e-4
            device = "cuda" if torch.cuda.is_available() else "cpu"
            save_dir = f'c:/resnet_checkpoint_1207_112/{seed}/{seed}_{idx}_reg1_cross1_1217'
            os.makedirs(save_dir, exist_ok=True)
            # model_path = f'resnet_checkpoint/{seed}/{seed}_{idx}_reg1_cross1/epoch_2.pth'
            # Logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(f'{save_dir}/training.log'),
                    logging.StreamHandler()
                ]
            )

            # 数据预处理
            transform = transforms.Compose([
                transforms.Resize((128, 171)),  # 128，171
                transforms.CenterCrop(112),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                     std=[0.225, 0.225, 0.225]),
            ])

            # 数据加载
            dataset = ImageDataset(label_file, root_dir=root_dir, transform=transform)
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=4, pin_memory=True)

            # 类别权重
            class_count = get_class_count(label_file)
            counts = torch.tensor(list(class_count.values()), dtype=torch.float)
            weights = 1.0 / counts
            weights = weights / weights.sum() * len(class_count)
            logging.info(f'Class weights: {weights.tolist()}')
            weights = weights.to(device)

            # 模型 & 优化器 & 损失
            model = ResNetModel(num_classes=num_classes, pretrained=True)
            # model.load_state_dict(torch.load(model_path, map_location=device))
            # print(f'加载模型成功:{model_path}')
            criterion = nn.CrossEntropyLoss(weight=weights)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion_reg = WeightedOrdinalRegressionLoss(class_weights=weights, distance_power=2)

            # 开始训练
            train_model(model, dataloader, criterion, criterion_reg, optimizer, device, num_epochs, save_dir)
