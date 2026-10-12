import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import glob
import random

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前运算设备：", DEVICE)
if torch.cuda.is_available():
    print("显卡型号：", torch.cuda.get_device_name(0))

# RTX4060 8G最优批次
BATCH_SIZE = 32
EPOCHS = 30
IMG_SIZE = 224
CLASS_NUM = 3

DATA_PATH = r"c:\Users\27890\Desktop\叶子分类数据集"

# 数据集划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 固定随机种子（CPU+GPU双同步，实验可复现）
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ===================== 网络模型 =====================
def get_mobilenetv2_model(num_classes):
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

# ===================== 数据集类 =====================
class LeafDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.img_paths = []
        self.labels = []
        for label in range(CLASS_NUM):
            folder = os.path.join(root_dir, str(label))
            imgs = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
            for path in imgs:
                self.img_paths.append(path)
                self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ===================== 分层数据集划分 =====================
def split_dataset(dataset):
    label_indices = [[] for _ in range(CLASS_NUM)]
    for idx, lab in enumerate(dataset.labels):
        label_indices[lab].append(idx)

    train_idx_all = []
    val_idx_all = []
    test_idx_all = []

    for lab_idx_list in label_indices:
        random.shuffle(lab_idx_list)
        total = len(lab_idx_list)
        train_num = int(total * TRAIN_RATIO)
        val_num = int(total * VAL_RATIO)

        train_idx = lab_idx_list[:train_num]
        val_idx = lab_idx_list[train_num:train_num+val_num]
        test_idx = lab_idx_list[train_num+val_num:]

        train_idx_all.extend(train_idx)
        val_idx_all.extend(val_idx)
        test_idx_all.extend(test_idx)

    train_set = Subset(dataset, train_idx_all)
    val_set = Subset(dataset, val_idx_all)
    test_set = Subset(dataset, test_idx_all)
    return train_set, val_set, test_set

# ===================== 图像预处理 =====================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=1.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== 验证/测试评估函数 =====================
def evaluate(model, loader, loss_func):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labs in loader:
            # non_blocking 异步数据迁移，不阻塞GPU计算
            imgs = imgs.to(DEVICE, non_blocking=True)
            labs = labs.to(DEVICE, non_blocking=True)
            out = model(imgs)
            loss = loss_func(out, labs)
            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc

# ===================== 主训练入口 =====================
if __name__ == "__main__":
    full_dataset = LeafDataset(DATA_PATH, transform=train_transform)
    print(f"数据集总图片：{len(full_dataset)}")

    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform
    print(f"训练集：{len(train_dataset)} 张 | 验证集：{len(val_dataset)} 张 | 测试集：{len(test_dataset)} 张")

    # 终端脚本支持多线程 num_workers=4，大幅提速读图，无jupyter死锁问题
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = get_mobilenetv2_model(num_classes=CLASS_NUM).to(DEVICE)
    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # FP16混合精度，RTX4060显卡提速30%~50%，降低显存占用
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0
    best_model_path = "best_mobilenetv2_leaf.pth"

    print("\n===== 开始GPU训练 MobileNetV2 叶片分类网络 =====")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, labs in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labs = labs.to(DEVICE, non_blocking=True)

            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = loss_func(out, labs)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss, val_acc = evaluate(model, val_loader, loss_func)
        print(f"第{epoch+1:2d}轮 | 训练损失:{train_loss:.4f} | 验证损失:{val_loss:.4f} | 验证准确率:{val_acc:.4f}")

        # 保存最优权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ 保存最优模型，当前最高验证准确率：{best_val_acc:.4f}")
        
        # ========== 导出ONNX模型 ==========
        import torch.onnx

        # 1. 加载训练好的最优权重
        model.load_state_dict(torch.load("best_mobilenetv2_leaf.pth", map_location=DEVICE))
        model.eval()

        # 2. 构造和输入图片尺寸一致的假输入 (224*224 RGB)
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

        # 3. 导出onnx文件
        torch.onnx.export(
        model,
        dummy_input,
        "leaf_mobilenetv2.onnx",  # 生成的onnx文件名
        input_names=["input"],
        output_names=["output"],
        opset_version=13  # 兼容大部分推理框架
        )
        print("✅ ONNX模型已导出：leaf_mobilenetv2.onnx")

    print("\n===== 训练结束，加载最优模型测试 =====")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_loss, test_acc = evaluate(model, test_loader, loss_func)
    print(f"测试集损失：{test_loss:.4f} | 测试集最终准确率：{test_acc:.4f}")

