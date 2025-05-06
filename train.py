import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models import DualBranchResNet
from utils.DataSets import CustomDataset
import config
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


def cosine_similarity_loss(features_v, features_d):
    """
    计算余弦相似度损失
    :param features_a: 动脉期分支的特征表示 [batch_size, feature_dim]
    :param features_v: 静脉期分支的特征表示 [batch_size, feature_dim]
    :return: 余弦相似度损失
    """
    # 计算余弦相似度
    cosine_sim = F.cosine_similarity(features_v, features_d, dim=1)  # [batch_size]
    # 计算损失
    loss = 1 - cosine_sim.mean()  # 对所有样本取平均
    return loss


# 设置参数
train_folder1 = './data/complete/train_v'  # 训练集的第一个文件夹路径
train_folder2 = './data/complete/train_d'  # 训练集的第二个文件夹路径
train_folder3 = './data/fat/train'  # 训练集的第二个文件夹路径
val_folder1 = './data/complete/test__v'  # 验证集的第一个文件夹路径
val_folder2 = './data/complete/test__d'  # 验证集的第二个文件夹路径
val_folder3 = './data/fat/test'  # 验证集的第二个文件夹路径

train_label_path = './data/train_excel.xlsx'  # 训练集标签路径
val_label_path = './data/test_excel.xlsx'  # 验证集标签路径
batch_size = 8
num_epochs = 200
learning_rate = 0.000001
img_size = (224, 224)

# 图像预处理
tra_transform = transforms.Compose([
    transforms.Resize(img_size),
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    # transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    # transforms.RandomRotation(15),
    # transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor(),
    # transforms.Normalize([0.0508267, 0.0508267, 0.0508267],
    #                      [0.16725375, 0.16725375, 0.16725375])
])
val_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    # transforms.Normalize([0.0508267, 0.0508267, 0.0508267],
    #                      [0.16725375, 0.16725375, 0.16725375])
])

# 初始化训练集和验证集的数据集和数据加载器
train_dataset = CustomDataset(folder1=train_folder1, folder2=train_folder2,folder3=train_folder3,
                              label_path=train_label_path,
                              transform=tra_transform)
val_dataset = CustomDataset(folder1=val_folder1, folder2=val_folder2,folder3=train_folder3, label_path=val_label_path,
                            transform=val_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualBranchResNet.DualBranchConvNext().to(device)

# 定义损失函数
criterion_ce = nn.CrossEntropyLoss()  # 交叉熵损失
# criterion_mse = nn.MSELoss()  # 特征一致性损失（MSE）
lambda_consistency = 0.1  # 特征一致性损失的权重

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma,
                                      last_epoch=-1)

best_val_acc = 0.0  # 用于保存最优模型
best_val_auc = 0.0  # 用于保存最优AUC

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for img_1, img_2, label in train_loader:
        img_1 = img_1.to(device).float()
        img_2 = img_2.to(device).float()
        label = label.to(device).long()

        # 前向传播
        outputs, features_v, features_d = model(img_1, img_2)  # 假设模型返回分类结果和两个分支的特征表示

        # 计算交叉熵损失
        loss_ce = criterion_ce(outputs, label)

        # 计算特征一致性损失（MSE）
        # 计算余弦相似度损失
        loss_consistency = cosine_similarity_loss(features_v, features_d)

        # 总损失
        loss = loss_ce + lambda_consistency * loss_consistency

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算训练集损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    epoch_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # 验证集评估
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for img_1, img_2, label in val_loader:
            img_1 = img_1.to(device).float()
            img_2 = img_2.to(device).float()
            label = label.to(device).long()

            outputs, _, _ = model(img_1, img_2)  # 验证时不需要特征表示
            loss = criterion_ce(outputs, label)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            # 保存所有的预测概率和真实标签
            probs = F.softmax(outputs, dim=1)[:, 1]  # 获取正类的概率（假设是二分类）
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # 计算 AUC
    val_auc = roc_auc_score(all_labels, all_probs)

    scheduler.step()
    print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
        f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Val AUC: {val_auc:.4f}')
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), config.save_path + 'test.pth')
        print(f'Best model saved with AUC: {best_val_auc:.4f}')

print(f'Training Finished. Best Validation AUC: {best_val_auc:.4f}')
