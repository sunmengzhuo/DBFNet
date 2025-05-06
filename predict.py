import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from models import DualBranchResNet
from utils.DataSets import CustomDataset
import config
import numpy as np
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, roc_curve
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = DualBranchResNet1.DualBranchConvNext().to(device)
model.load_state_dict(torch.load(config.save_path + 'csmodel_tumor_fat_0.8889.pth', map_location=device))
model.eval()

# 存储 fc1 激活值
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


# 注册 fc1 层的钩子
model.fc1.register_forward_hook(get_activation('fc1'))

# 内部
test_folder1 = './data/complete/ex_test_v'
test_folder2 = './data/complete/ex_test_d'
test_folder3 = './data/fat/ex_test'
test_label_path = './data/医大二MVI动脉期终版.xlsx'  # 若无标签可忽略 AUC 计算

batch_size = 8
img_size = (224, 224)
threshold = 0.444  # 设定阈值（可以手动调整）

# 预处理
test_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

# 加载数据集
test_dataset = CustomDataset(folder1=test_folder1, folder2=test_folder2, folder3=test_folder3,
                             label_path=test_label_path,
                             transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 进行预测
all_probs = []
all_labels = []
all_preds = []

# 打开 CSV 文件保存特征数据
with open("Result/随便试试.csv", "w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')

    # 另存预测类别、真实类别和预测概率
    with open("Result/predictions_ex_test.csv", "w", newline='', encoding='utf-8') as pred_file:
        pred_writer = csv.writer(pred_file)
        pred_writer.writerow(["Sample_ID", "True_Label", "Pred_Label", "Pred_Probability"])  # 写入标题

        sample_id = 1  # 初始化样本ID

        for img_1, img_2, label in test_loader:
            img_1 = img_1.to(device).float()
            img_2 = img_2.to(device).float()
            label = label.to(device).long()

            outputs, _, _ = model(img_1, img_2)
            probs = F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()  # 获取正类的概率
            preds = (probs >= threshold).astype(int)  # 按照阈值分类

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())

            # 记录 fc1 特征
            fc1_features = activation['fc1'].cpu().numpy()
            for i in range(fc1_features.shape[0]):
                writer.writerow([int(label[i].cpu().numpy())] + fc1_features[i].tolist())

            # 记录预测类别、真实类别和预测概率
            for true_label, pred_label, pred_prob in zip(label.cpu().numpy(), preds, probs):
                pred_writer.writerow([sample_id, true_label, pred_label, pred_prob])
                sample_id += 1

# 计算 AUC（如果有真实标签）
if test_label_path:
    auc_score = roc_auc_score(all_labels, all_probs)
    print(f'Test AUC: {auc_score:.4f}')

# 计算 ACC、Sensitivity、Specificity
acc = accuracy_score(all_labels, all_preds)
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # 避免除零错误
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

# 输出评估结果
print(f'Threshold: {threshold}')
print(f'Test Accuracy: {acc:.4f}')
print(f'Test Sensitivity (Recall): {sensitivity:.4f}')
print(f'Test Specificity: {specificity:.4f}')

# predict_pro_np = np.array(all_probs)
# test_label_np = np.array(all_labels)
# np.save('./Result/visualized/dca/ex_Tumor_Fat_prepro', predict_pro_np)
# np.save('./Result/visualized/dca/ex_Tumor_Fat_label', test_label_np)

# 计算并绘制 ROC 曲线
fpr, tpr, _ = roc_curve(all_labels, all_probs)

# 保存 FPR 和 TPR 到 CSV 文件
# np.save("Result/in_radio_fpr.npy", fpr)
# np.save("Result/in_radio_tpr.npy", tpr)
# # 绘制 ROC 曲线
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.savefig("Result/roc_curve.png")
# plt.show()
from sklearn.utils import resample

def compute_confidence_interval(metric_fn, y_true, y_pred, n_bootstraps=5, ci=0.95):
    stats = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for _ in range(n_bootstraps):
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue  # 跳过没有正负样本的情况
        stat = metric_fn(y_true[indices], y_pred[indices])
        stats.append(stat)

    lower = np.percentile(stats, ((1 - ci) / 2) * 100)
    upper = np.percentile(stats, (1 - (1 - ci) / 2) * 100)
    return np.mean(stats), (lower, upper)


# AUC 置信区间
auc_mean, auc_ci = compute_confidence_interval(roc_auc_score, all_labels, all_probs)
print(f"AUC: {auc_mean:.4f}, 95% CI: ({auc_ci[0]:.4f}, {auc_ci[1]:.4f})")

# Accuracy 置信区间
acc_mean, acc_ci = compute_confidence_interval(accuracy_score, all_labels, all_preds)
print(f"Accuracy: {acc_mean:.4f}, 95% CI: ({acc_ci[0]:.4f}, {acc_ci[1]:.4f})")

# Sensitivity 置信区间
def sensitivity_fn(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) != 0 else 0

sens_mean, sens_ci = compute_confidence_interval(sensitivity_fn, all_labels, all_preds)
print(f"Sensitivity: {sens_mean:.4f}, 95% CI: ({sens_ci[0]:.4f}, {sens_ci[1]:.4f})")

# Specificity 置信区间
def specificity_fn(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) != 0 else 0

spec_mean, spec_ci = compute_confidence_interval(specificity_fn, all_labels, all_preds)
print(f"Specificity: {spec_mean:.4f}, 95% CI: ({spec_ci[0]:.4f}, {spec_ci[1]:.4f})")
