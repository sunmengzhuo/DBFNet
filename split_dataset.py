import pandas as pd
from sklearn.model_selection import train_test_split

def split_excel_by_label(input_excel, output_train, output_test, label_column='MVI', train_ratio=0.8, random_state=42):
    """
    根据标签列的值按比例划分 Excel 表为训练集和测试集
    :param input_excel: 输入的 Excel 文件路径
    :param output_train: 训练集保存路径
    :param output_test: 测试集保存路径
    :param label_column: 标签列名（默认为 'MVI'）
    :param train_ratio: 训练集比例（默认为 0.8）
    :param random_state: 随机种子（默认为 42）
    """
    # 读取 Excel 文件
    df = pd.read_excel(input_excel)

    # 按标签列的值划分数据集
    train_df, test_df = train_test_split(df, train_size=train_ratio, stratify=df[label_column], random_state=random_state)

    # 保存训练集和测试集
    train_df.to_excel(output_train, index=False)
    test_df.to_excel(output_test, index=False)

    print(f"数据集划分完成：训练集 {len(train_df)} 条，测试集 {len(test_df)} 条")

# 示例调用
input_excel = './data/MVI.xlsx'  # 输入的 Excel 文件路径
output_train = 'train.xlsx'  # 训练集保存路径
output_test = 'test.xlsx'  # 测试集保存路径
split_excel_by_label(input_excel, output_train, output_test, label_column='MVI', train_ratio=0.8)