from PIL import Image
import numpy as np
import os

# 输入文件夹
image_folder = "D:\\肝癌\\图像\\二院图像\\脂肪\\原图"  # 原图文件夹
mask_folder = "D:\\肝癌\\图像\\二院图像\\脂肪\\mask"  # mask 文件夹
output_folder = "D:\\肝癌\\图像\\二院图像\\脂肪\\ROI"  # ROI 输出文件夹

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取所有原图文件名
image_files = [f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

for filename in image_files:
    img_path = os.path.join(image_folder, filename)
    mask_path = os.path.join(mask_folder, filename)  # 假设 mask 文件名与原图相同

    if not os.path.exists(mask_path):
        print(f"未找到 {filename} 的 mask，跳过...")
        continue

    # 读取原图和 mask
    img = Image.open(img_path).convert("RGB")  # 确保原图是 RGB
    mask = Image.open(mask_path).convert("L")  # 转换为灰度

    # 转换为 NumPy 数组
    img_np = np.array(img)
    mask_np = np.array(mask) / 255  # 归一化到 [0,1]

    # 进行逐通道相乘
    roi_np = (img_np * mask_np[:, :, None]).astype(np.uint8)

    # 转换回 PIL 图像
    roi_img = Image.fromarray(roi_np)

    # 保存提取的 ROI
    roi_img.save(os.path.join(output_folder, filename))

print("ROI 提取完成，结果保存在:", output_folder)
