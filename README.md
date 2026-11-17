# leaf-intelligent-classification
基于深度学习的叶子图像智能分类系统课程设计
# 技术栈
Python、Pytorch、Jupyter Notebook、OpenCv、Matplotlib
# 功能
- 叶片图像加载、归一化、数据增强
- 基于轻量化迁移学习（mobilenetv2）的分类模型训练
- 模型精度、损失曲线可视化
- 单张图片预测，输出分类结果
- 网页端与移动端皆可预测输出
# 运行说明
1. 配置 Python 环境，安装依赖包
2. 准备叶片数据集，划分训练集、验证集（本实验数据集为作者本人线下采集真实叶子样本）
3. 运行 notebook 脚本训练模型
4. 加载训练好的权重，完成叶片图像分类预测
