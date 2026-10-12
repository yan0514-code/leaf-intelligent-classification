import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 多线程冲突解决方案 
from flask import Flask, request, jsonify, render_template, Response
import numpy as np
from model import get_mobilenetv2_model
import os
import torch
from PIL import Image
import io
import csv
from io import StringIO

app = Flask(__name__)

# 全局单例模型变量
model = None
DEBUG = False

# ===================== 叶子标签映射【和文件夹0/1/2严格对应】 =====================
# 文件夹0 → 黄栌
# 文件夹1 → 小叶黄杨
# 文件夹2 → 紫叶李
leaf_labels = [
    "黄栌",
    "小叶黄杨",
    "紫叶李"
]
# MobileNetV2 标准化参数（ImageNet）
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_model():
    """加载训练好的MobileNetV2叶片分类模型"""
    global model
    print("📥 加载MobileNetV2叶子分类模型...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'best_mobilenetv2_leaf.pth')

    if DEBUG:
        print("模型路径：", model_path)

    print("🚀 加载模型权重...")
    state_dict = torch.load(model_path, map_location='cpu')
    model = get_mobilenetv2_model(num_classes=3)
    model.load_state_dict(state_dict)
    model.eval()

    if DEBUG:
        print(model)

    print("✅ MobileNetV2模型加载完成！")
    return model


@app.route('/')
def index():
    """返回图片上传识别前端页面"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """单张图片上传识别接口"""
    print('📝 开始叶子图像识别...')
    try:
        # 接收上传的图片文件
        file = request.files.get("leaf_img")
        if not file:
            return jsonify({'error': '未上传图片文件'}), 400
        
        # 读取RGB三通道图，resize至224×224（MobileNetV2标准输入）
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        # 标准化处理，和训练时保持一致
        img_array = (img_array - MEAN) / STD
        # 维度转换：HWC → CHW，构建张量 [1,3,224,224]
        img_array = np.transpose(img_array, (2, 0, 1))
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        # 推理不计算梯度，加速
        with torch.no_grad():
            out = model(img_tensor)
            probs = out.softmax(dim=1)

        # 取置信度前三的结果
        topk_probs, topk_indices = torch.topk(probs, 3, dim=1, largest=True, sorted=True)
        top3_probs = topk_probs[0].tolist()
        top3_preds = topk_indices[0].tolist()

        # 数字标签转中文叶子名
        first_result_name = leaf_labels[top3_preds[0]]
        print(f"✅ 识别结果: {first_result_name}, 置信度: {top3_probs[0]:.2%}")

        # 组装返回给前端的top3结果
        res_top3 = []
        for idx, prob in zip(top3_preds, top3_probs):
            res_top3.append({
                "leaf_name": leaf_labels[idx],
                "confidence": prob
            })

        return jsonify({
            'prediction': first_result_name,
            'confidence': top3_probs[0],
            'top3': res_top3
        })

    except Exception as e:
        print(f"❌ 预测出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ===================== 批量识别接口 =====================
@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """批量叶片图片识别，返回全部图片预测结果与CSV导出文本"""
    try:
        # 获取多张上传图片
        file_list = request.files.getlist("leaf_batch_img")
        if len(file_list) == 0:
            return jsonify({"error": "未上传任何批量图片文件"}), 400

        batch_result = []
        for file in file_list:
            filename = file.filename
            # 图像预处理逻辑和单图完全统一
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = (img_array - MEAN) / STD
            img_array = np.transpose(img_array, (2, 0, 1))
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)

            # GPU推理
            with torch.no_grad():
                out = model(img_tensor)
                probs = out.softmax(dim=1)
            topk_probs, topk_indices = torch.topk(probs, 3, dim=1)
            top3_probs = topk_probs[0].tolist()
            top3_preds = topk_indices[0].tolist()
            first_name = leaf_labels[top3_preds[0]]
            first_conf = top3_probs[0]

            # 单张图片结果封装
            single_res = {
                "file_name": filename,
                "pred_leaf": first_name,
                "confidence": round(first_conf, 4),
                "top3": [
                    {"name": leaf_labels[idx], "conf": round(prob,4)}
                    for idx, prob in zip(top3_preds, top3_probs)
                ]
            }
            batch_result.append(single_res)

        # 生成CSV文本流
        output = StringIO()
        csv_writer = csv.writer(output)
        csv_writer.writerow(["图片文件名", "预测叶片种类", "置信度", "Top2类别", "Top2置信度", "Top3类别", "Top3置信度"])
        for item in batch_result:
            row = [
                item["file_name"],
                item["pred_leaf"],
                item["confidence"],
                item["top3"][1]["name"],
                item["top3"][1]["conf"],
                item["top3"][2]["name"],
                item["top3"][2]["conf"]
            ]
            csv_writer.writerow(row)
        csv_content = output.getvalue()

        return jsonify({
            "batch_data": batch_result,
            "csv_text": csv_content
        })

    except Exception as e:
        print(f"❌ 批量识别失败：{str(e)}")
        return jsonify({"error": str(e)}), 500


# ===================== CSV文件下载接口 =====================
@app.route('/download_csv', methods=['POST'])
def download_csv():
    csv_text = request.form.get("csv_text")
    if not csv_text:
        return "无导出数据", 400
    # 返回可下载csv文件流
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="叶片批量识别结果.csv"'}
    )


@app.route('/health', methods=['GET'])
def health():
    """服务器健康检测接口"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    print("=" * 50)
    print("🌿 MobileNetV2三叶分类识别系统 v1.0（支持批量识别）")
    print("=" * 50)

    try:
        model = load_model()
        print("🌐 Web服务启动成功（HTTPS局域网）")
        print("👉 手机访问地址: https://172.16.107.23:5001")
        print("=" * 50)
        # ssl_context='adhoc 自动生成临时证书，开启https
        # host=0.0.0.0 允许手机局域网连接，ssl开启加密支持摄像头
        app.run(host="0.0.0.0", port=5001, debug=True, ssl_context="adhoc")
    except Exception as e:
        print(f"❌ 服务启动失败: {e}")
        input("按回车键退出程序...")