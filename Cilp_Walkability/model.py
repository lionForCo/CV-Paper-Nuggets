import torch
import clip
from PIL import Image
import numpy as np
import os

# ==========================================
# 1. 环境配置与模型加载
# ==========================================

# 自动检测 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用设备: {device}")

# 加载 CLIP 模型 (ViT-B/32 是论文中使用的标准架构)
print("正在加载 CLIP 模型，请稍候...")
model, preprocess = clip.load("ViT-B/32", device=device)
print("模型加载完成！")

# ==========================================
# 2. 录入完整指标 (基于论文附录 A.1)
# ==========================================
# direction: 1 代表正向指标 (Positive), -1 代表负向指标 (Negative)

indicators = [
    # --- Obstacles / Sidewalk Condition ---
    {"id": 1,  "prompt": "There are vehicles parked on the sidewalk", "direction": -1},
    {"id": 2,  "prompt": "There are scooters parked on the sidewalk", "direction": -1},
    {"id": 3,  "prompt": "There are bicycles, and motorcycles parked on the sidewalk", "direction": -1},
    {"id": 4,  "prompt": "There is a wide sidewalk", "direction": 1},
    {"id": 5,  "prompt": "There is a narrow sidewalk", "direction": -1},
    {"id": 6,  "prompt": "There is a fenced sidewalk", "direction": 1},
    {"id": 7,  "prompt": "There is a heightened sidewalk", "direction": 1},
    {"id": 8,  "prompt": "There is a tile pavement", "direction": 1},
    {"id": 9,  "prompt": "There are cracks, depressions, and flooded sidewalks.", "direction": -1},
    
    # --- Traffic Safety ---
    {"id": 10, "prompt": "There is road name/direction signs", "direction": 1},
    {"id": 11, "prompt": "There is a pedestrian symbol marked on the pavement", "direction": 1},
    {"id": 12, "prompt": "There are green belts and fences between sidewalks and vehicle lane", "direction": 1},
    {"id": 13, "prompt": "There is crosswalk", "direction": 1},
    {"id": 14, "prompt": "There is a pavement traffic light", "direction": 1},
    {"id": 15, "prompt": "There are scaffolding or construction sites on the sidewalk", "direction": -1},
    {"id": 16, "prompt": "There are many vehicles on the road", "direction": -1},
    {"id": 17, "prompt": "There are people on the pavement.", "direction": 1},
    {"id": 18, "prompt": "There are railways along the road", "direction": -1},
    {"id": 19, "prompt": "There are streetlights on the pavements", "direction": 1},
    
    # --- Security ---
    {"id": 20, "prompt": "There are police officers along the road", "direction": 1},
    {"id": 21, "prompt": "There are security cameras", "direction": 1},
    
    # --- Comfort ---
    {"id": 22, "prompt": "There are skyscrapers", "direction": -1},
    {"id": 23, "prompt": "There are various heights of buildings", "direction": 1},
    {"id": 24, "prompt": "There is graffiti on the walls", "direction": -1},
    {"id": 25, "prompt": "There are billboards or advertising signs on the pavement", "direction": -1},
    {"id": 26, "prompt": "There are garbages on the road", "direction": -1},
    {"id": 27, "prompt": "There are trees", "direction": 1},
    {"id": 28, "prompt": "There are shadow areas", "direction": 1},
    {"id": 29, "prompt": "There are benches along the sidewalks", "direction": 1},
    {"id": 30, "prompt": "There are unleashed dogs on the sidewalk", "direction": -1},
    {"id": 31, "prompt": "There are birds", "direction": 1},
    
    # --- Attractiveness ---
    {"id": 32, "prompt": "There are shops along the road", "direction": 1},
    {"id": 33, "prompt": "There are cafes along the road", "direction": 1},
    {"id": 34, "prompt": "There are public buildings such as hospitals, schools, libraries, and office complexes", "direction": 1},
    {"id": 35, "prompt": "There are open green zone beside the road", "direction": 1},
    {"id": 36, "prompt": "There are playground beside the road", "direction": 1},
    {"id": 37, "prompt": "There are bus or underground stations", "direction": 1},
    {"id": 38, "prompt": "There are landmarks", "direction": 1},
    {"id": 39, "prompt": "There is river", "direction": 1},
    {"id": 40, "prompt": "There are flowers", "direction": 1}
]

# ==========================================
# [cite_start]3. 核心计算函数 (论文公式复现) [cite: 129-130]
# ==========================================

def calculate_walkability(image_path, indicators_list):
    """
    输入: 图片路径, 指标列表
    输出: 感知步行友好度总分, 详细概率分布
    """
    
    # --- A. 数据准备 ---
    # 提取所有 Prompts 和 Directions
    text_prompts = [item["prompt"] for item in indicators_list]
    directions = np.array([item["direction"] for item in indicators_list])
    
    # 预处理图片并搬运到 GPU
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    except Exception as e:
        print(f"错误: 无法读取图片 {image_path}. {e}")
        return None, None

    # Tokenize 文本并搬运到 GPU
    text = clip.tokenize(text_prompts).to(device)

    # --- B. 模型推理 (Zero-shot Prediction) ---
    with torch.no_grad():
        # 计算图文相似度 (Logits)
        logits_per_image, logits_per_text = model(image, text)
        
        # 使用 Softmax 将相似度转换为概率分布 p(x)
        # props 的形状是 (1, 40), 我们取 [0] 变成 (40,)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # --- C. 计算得分 (论文公式 1 & 2) ---
    # 公式: Score = Sum(d_i * p(x_i)) + H(positive) - H(negative)
    
    # 1. 第一部分: 加权概率和
    weighted_sum = np.sum(directions * probs)
    
    # 2. 第二部分: 熵值调整 (Entropy Adjustment)
    # H(X) = - sum( p(x) * log2(p(x)) )
    # 需要添加一个极小值 epsilon 防止 log2(0) 报错
    epsilon = 1e-10
    
    # 筛选正向指标的概率
    pos_indices = [i for i, d in enumerate(directions) if d == 1]
    p_pos = probs[pos_indices]
    
    # 筛选负向指标的概率
    neg_indices = [i for i, d in enumerate(directions) if d == -1]
    p_neg = probs[neg_indices]
    
    # 计算正向熵 H_positive
    h_positive = -np.sum(p_pos * np.log2(p_pos + epsilon))
    
    # 计算负向熵 H_negative
    h_negative = -np.sum(p_neg * np.log2(p_neg + epsilon))
    
    # 3. 最终得分
    final_score = weighted_sum + h_positive - h_negative
    
    return final_score, probs

# ==========================================
# 4. 运行示例
# ==========================================

if __name__ == "__main__":
    # 请将此处替换为你本地的一张真实图片路径
    # 建议使用一张清晰的街道照片
    test_image_path = "test_street.jpg" 
    
    # 创建一个假的图片文件用于测试代码逻辑 (如果你没有现成图片)
    if not os.path.exists(test_image_path):
        print(f"提示: 未找到 {test_image_path}, 正在生成一张空白测试图...")
        Image.new('RGB', (224, 224), color='gray').save(test_image_path)

    print(f"正在分析图片: {test_image_path} ...")
    
    score, distribution = calculate_walkability(test_image_path, indicators)
    
    if score is not None:
        print("-" * 50)
        print(f"【最终感知步行友好度得分】: {score:.4f}")
        print("-" * 50)
        
        # 打印匹配度最高的前 5 个特征
        print("AI 认为最符合该图片的特征 (Top 5):")
        # 将概率和指标打包排序
        indexed_probs = list(enumerate(distribution))
        sorted_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)
        
        for idx, prob in sorted_probs[:5]:
            item = indicators[idx]
            dir_str = "(+ 正向)" if item["direction"] == 1 else "(- 负向)"
            print(f"  {prob*100:.2f}% : {item['prompt']} {dir_str}")