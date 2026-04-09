# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    # Step 1: 检查整体结构是否包含 <think> 和 <answer> 标签且顺序正确
    structure_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    structure_match = re.fullmatch(structure_pattern, response.strip(), re.DOTALL)
    structure_score = 0.5 if structure_match else 0.0

    # Step 2: 提取 <answer> 标签内容并检查是否为纯数字
    content_score = 0.0
    if structure_match:
        content_match = re.search(r"<answer>(.*?)</answer>", response.strip(), re.IGNORECASE | re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
            if content.isdigit():
                content_score = 0.5

    # Step 3: 总分为结构分 + 内容分
    return structure_score + content_score

def simple_distance_reward(response: str, solution: str, arrow_to_goal_distances: Dict[str, float]) -> float:

    if not isinstance(arrow_to_goal_distances, dict):
         raise ValueError("arrow_points must be a dict")
    
    reward = 0.0  # Default reward if parsing fails or point is invalid
    try:
        # 1. Parse completion to find the chosen arrow number (N)
        # Adjust regex if the format is different
        # match = re.search(r'<answer>Arrow\s*(\d+)</answer>', content, re.IGNORECASE)
        match = re.search(r'<answer>(\d+)</answer>', response, re.IGNORECASE)
        answer = re.search(r'<answer>(\d+)</answer>', solution, re.IGNORECASE)
        if match and answer:
            arrow_number = int(match.group(1))
            answer_number = int(answer.group(1))
            if arrow_number == 0:
                arrow_number = len(arrow_to_goal_distances)
            
            if arrow_number == answer_number:
                reward = 1.0
            else:
                reward = 0.0
        else:
            print(f"Warning: Could not parse arrow number from completion: {response}")

    except Exception as e:
        print(f"Error calculating distance reward for completion '{response}': {e}")
                
    return reward

def min_max_reward(response: str, solution: str, arrow_to_goal_distances: Dict[str, float]) -> float:

    if not isinstance(arrow_to_goal_distances, dict):
         raise ValueError("arrow_points must be a dict")
    
    reward = 0.0  # Default reward if parsing fails or point is invalid
    try:
        # 1. Parse completion to find the chosen arrow number (N)
        # Adjust regex if the format is different
        # match = re.search(r'<answer>Arrow\s*(\d+)</answer>', content, re.IGNORECASE)
        match = re.search(r'<answer>(\d+)</answer>', response, re.IGNORECASE)
        if match:
            arrow_number = int(match.group(1))
            
            if arrow_number == 0:
                arrow_number = len(arrow_to_goal_distances)
            
            selected_key = str(arrow_number)

            if selected_key in arrow_to_goal_distances:
                selected_distance = arrow_to_goal_distances[selected_key]

                # 归一化所有距离
                all_distances = list(arrow_to_goal_distances.values())
                min_distance = min(all_distances)
                max_distance = max(all_distances)

                if min_distance != max_distance:
                    reward = (max_distance - selected_distance) / (max_distance - min_distance)
                else:
                    reward = 1
            else:
                print(f"Warning: Arrow number {arrow_number} not found in distances dictionary.")
        else:
            print(f"Warning: Could not parse arrow number from completion: {response}")

    except Exception as e:
        print(f"Error calculating distance reward for completion '{response}': {e}")
                
    return reward
def hybrid_reward(response: str, solution: str, arrow_to_goal_distances: Dict[str, float]) -> float:
    """
    Hybrid reward without NumPy:
    - base: rank 或 softmax(-distance/T)
    - bonus: 命中最优(最小距离)时，按 (次优-最优)/|最优| 的“确定度”给动态加成
    - 返回值 ∈ [0,1]
    """
    import re, math

    # ---- 可调超参（不改函数签名） ----
    MODE = "softmax"     # "rank" 或 "softmax"
    T = 0.2           # softmax 温度（仅在 MODE="softmax" 时生效），越小越“尖锐”
    BONUS_MAX = 1.0  # exact match 的最大额外加成（确定度=1时）
    EPS = 1e-6

    if not isinstance(arrow_to_goal_distances, dict):
        raise ValueError("arrow_to_goal_distances must be a dict")

    # 解析 <answer>N</answer>
    match = re.search(r'<answer>(\d+)</answer>', response or "", re.IGNORECASE)
    if not match:
        return 0.0
    arrow_number = int(match.group(1))
    if arrow_number == 0:
        arrow_number = len(arrow_to_goal_distances)

    # 按数字顺序稳定排序 keys，保证索引一致
    try:
        keys = sorted(arrow_to_goal_distances.keys(), key=lambda x: int(x))
    except Exception:
        keys = sorted(arrow_to_goal_distances.keys())
    selected_key = str(arrow_number)
    if selected_key not in keys:
        return 0.0

    # 距离列表（越小越好）
    d = [float(arrow_to_goal_distances[k]) for k in keys]
    idx = keys.index(selected_key)
    n = len(d)

    # 退化处理
    if n <= 1:
        return 1.0
    dmin, dmax = min(d), max(d)
    if abs(dmax - dmin) < EPS:
        # 所有距离几乎一样：不可区分，给均匀分，不加成
        return 1.0 / n

    # base：rank 或 softmax 概率
    if MODE.lower() == "rank":
        order = sorted(range(n), key=lambda i: d[i])  # 升序，最小在前
        rank = [0] * n
        for r, i in enumerate(order):
            rank[i] = r  # 0=最好, n-1=最差
        base = 1.0 - rank[idx] / (n - 1 + EPS)  # ∈[0,1]
    else:
        # softmax(-d/T)（数值稳定：先减最大值）
        logits = [ -x / max(T, EPS) for x in d ]
        m = max(logits)
        exps = [ math.exp(x - m) for x in logits ]
        s = sum(exps)
        base = exps[idx] / s if s > 0 else 0.0

    # 动态标签加成：gap 越大越确定，加成越多
    order = sorted(range(n), key=lambda i: d[i])
    best = d[order[0]]
    second = d[order[1]] if n > 1 else best + EPS
    gap = (second - best) / (abs(best) + EPS)    # 归一化间隔
    certainty = max(0.0, min(1.0, gap))          # clip 到 [0,1]
    is_exact = (idx == order[0])
    bonus = (BONUS_MAX * certainty) if is_exact else 0.0

    reward = base + bonus
    if reward < 0.0: reward = 0.0
    if reward > 1.0: reward = 1.0
    return float(reward)

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        distance_score = hybrid_reward(response, reward_input["ground_truth"], reward_input["action_to_goal_dis"])
        scores.append(
            {
                "overall": (1 - format_weight) * distance_score + format_weight * format_score,
                "format": format_score,
                "accuracy": distance_score,
            }
        )

    return scores
