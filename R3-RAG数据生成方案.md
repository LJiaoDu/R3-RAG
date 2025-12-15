# R3-RAG数据生成方案（正确版）

## 🎯 数据用途划分

### ✅ 正确理解

```
15000个问题分为两部分：

1. 10000条 → 生成推理链 → SFT训练（冷启动阶段）
2. 5000条 → 生成参考答案 → GRPO训练 + 测试集
```

**不是简单/复杂问题的区分，而是训练阶段的需求！**

---

## 📚 详细说明

### 1️⃣ SFT训练数据（10000条）

**格式**：Question + Reasoning Chain + Answer

**作用**：
- 冷启动训练
- 教会模型如何推理
- 建立基础能力

**示例**：

```
Question: "I want to take Machine Learning. What are the prerequisites?"

Answer:
Thought: I need to find the official prerequisites for Machine Learning at TUM.
Action: Search[Machine Learning IN2064 prerequisites]
Observation: Machine Learning (IN2064) requires: Linear Algebra, Probability Theory, Python.

Thought: Let me check if these have prerequisites.
Action: Search[Linear Algebra prerequisites]
Observation: Linear Algebra (MA1001) has no prerequisites.

Thought: Check Probability Theory.
Action: Search[Probability Theory prerequisites]
Observation: Probability Theory requires Linear Algebra.

Thought: I have complete information.
Action: Finish[To take Machine Learning, you need:
1. Linear Algebra (MA1001) - 1st semester
2. Probability Theory (MA2009) - 2nd semester
3. Python programming
Total: 2-3 semesters preparation]
```

**特点**：
- ✅ 完整的推理过程
- ✅ Thought/Action/Observation格式
- ✅ 展示检索和推理步骤

---

### 2️⃣ GRPO训练数据（4000条）

**格式**：Question + Reference Answer（无推理链）

**作用**：
- GRPO强化学习训练
- 模型自己生成推理过程
- 和参考答案对比计算reward

**示例**：

```
Question: "I want to take Machine Learning. What are the prerequisites?"

Reference Answer:
To take Machine Learning (IN2064) at TUM, you need:

1. Linear Algebra (MA1001) - 8 ECTS
   - No prerequisites
   - 1st semester

2. Probability Theory (MA2009) - 6 ECTS
   - Requires Linear Algebra
   - 2nd semester

3. Python programming experience

Study path: Linear Algebra → Probability Theory → Machine Learning
Total preparation: 2-3 semesters
```

**特点**：
- ✅ 只有最终答案
- ✅ 无推理步骤
- ✅ 清晰完整的信息

**GRPO训练流程**：
```
1. 给模型问题
2. 模型自己生成推理链 + 答案
3. 用参考答案计算reward
4. 优化policy
```

---

### 3️⃣ 测试数据（1000条）

**格式**：Question + Reference Answer

**作用**：
- 评估模型性能
- 不参与训练
- 计算准确率、F1等指标

---

## 💰 成本估算

### 使用 GPT-4o-mini

```
SFT数据（10000条，推理链）:
  输入: 10000 × 200 tokens = 2,000,000 tokens → $0.30
  输出: 10000 × 600 tokens = 6,000,000 tokens → $3.60
  小计: $3.90

GRPO数据（5000条，参考答案）:
  输入: 5000 × 150 tokens = 750,000 tokens → $0.11
  输出: 5000 × 300 tokens = 1,500,000 tokens → $0.90
  小计: $1.01

总成本: $4.91
```

**时间**：5-6小时（异步模式）

---

## 🚀 使用方法

### Step 1: 准备问题文件

```jsonl
{"id": 1, "question": "What courses do I need before ML?"}
{"id": 2, "question": "How many ECTS is Computer Vision?"}
...
```

保存为 `questions.jsonl`

### Step 2: 运行脚本

```bash
# 设置API Key
export OPENAI_API_KEY='your-key'

# 运行
python generate_r3rag_data.py
```

### Step 3: 选择分配策略

```
选择数据分配策略:
1. 随机分配（推荐）- 避免数据偏差
2. 按顺序分配 - 前10000个SFT，后5000个GRPO
3. 平衡分配 - 保持每类问题比例一致

请选择 (1/2/3): 1
```

**推荐选1（随机）**，避免某类问题集中在某个数据集。

### Step 4: 查看结果

生成3个文件：

```
sft_training_data.jsonl      # 10000条，带推理链
grpo_train.jsonl             # 4000条，参考答案（训练）
grpo_test.jsonl              # 1000条，参考答案（测试）
```

---

## 📊 数据格式对比

### SFT训练数据格式

```json
{
  "id": 1,
  "question": "What courses do I need before ML?",
  "answer": "Thought: ...\nAction: Search[...]\nObservation: ...\nAction: Finish[...]",
  "data_split": "sft",
  "model": "gpt-4o-mini"
}
```

### GRPO训练数据格式

```json
{
  "id": 2,
  "question": "How many ECTS is Computer Vision?",
  "answer": "Computer Vision (IN2128) is worth 6 ECTS credits.",
  "data_split": "grpo",
  "model": "gpt-4o-mini"
}
```

**关键区别**：
- SFT：`answer`字段包含完整推理链
- GRPO：`answer`字段只有最终答案

---

## 🔄 转换为训练格式

### 1. SFT训练数据 → LLaMA-Factory

```python
# convert_sft_data.py
import json

# 读取SFT数据
with open('sft_training_data.jsonl', 'r') as f:
    sft_data = [json.loads(line) for line in f]

# 转换为Alpaca格式
alpaca_data = []
for item in sft_data:
    if not item.get('answer'):
        continue

    alpaca_data.append({
        "instruction": item['question'],
        "input": "",
        "output": item['answer']
    })

# 保存
with open('tum_sft_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

print(f"✅ 转换了 {len(alpaca_data)} 条SFT训练数据")
```

### 2. 配置dataset_info.json

```json
{
  "tum_sft": {
    "file_name": "tum_sft_dataset.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

### 3. SFT训练配置

```yaml
# llama3_full_sft_tum.yaml
model_name_or_path: "Qwen/Qwen2.5-7B"
stage: sft
finetuning_type: full
deepspeed: deepspeed/ds_z3_config.json

dataset: tum_sft
per_device_train_batch_size: 4
learning_rate: 7.0e-6
num_train_epochs: 3

output_dir: ./output/tum_sft
```

### 4. GRPO训练数据准备

```python
# GRPO训练时，模型会自己生成推理链
# grpo_train.jsonl提供参考答案用于计算reward

# OpenRLHF格式
grpo_data = []
with open('grpo_train.jsonl', 'r') as f:
    for line in f:
        item = json.loads(line)
        if item.get('answer'):
            grpo_data.append({
                "prompt": item['question'],
                "reference": item['answer']  # 参考答案
            })

with open('tum_grpo_dataset.json', 'w') as f:
    json.dump(grpo_data, f, ensure_ascii=False, indent=2)
```

---

## 📈 完整训练流程

### Stage 1: SFT训练（冷启动）

```bash
# 使用10000条推理链数据
llamafactory-cli train llama3_full_sft_tum.yaml
```

**效果**：模型学会了基础推理能力

### Stage 2: GRPO训练（强化学习）

```bash
# 使用4000条参考答案
# 模型自己生成推理，和参考答案对比
cd OpenRLHF
python train_grpo.py --dataset tum_grpo_dataset.json
```

**GRPO过程**：
1. 模型收到问题
2. 模型生成推理链 + 答案
3. 和参考答案对比（EM、F1、ROUGE）
4. 计算reward
5. 更新policy

### Stage 3: 测试评估

```bash
# 使用1000条测试数据
python evaluate.py \
  --model ./output/tum_grpo \
  --test-data grpo_test.jsonl
```

---

## 🎯 为什么这样设计？

### 1️⃣ SFT需要推理链

**原因**：
- 冷启动阶段，模型不会推理
- 需要示例教会模型"如何思考"
- 展示Thought → Action → Observation循环

**类比**：
就像教小孩解数学题，要展示完整的解题步骤。

### 2️⃣ GRPO不需要推理链

**原因**：
- 模型已经学会推理（从SFT）
- GRPO阶段让模型自己推理
- 只需参考答案来判断对错

**类比**：
学生已经学会解题方法，现在让他自己做题，老师只提供答案批改。

### 3️⃣ 为什么不全部用推理链？

**回答**：
- GRPO的目标是**优化推理过程**
- 如果给了推理链，就变成了监督学习
- 只给答案，模型会探索不同推理路径
- 通过reward signal找到最优路径

---

## 📊 对比总结

| 阶段 | 数据量 | 格式 | 用途 | 是否包含推理链 |
|------|--------|------|------|----------------|
| **SFT** | 10000 | Q + 推理链 + A | 冷启动训练 | ✅ 完整推理链 |
| **GRPO训练** | 4000 | Q + 参考答案 | 强化学习 | ❌ 只有答案 |
| **测试** | 1000 | Q + 参考答案 | 评估性能 | ❌ 只有答案 |

---

## 🎤 面试时怎么说？

**面试官：你如何生成15000条数据？**

**你的回答：**

> "我将15000个问题分为两部分：
>
> **第一部分（10000条）**用于SFT训练，生成完整的推理链。这些数据教会模型如何一步步推理，包含Thought、Action、Observation的完整流程。这是冷启动阶段，让模型建立基础推理能力。
>
> **第二部分（5000条）**用于GRPO训练和测试，只生成参考答案，不包含推理链。在GRPO阶段，模型会自己生成推理过程，然后和参考答案对比来计算reward。这样模型可以探索不同的推理路径，找到最优解。
>
> 我进一步将5000条分为4000训练+1000测试，确保有独立的测试集评估性能。
>
> 这个设计符合R3-RAG两阶段训练的思路：SFT建立基础，GRPO优化策略。总成本约$5，用GPT-4o-mini生成，异步处理5-6小时完成。"

**专业、清晰、有理有据！** 🎯

---

## ✅ 检查清单

在开始之前，确保：

- [ ] 有15000个问题（JSONL格式）
- [ ] 设置了OPENAI_API_KEY
- [ ] 安装了依赖：`pip install openai tqdm aiohttp`
- [ ] 有$5预算（GPT-4o-mini）
- [ ] 有5-6小时时间（可以挂机运行）

在生成之后，检查：

- [ ] `sft_training_data.jsonl` 有~10000条，包含推理链
- [ ] `grpo_train.jsonl` 有~4000条，只有答案
- [ ] `grpo_test.jsonl` 有~1000条，只有答案
- [ ] 抽查10条SFT数据，确认格式正确
- [ ] 抽查10条GRPO数据，确认无推理链

---

## 🚀 开始生成！

```bash
python generate_r3rag_data.py
```

**预祝生成顺利！** 🎉
