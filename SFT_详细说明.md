# R3-RAG项目SFT详细说明

## 一、SFT概述

R3-RAG项目的**SFT（Supervised Fine-Tuning）阶段**被称为**Cold Start（冷启动）**训练，是整个训练流程的第一阶段。这个阶段的目标是让基础模型学会基本的**推理-检索步骤分解**能力，为后续的强化学习训练打下基础。

## 二、模型选择

### 2.1 基础模型

项目支持两个基础模型系列：

1. **Llama-3.1-8B**
   - 配置文件：`train/LLaMA-Factory/examples/llama3_full_sft_llama.yaml`
   - 模板类型：`llama3`
   - 输出模型：**R3-RAG-CS-Llama**（Cold Start）

2. **Qwen2.5-7B**
   - 配置文件：`train/LLaMA-Factory/examples/llama3_full_sft_qwen.yaml`
   - 模板类型：`qwen`
   - 输出模型：**R3-RAG-CS-Qwen**（Cold Start）

### 2.2 训练框架

- **训练框架**：LLaMA-Factory
- **微调类型**：Full Parameter Fine-tuning（全参数微调）
- **分布式训练**：DeepSpeed ZeRO Stage 3

## 三、训练数据生产流程

### 3.1 数据源

训练数据来自三个多跳问答数据集：

1. **HotpotQA** - 训练集抽取3万样本
2. **2WikiMultiHopQA** - 完整训练集
3. **MuSiQue** - 完整训练集

### 3.2 Cold Start数据生成流程

整个数据生成过程分为**4个主要步骤**：

#### **步骤1：数据集拆分** (`data/construct/01split_dataset/`)

```bash
# 将原始数据集拆分成小块，便于并行处理
python split.py \
    --chunk_size 100 \
    --input_file <原始数据集> \
    --output_dir <输出目录>
```

**目的**：将大规模数据集拆分成多个小文件，每个文件包含100条数据，便于后续多进程并行生成。

#### **步骤2：生成搜索链** (`data/construct/02sample_from_dataset/`)

这是数据生成的**核心步骤**，使用外部API模型生成高质量的推理-检索轨迹。

**核心脚本**：`main.py` 或 `mainv1.py`

**关键参数**：
```python
num_passages_one_retrieval = 3      # 每次检索返回3个文档
num_attempts_one_question = 7       # 每个问题最多尝试7次
num_search_one_attempt = 7          # 每次尝试最多7步检索链
num_attempts_one_generation = 2     # 格式错误时重试2次
```

**数据生成策略**：

1. **使用外部API模型**（如DeepSeek）作为**Teacher模型**
   ```python
   --model_api_key "deepseek_api_key"
   --model_api_url "https://api.deepseek.com/beta"
   --model_api_name "deepseek-chat"
   ```

2. **检索服务配置**：
   ```python
   --retriever_host "10.176.52.122"
   --retriever_port "8002"
   ```

3. **迭代式推理-检索过程**：
   - **第一步**：使用 `prompt_question_initv2()` 生成初始分析和查询
   - **后续步骤**：使用 `prompt_question_newv2()` 基于之前步骤继续
   - **每一步输出格式**：
     ```
     The problem analysis: [分析当前问题状态]
     The retrieval query: [生成检索查询] 或
     The final answer: [最终答案]
     ```

4. **检索增强**：
   - 每次生成query后，调用检索服务获取相关文档
   - 检索结果会添加到下一步的context中
   - 如果第一次尝试失败，后续尝试会使用**query decomposition**（查询分解）

5. **答案验证**：
   - 使用LLM判断生成的答案是否与标准答案一致
   - 函数：`check_correctness(question, answer, golden_answers)`
   - 只有答案正确的搜索链才会被保存到 `search_chain_success`

6. **温度控制**：
   - 第一次尝试：`temperature=0.0`（确定性）
   - 后续尝试：`temperature=0.9`（增加多样性）

#### **步骤3：合并样本** (`data/construct/03merge_samples/`)

```bash
# 合并所有成功的样本
cat sample_dir/*.jsonl > merge.jsonl

# 统计成功率
python statistic.py merge.jsonl statistics.txt
```

**输出统计信息**：
- Total data rows: 总样本数
- Useful data rows: 成功生成的样本数
- Utilization rate: 成功率

#### **步骤4：格式转换** (`data/construct/format_single_query/`)

将生成的搜索链转换为LLaMA-Factory可用的格式：

```python
# 转换为标准格式
python format.py <input_file> <output_file>
```

**数据格式转换为**：
- **单轮对话格式**：用于标准SFT
- **多轮对话格式**：用于多轮SFT（ShareGPT格式）

### 3.3 Prompt设计核心

#### **初始Prompt** (`prompt_question_initv2`)

特点：
- 教会模型识别**并行分解**和**顺序分解**两种问题结构
- 提供Few-shot示例教学
- 强调只生成**第一步**的分析和查询

示例：
```
输入：地球卫星的表面温度范围与金星的表面温度范围是否重叠？
输出：
The problem analysis: 这是一个复合结构，需要并行分解...
The retrieval query: 地球卫星的表面温度范围是多少？
```

#### **后续步骤Prompt** (`prompt_question_newv2`)

特点：
- 评估上一步检索结果的相关性
- 如果检索失败，利用**参数知识**补充
- 决定是继续检索还是给出最终答案
- 支持**问题分解**策略

关键能力：
1. **决策能力**：判断信息是否充足
2. **分解能力**：将复杂问题分解为子问题
3. **知识补充**：检索失败时使用模型自身知识

### 3.4 数据质量控制

1. **格式验证**：
   - 检查每步输出是否包含必需字段
   - 格式错误最多重试2次

2. **答案验证**：
   - 使用LLM评估答案正确性
   - 只保留答案正确的轨迹

3. **多次尝试**：
   - 每个问题最多尝试7次生成
   - 至少要有1条成功的轨迹

## 四、SFT训练配置

### 4.1 训练超参数

#### **Llama-3.1-8B配置**：
```yaml
# 模型
model_name_or_path: "your_Llama-3.1-8B_model_path"
template: llama3
cutoff_len: 4096

# 训练
finetuning_type: full
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

# 数据
dataset: 2wikimultihopqa_train, hotpotqa_train, musique_train
val_size: 0.1

# DeepSpeed
deepspeed: deepspeed/ds_z3_config.json
```

#### **Qwen2.5-7B配置**：
```yaml
# 主要差异
per_device_train_batch_size: 4  # 更大的batch size
learning_rate: 7.0e-6            # 更小的学习率
template: qwen                   # 不同的模板
```

### 4.2 DeepSpeed ZeRO-3配置

```json
{
  "zero_optimization": {
    "stage": 3,                              // ZeRO-3优化
    "overlap_comm": true,                    // 通信重叠
    "contiguous_gradients": true,            // 连续梯度
    "stage3_max_live_parameters": 1e9,       // 最大存活参数
    "stage3_max_reuse_distance": 1e9,        // 最大重用距离
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

### 4.3 训练脚本

```bash
#!/bin/bash
# 自动识别GPU数量
gpu_count=$(nvidia-smi --list-gpus | wc -l)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count - 1)))

# 启动训练
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
FORCE_TORCHRUN=1 \
llamafactory-cli train examples/llama3_full_sft_qwen.yaml
```

## 五、数据集定义

在 `data/dataset_info.json` 中定义了多种数据格式：

```json
{
  // 标准格式（推理-检索链）
  "2wikimultihopqa_train": {
    "file_name": "path/to/format.json"
  },

  // 直接回答格式
  "2wikimultihopqa_train_direct_answer": {
    "file_name": "path/to/format_answer_list.json"
  },

  // 多轮对话格式
  "2wikimultihopqa_train_multi_turn": {
    "file_name": "path/to/format_multi_turn.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```

## 六、SFT训练输出

### 6.1 输出模型

- **R3-RAG-CS-Llama**: Llama-3.1-8B的Cold Start模型
- **R3-RAG-CS-Qwen**: Qwen2.5-7B的Cold Start模型

### 6.2 检查点保存

```yaml
output_dir: "your_output_path"
save_steps: 200              # 每200步保存一次
plot_loss: true              # 绘制loss曲线
report_to: tensorboard       # 使用TensorBoard记录
```

## 七、数据生成关键技术点

### 7.1 问题分解策略

**两种基本分解形式**：

1. **顺序分解（Sequential Decomposition）**：
   - 将问题分解为一系列依赖步骤
   - 每步依赖前一步的结果
   - 示例：先找到地球卫星 → 再查温度范围

2. **并行分解（Parallel Decomposition）**：
   - 将问题分解为独立子问题
   - 各子问题可并行解决
   - 最后合并结果
   - 示例：同时查询地球卫星温度和金星温度

### 7.2 检索失败处理

当检索结果不相关时：
1. **评估相关性**：判断检索文档是否回答了query
2. **问题过于宽泛**：尝试分解为更细粒度的子问题
3. **利用参数知识**：使用模型自身的知识补充
4. **重新生成query**：基于新的分析生成更精确的查询

### 7.3 Query分解机制

```python
def split_query(query):
    # 使用LLM判断query是否需要分解
    # 如果需要，分解为多个子query
    # 每个子query独立检索
    # 合并检索结果
```

示例：
```
原始query: "导演是谁？电影上映年份是什么？"
分解后:
  - "导演是谁？"
  - "电影上映年份是什么？"
```

## 八、训练数据特点

### 8.1 数据规模

- HotpotQA: ~30,000条训练样本
- 2WikiMultiHopQA: 完整训练集
- MuSiQue: 完整训练集
- **总计**: 约数万条高质量推理-检索轨迹

### 8.2 数据质量

1. **高质量teacher模型**：使用DeepSeek等强大API模型生成
2. **答案验证**：只保留正确答案的轨迹
3. **多样性**：通过温度控制和多次尝试增加多样性
4. **完整轨迹**：包含完整的推理-检索过程

### 8.3 轨迹特征

每条训练样本包含：
- **问题**：原始多跳问题
- **搜索链**：多步推理-检索过程
  - 每步包含：
    - `analysis`: 问题分析
    - `query`: 检索查询
    - `doc`: 检索到的文档
    - `answer`: 最终答案（最后一步）
- **标准答案**：用于验证

## 九、SFT与后续RL的关系

### 9.1 Cold Start作用

SFT阶段训练出的模型（R3-RAG-CS）具备：
1. **基础推理能力**：学会分解复杂问题
2. **检索意识**：知道何时需要检索
3. **格式规范**：输出符合特定格式
4. **初步策略**：简单的推理-检索策略

### 9.2 为RL铺路

Cold Start模型为强化学习提供：
1. **良好初始化**：避免从随机策略开始
2. **稳定训练**：减少RL训练的不稳定性
3. **加速收敛**：更快达到高性能
4. **策略基础**：提供基本策略框架

## 十、实际运行示例

### 10.1 完整训练流程

```bash
# 步骤1：拆分数据集
bash data/construct/01split_dataset/split_hotpotqa.sh

# 步骤2：生成搜索链
bash data/construct/02sample_from_dataset/main_hotpotqa.sh

# 步骤3：合并样本
bash data/construct/03merge_samples/merge_hotpotqa.sh

# 步骤4：格式转换
bash data/construct/format_single_query/format_hotpotqa.sh

# 步骤5：SFT训练
cd train/LLaMA-Factory
bash sft_train_qwen.sh  # 或 sft_train_llama.sh
```

### 10.2 数据统计

训练完成后查看统计：
```bash
python data/construct/03merge_samples/statistic.py \
    input_file.jsonl \
    output_stats.txt
```

输出示例：
```
Total data rows: 30000
Useful data rows: 25000
Utilization rate: 0.833
```

## 十一、总结

R3-RAG的SFT阶段是一个**精心设计的数据生成和模型训练流程**：

### 关键特点：
1. **Teacher-Student范式**：使用强大API模型生成训练数据
2. **质量优先**：严格的答案验证和多次尝试机制
3. **结构化推理**：教会模型并行和顺序分解策略
4. **检索整合**：将检索无缝融入推理过程
5. **全参数微调**：使用DeepSpeed ZeRO-3实现高效训练

### 创新点：
1. **问题分解教学**：通过精心设计的prompt教会模型分解问题
2. **自适应检索**：模型学会判断何时需要检索
3. **知识融合**：结合检索文档和参数知识
4. **质量控制**：多层次的验证机制保证数据质量

这个SFT阶段为后续的强化学习训练打下了坚实的基础，使得模型能够学习更复杂的推理-检索策略。
