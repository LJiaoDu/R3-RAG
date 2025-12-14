# R3-RAG项目使用Swift框架的可行性分析

## 📋 当前训练框架

### SFT阶段
- **框架**: LLaMA-Factory
- **版本**: 基于transformers 4.41.2+
- **功能**: 全参数微调、LoRA、DeepSpeed ZeRO-3
- **位置**: `train/LLaMA-Factory/`

### RL阶段
- **框架**: OpenRLHF
- **算法**: PPO
- **位置**: `train/R3RAG_OpenRLHF/`

---

## ✅ Swift框架可行性分析

### 什么是Swift？

**Swift（也叫ms-swift）** 是阿里云ModelScope团队开发的大模型训练框架。

- **GitHub**: https://github.com/modelscope/swift
- **功能**: SFT、RLHF、多模态训练
- **特点**: 轻量级、易用、支持多种模型

### 核心问题：能否替代LLaMA-Factory？

**答案：可以！** ✅

理由：
1. **功能对等**：Swift支持全参数微调、LoRA、QLoRA等
2. **模型兼容**：支持Llama、Qwen等项目使用的模型
3. **分布式训练**：支持DeepSpeed、FSDP
4. **数据格式**：支持多种数据格式（json、jsonl、sharegpt等）

---

## 🔄 迁移方案

### 方案1: SFT阶段使用Swift

#### 安装Swift
```bash
pip install ms-swift -U

# 或者从源码安装
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .
```

#### 准备数据
Swift支持多种数据格式，R3-RAG的数据格式可以直接使用：

```json
// 当前LLaMA-Factory格式
{
  "instruction": "问题",
  "input": "",
  "output": "推理-检索链"
}

// Swift也支持这种格式，或者使用sharegpt格式
{
  "messages": [
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "推理-检索链"}
  ]
}
```

#### 训练脚本示例

**替代 `sft_train_qwen.sh`**：

```bash
#!/bin/bash

# Swift SFT训练脚本
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model_type qwen2-7b-instruct \
    --model_id_or_path /path/to/Qwen2.5-7B \
    --dataset /path/to/your/dataset.jsonl \
    --output_dir ./output/qwen-sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 7e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --deepspeed default-zero3 \
    --max_length 4096 \
    --logging_steps 2 \
    --save_steps 200 \
    --eval_steps 500 \
    --val_dataset_sample 0.1
```

**替代 `sft_train_llama.sh`**：

```bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model_type llama3_1-8b-instruct \
    --model_id_or_path /path/to/Llama-3.1-8B \
    --dataset /path/to/your/dataset.jsonl \
    --output_dir ./output/llama-sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --deepspeed default-zero3 \
    --max_length 4096 \
    --logging_steps 2 \
    --save_steps 200 \
    --eval_steps 500
```

#### 配置文件方式（推荐）

创建 `swift_sft_config.json`：

```json
{
  "model_type": "qwen2-7b-instruct",
  "model_id_or_path": "/path/to/Qwen2.5-7B",
  "dataset": ["2wikimultihopqa_train", "hotpotqa_train", "musique_train"],
  "output_dir": "./output/qwen-sft",
  "num_train_epochs": 3,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 4,
  "learning_rate": 7e-6,
  "lr_scheduler_type": "cosine",
  "warmup_ratio": 0.1,
  "bf16": true,
  "deepspeed": "default-zero3",
  "max_length": 4096,
  "logging_steps": 2,
  "save_steps": 200,
  "eval_steps": 500,
  "save_total_limit": 5
}
```

运行：
```bash
swift sft --config swift_sft_config.json
```

---

## 📊 功能对比

| 功能 | LLaMA-Factory | Swift | 说明 |
|------|---------------|-------|------|
| **基础功能** | | | |
| 全参数微调 | ✅ | ✅ | 两者都支持 |
| LoRA/QLoRA | ✅ | ✅ | 两者都支持 |
| DeepSpeed | ✅ | ✅ | 都支持ZeRO 1/2/3 |
| FSDP | ✅ | ✅ | 都支持 |
| **模型支持** | | | |
| Llama系列 | ✅ | ✅ | 都支持 |
| Qwen系列 | ✅ | ✅ | 都支持 |
| 其他模型 | ✅ | ✅ | 都支持100+模型 |
| **数据格式** | | | |
| JSON/JSONL | ✅ | ✅ | 都支持 |
| ShareGPT | ✅ | ✅ | 都支持 |
| 自定义格式 | ✅ | ✅ | 都支持 |
| **高级功能** | | | |
| 多模态 | ✅ | ✅ | 都支持 |
| RLHF | ✅ | ✅ | Swift也支持 |
| Agent训练 | ⚠️ | ✅ | Swift更强 |
| 推理加速 | ✅ | ✅ | 都支持vLLM |
| **易用性** | | | |
| CLI接口 | ✅ | ✅ | 都很方便 |
| Web UI | ✅ | ✅ | 都有 |
| 文档质量 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Swift中文文档更好 |
| 社区活跃度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | LLaMA-Factory更活跃 |

---

## ⚖️ 优缺点对比

### LLaMA-Factory优势

✅ **社区更活跃**
- GitHub Star更多（约70K+ vs 15K+）
- 问题响应更快
- 更多实践案例

✅ **文档更全面**
- 英文文档完善
- 示例丰富

✅ **项目已集成**
- 无需改动现有代码
- 配置文件已调优

### Swift优势

✅ **ModelScope生态**
- 与ModelScope无缝集成
- 可以直接从ModelScope下载模型
- 支持模型上传到ModelScope

✅ **中文文档友好**
- 中文文档非常详细
- 国内团队维护，响应快

✅ **Agent训练支持更好**
- 对Agent、Tool Calling支持更完善
- 更适合构建智能体

✅ **内置更多功能**
- 自带推理脚本
- 自带评测脚本
- 自带部署工具

---

## 🎯 迁移建议

### 场景1: 新项目或重新训练

**建议：可以尝试Swift** ✅

原因：
- 功能对等，性能相当
- 中文文档友好
- ModelScope生态完善

迁移步骤：
1. 安装Swift：`pip install ms-swift -U`
2. 转换数据格式（如果需要）
3. 修改训练脚本（参考上面的示例）
4. 测试训练流程
5. 对比性能

### 场景2: 已有训练好的模型

**建议：继续使用LLaMA-Factory** ⚠️

原因：
- 已有配置经过调优
- checkpoint格式兼容性最好
- 避免不必要的风险

### 场景3: 只是想体验新框架

**建议：并行尝试** 💡

可以：
- 在小数据集上用Swift训练
- 对比训练速度和效果
- 评估是否值得切换

---

## 🚀 实际操作步骤

### Step 1: 安装Swift（保留LLaMA-Factory）

```bash
# 创建新的conda环境（推荐）
conda create -n swift python=3.10
conda activate swift

# 安装Swift
pip install ms-swift -U

# 或者源码安装（获取最新功能）
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .
```

### Step 2: 准备数据

Swift可以直接使用R3-RAG的数据格式，无需转换。

如果想用Swift的数据格式，可以写个转换脚本：

```python
# convert_to_swift.py
import json

def convert_llamafactory_to_swift(input_file, output_file):
    """转换LLaMA-Factory格式到Swift格式"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)

            # 转换为messages格式
            swift_data = {
                "messages": [
                    {"role": "user", "content": data["instruction"]},
                    {"role": "assistant", "content": data["output"]}
                ]
            }

            f_out.write(json.dumps(swift_data, ensure_ascii=False) + '\n')

# 使用
convert_llamafactory_to_swift(
    "data/hotpotqa_train.jsonl",
    "data/hotpotqa_train_swift.jsonl"
)
```

### Step 3: 创建训练脚本

在 `train/Swift/` 目录下创建：

```bash
mkdir -p train/Swift
cd train/Swift

# 创建训练脚本
cat > sft_train_qwen_swift.sh << 'EOF'
#!/bin/bash

# 自动识别GPU数量
gpu_count=$(nvidia-smi --list-gpus | wc -l)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpu_count - 1)))

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Swift SFT训练
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
swift sft \
    --model_type qwen2-7b-instruct \
    --model_id_or_path /path/to/Qwen2.5-7B \
    --dataset ../../data/2wikimultihopqa_train.jsonl \
             ../../data/hotpotqa_train.jsonl \
             ../../data/musique_train.jsonl \
    --output_dir ./output/qwen-r3rag-sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 7e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --deepspeed default-zero3 \
    --max_length 4096 \
    --logging_steps 2 \
    --save_steps 200 \
    --eval_steps 500 \
    --save_total_limit 5 \
    --gradient_checkpointing true
EOF

chmod +x sft_train_qwen_swift.sh
```

### Step 4: 运行训练

```bash
cd train/Swift
bash sft_train_qwen_swift.sh
```

### Step 5: 监控训练

```bash
# Swift自带tensorboard支持
tensorboard --logdir ./output/qwen-r3rag-sft
```

### Step 6: 推理测试

```bash
# Swift提供简单的推理接口
swift infer \
    --ckpt_dir ./output/qwen-r3rag-sft/checkpoint-xxx \
    --load_dataset_config true
```

---

## 🔍 注意事项

### 1. checkpoint兼容性

⚠️ **Swift和LLaMA-Factory的checkpoint可能不完全兼容**

建议：
- 训练完成后导出为标准HuggingFace格式
- 使用统一的推理框架（如vLLM）

### 2. 配置差异

两个框架的默认配置可能不同，需要注意：
- 学习率调度策略
- 梯度累积步数
- warmup设置
- 保存策略

### 3. RL阶段

⚠️ **目前R3-RAG的RL阶段使用OpenRLHF**

Swift虽然也支持RLHF，但：
- API可能不同
- 需要重写RL训练代码
- 建议RL阶段仍使用OpenRLHF

### 4. 性能对比

建议迁移后做性能对比：
- 训练速度
- 显存占用
- 最终模型效果

---

## 📝 总结和建议

### 我的建议

**对于R3-RAG项目**：

#### 短期（1-2周）
**继续使用LLaMA-Factory** ✅
- 已有配置稳定可靠
- 避免不必要风险
- 专注于项目本身

#### 中期（1-2个月）
**可以尝试Swift** 💡
- 在小规模实验中测试
- 对比性能和易用性
- 评估是否值得切换

#### 长期（3个月+）
**根据需求选择** 🎯
- 如果需要ModelScope生态 → Swift
- 如果需要更多社区支持 → LLaMA-Factory
- 如果两者都需要 → 都保留

### 快速决策树

```
是否已有训练好的模型？
├─ 是 → 继续用LLaMA-Factory
└─ 否 → 是否需要ModelScope生态？
        ├─ 是 → 用Swift
        └─ 否 → 是否有充足时间测试？
                ├─ 是 → 都试试，选更好的
                └─ 否 → 用LLaMA-Factory（稳妥）
```

### 面试时如何回答

**面试官问：这个项目能用Swift吗？**

你可以这样回答：
> "可以的。Swift和LLaMA-Factory功能上基本对等，都支持全参数微调、LoRA、DeepSpeed等。Swift是阿里云的框架，中文文档更友好，和ModelScope生态集成也更好。
>
> 不过我们项目当前用的是LLaMA-Factory，主要考虑是：一、社区更活跃，有问题容易找到解决方案；二、已有配置经过调优，比较稳定。
>
> 如果重新做这个项目，我会考虑两个框架都试一下，选择更适合的。迁移的话主要就是改一下训练脚本，数据格式基本兼容，技术上没有太大障碍。"

---

## 🔗 参考资源

- **Swift GitHub**: https://github.com/modelscope/swift
- **Swift 文档**: https://swift.readthedocs.io/
- **LLaMA-Factory GitHub**: https://github.com/hiyouga/LLaMA-Factory
- **ModelScope**: https://modelscope.cn/

---

**结论**：技术上完全可行，但建议根据项目阶段和需求谨慎决策！
