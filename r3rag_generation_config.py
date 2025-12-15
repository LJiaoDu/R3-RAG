#!/usr/bin/env python3
"""
R3-RAG数据生成参数配置
基于原论文 data/construct/02sample_from_dataset/main.py
"""

# ===================== 核心参数 =====================

# 推理链生成参数
REASONING_CHAIN_CONFIG = {
    # 每次检索返回的文档数
    "num_passages_per_retrieval": 3,

    # 每个问题尝试生成的次数
    "num_attempts_per_question": 7,

    # 每次尝试最多允许的检索步数
    "max_search_steps_per_attempt": 7,

    # Temperature设置
    "temperature_first_attempt": 0.0,   # 第1次尝试：最保守
    "temperature_other_attempts": 0.9,  # 第2-7次：探索多样性

    # 最大输出tokens（每步）
    "max_tokens_per_step": 500,

    # 答案验证阈值
    "min_answer_length": 50,  # 最少字符数
    "max_answer_length": 2000,  # 最多字符数
}

# 数据集配置
DATASET_CONFIG = {
    "num_sft_training": 8000,   # SFT训练数据（推理链）
    "num_grpo_training": 5000,  # GRPO训练数据（参考答案）
    "num_test": 2000,           # 测试集（参考答案）

    # 问题分配
    "num_questions_for_sft": 2000,    # 为SFT准备的问题数
    "num_questions_for_grpo": 7000,   # 为GRPO+TEST准备的问题数
}

# ===================== 详细说明 =====================

"""
1. num_passages_per_retrieval = 3

   每次Search[query]返回3个最相关文档

   示例：
   Search[Machine Learning prerequisites]
   → 返回：
     [1] ML requires: Linear Algebra, Probability, Python
     [2] Prerequisites must be passed before enrollment
     [3] ML is a Kernmodul in ML track

   为什么是3？
   - 提供足够信息（不像1个那么少）
   - 避免信息过载（不像10个那么多）
   - 经验最佳值

2. num_attempts_per_question = 7

   每个问题尝试生成7次，增加数据多样性

   流程：
   attempt 1: temperature=0.0  → 保守路径（最确定的推理）
   attempt 2: temperature=0.9  → 探索不同路径
   attempt 3: temperature=0.9  → 更多变化
   ...
   attempt 7: temperature=0.9  → 最大多样性

   实际结果（假设70%成功率）：
   - 每个问题平均得到 7 × 70% = 4.9条成功的链
   - 这些链有不同的推理路径

   示例（同一问题的3条不同链）：

   链1 (temp=0.0):
     Search[ML prerequisites] → Search[LA prereqs] → Finish

   链2 (temp=0.9):
     Search[ML requirements] → Search[study path] → Search[ECTS] → Finish

   链3 (temp=0.9):
     Search[TUM ML course] → Search[semester planning] → Finish

3. max_search_steps_per_attempt = 7

   每次尝试最多允许7步检索 + 1步Finish

   步数分布（实际数据，经验值）：
   2步（1 Search + 1 Finish）: 15% - 简单问题
     "How many ECTS is ML?"
     → Search[ML ECTS] → Finish

   3步（2 Search + 1 Finish）: 25% - 单链推理
     "What's before ML?"
     → Search[ML prereqs] → Search[LA prereqs] → Finish

   4步（3 Search + 1 Finish）: 30% - 双链推理
     "Complete path to ML?"
     → Search[ML] → Search[LA] → Search[Prob] → Finish

   5步（4 Search + 1 Finish）: 20% - 多步推理
   6步（5 Search + 1 Finish）: 8%  - 复杂推理
   7步（6 Search + 1 Finish）: 2%  - 极复杂

   平均：3.8步/链

   为什么限制7步？
   - 避免过长的推理链（难以训练）
   - 大部分问题3-5步就够
   - 超过7步可能说明问题分解有问题

4. temperature_first_attempt = 0.0

   第一次尝试用temperature=0.0，生成最确定的答案

   特点：
   - 推理路径最保守
   - 选择最常见的检索策略
   - 答案最可靠

   用途：
   - 确保至少有1条高质量的链
   - 作为baseline

5. temperature_other_attempts = 0.9

   第2-7次尝试用temperature=0.9，探索多样性

   特点：
   - 推理路径更多样
   - 可能尝试不同的检索角度
   - 答案表述更丰富

   用途：
   - 增加训练数据多样性
   - 让模型学会不同的推理策略
   - 避免过拟合到单一路径
"""

# ===================== 数据量计算 =====================

def estimate_data_generation():
    """估算数据生成量"""

    num_questions = DATASET_CONFIG["num_questions_for_sft"]
    attempts_per_q = REASONING_CHAIN_CONFIG["num_attempts_per_question"]

    # 假设成功率
    success_rate = 0.70  # 70%的尝试会成功

    total_attempts = num_questions * attempts_per_q
    expected_successful_chains = total_attempts * success_rate
    avg_chains_per_question = expected_successful_chains / num_questions

    print("="*60)
    print("数据生成估算")
    print("="*60)
    print(f"\n输入问题数: {num_questions}")
    print(f"每个问题尝试次数: {attempts_per_q}")
    print(f"总尝试次数: {total_attempts}")
    print(f"\n假设成功率: {success_rate*100:.0f}%")
    print(f"预期成功链数: {expected_successful_chains:.0f}")
    print(f"平均每个问题: {avg_chains_per_question:.1f}条链")

    # 步数分布
    print(f"\n预期步数分布:")
    step_distribution = {
        2: 0.15,
        3: 0.25,
        4: 0.30,
        5: 0.20,
        6: 0.08,
        7: 0.02
    }

    avg_steps = 0
    for steps, ratio in step_distribution.items():
        count = expected_successful_chains * ratio
        avg_steps += steps * ratio
        print(f"  {steps}步: {count:6.0f}条 ({ratio*100:4.0f}%)")

    print(f"\n平均步数: {avg_steps:.1f}步/链")

    # 目标数据量
    target_sft = DATASET_CONFIG["num_sft_training"]
    print(f"\n目标SFT数据: {target_sft}条")

    if expected_successful_chains >= target_sft:
        print(f"✅ 预期可生成 {expected_successful_chains:.0f}条，足够！")
        print(f"   富余: {expected_successful_chains - target_sft:.0f}条")
    else:
        shortfall = target_sft - expected_successful_chains
        extra_questions_needed = shortfall / avg_chains_per_question
        print(f"⚠️  预期只能生成 {expected_successful_chains:.0f}条，不够！")
        print(f"   缺少: {shortfall:.0f}条")
        print(f"   需要额外 {extra_questions_needed:.0f}个问题")

    return expected_successful_chains


# ===================== 成本估算 =====================

def estimate_cost():
    """估算OpenAI API成本"""

    # GPT-4o-mini价格（per 1M tokens）
    price_input = 0.150 / 1_000_000
    price_output = 0.600 / 1_000_000

    num_questions = DATASET_CONFIG["num_questions_for_sft"]
    attempts = REASONING_CHAIN_CONFIG["num_attempts_per_question"]
    avg_steps = 3.8  # 平均步数

    # 每步的token消耗
    input_tokens_per_step = 200   # system prompt + history
    output_tokens_per_step = 150  # thought + action

    # 每次尝试的token消耗
    tokens_per_attempt = avg_steps * (input_tokens_per_step + output_tokens_per_step)

    # 总token消耗
    total_attempts = num_questions * attempts
    total_input_tokens = total_attempts * avg_steps * input_tokens_per_step
    total_output_tokens = total_attempts * avg_steps * output_tokens_per_step

    # 成本
    cost_input = total_input_tokens * price_input
    cost_output = total_output_tokens * price_output
    total_cost = cost_input + cost_output

    print("\n" + "="*60)
    print("成本估算（SFT推理链生成）")
    print("="*60)
    print(f"\n问题数: {num_questions}")
    print(f"每个问题尝试: {attempts}次")
    print(f"总尝试次数: {total_attempts}")
    print(f"平均步数: {avg_steps}步/尝试")

    print(f"\nToken消耗:")
    print(f"  输入: {total_input_tokens:,} tokens")
    print(f"  输出: {total_output_tokens:,} tokens")

    print(f"\n成本:")
    print(f"  输入成本: ${cost_input:.2f}")
    print(f"  输出成本: ${cost_output:.2f}")
    print(f"  总成本: ${total_cost:.2f}")

    # GRPO数据成本
    num_grpo = DATASET_CONFIG["num_grpo_training"] + DATASET_CONFIG["num_test"]
    grpo_input = num_grpo * 150
    grpo_output = num_grpo * 300
    grpo_cost = grpo_input * price_input + grpo_output * price_output

    print(f"\n参考答案生成成本（GRPO+TEST）:")
    print(f"  {num_grpo}条 × 450 tokens = ${grpo_cost:.2f}")

    print(f"\n总计: ${total_cost + grpo_cost:.2f}")

    return total_cost + grpo_cost


# ===================== 时间估算 =====================

def estimate_time():
    """估算生成时间"""

    num_questions = DATASET_CONFIG["num_questions_for_sft"]
    attempts = REASONING_CHAIN_CONFIG["num_attempts_per_question"]
    avg_steps = 3.8

    # 每步耗时（包括API调用 + 检索）
    seconds_per_step = 3  # GPT-4 API通常1-2秒，检索1秒

    # 单个尝试耗时
    seconds_per_attempt = avg_steps * seconds_per_step

    # 总耗时（串行）
    total_attempts = num_questions * attempts
    total_seconds = total_attempts * seconds_per_attempt
    total_hours = total_seconds / 3600
    total_days = total_hours / 24

    # 并行处理
    parallel_workers = 10  # 假设10个并行
    parallel_hours = total_hours / parallel_workers
    parallel_days = parallel_hours / 24

    print("\n" + "="*60)
    print("时间估算")
    print("="*60)
    print(f"\n单次尝试: {avg_steps:.1f}步 × {seconds_per_step}秒 = {seconds_per_attempt:.0f}秒")
    print(f"总尝试次数: {total_attempts}")

    print(f"\n串行处理:")
    print(f"  总时间: {total_seconds:,}秒")
    print(f"  = {total_hours:.1f}小时")
    print(f"  = {total_days:.1f}天")

    print(f"\n并行处理（{parallel_workers}线程）:")
    print(f"  总时间: {parallel_hours:.1f}小时")
    print(f"  = {parallel_days:.1f}天")

    # GRPO时间
    num_grpo = DATASET_CONFIG["num_grpo_training"] + DATASET_CONFIG["num_test"]
    grpo_seconds = num_grpo * 10  # 每条10秒
    grpo_hours = grpo_seconds / 3600

    print(f"\n参考答案生成（GRPO+TEST）:")
    print(f"  {num_grpo}条 × 10秒 = {grpo_hours:.1f}小时")

    total_time = parallel_hours + grpo_hours
    print(f"\n总计（并行）: {total_time:.1f}小时 = {total_time/24:.1f}天")

    return total_time


# ===================== 使用示例 =====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("R3-RAG数据生成参数配置")
    print("="*60)

    print("\n核心参数:")
    print(f"  每次检索文档数: {REASONING_CHAIN_CONFIG['num_passages_per_retrieval']}")
    print(f"  每个问题尝试次数: {REASONING_CHAIN_CONFIG['num_attempts_per_question']}")
    print(f"  每次尝试最大步数: {REASONING_CHAIN_CONFIG['max_search_steps_per_attempt']}")
    print(f"  Temperature（第1次）: {REASONING_CHAIN_CONFIG['temperature_first_attempt']}")
    print(f"  Temperature（其他）: {REASONING_CHAIN_CONFIG['temperature_other_attempts']}")

    print("\n数据集配置:")
    print(f"  SFT训练: {DATASET_CONFIG['num_sft_training']}条")
    print(f"  GRPO训练: {DATASET_CONFIG['num_grpo_training']}条")
    print(f"  测试集: {DATASET_CONFIG['num_test']}条")

    # 运行估算
    estimate_data_generation()
    estimate_cost()
    estimate_time()

    print("\n" + "="*60)
    print("配置完成！")
    print("="*60)
