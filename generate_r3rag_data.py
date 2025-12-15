#!/usr/bin/env python3
"""
为R3-RAG生成训练数据：
- 10000条推理链 → SFT训练（冷启动）
- 5000条参考答案 → GRPO训练 + 测试集
"""

import openai
import json
import time
from pathlib import Path
from tqdm import tqdm
import asyncio
import aiohttp
from typing import List, Dict
import os
import random

# ===================== 配置 =====================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

MODEL = "gpt-4o-mini"

# 文件配置
INPUT_FILE = "questions.jsonl"
OUTPUT_SFT = "sft_training_data.jsonl"          # SFT训练数据（带推理链）
OUTPUT_GRPO = "grpo_reference_data.jsonl"      # GRPO参考答案（无推理链）
CHECKPOINT_FILE = "checkpoint_r3rag.json"

# 数量配置
NUM_SFT_TRAINING = 10000      # SFT训练数据（推理链）
NUM_GRPO_REFERENCE = 5000     # GRPO + 测试（参考答案）

# 建议进一步分割GRPO数据
NUM_GRPO_TRAIN = 4000         # GRPO训练
NUM_GRPO_TEST = 1000          # 测试集

# 批处理配置
BATCH_SIZE = 50
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.0

# ===================== Prompt模板 =====================

# SFT训练：生成完整推理链
SYSTEM_PROMPT_SFT = """You are an expert assistant for TUM (Technical University of Munich) course consultation.

You have access to a course database. Answer questions using step-by-step reasoning:

Thought: [Your reasoning about what to do next]
Action: Search[query] OR Finish[answer]
Observation: [Search results OR final answer]

Important:
- Use Search[query] to look up TUM course information (prerequisites, ECTS, schedules, etc.)
- Show clear reasoning at each step
- Use Finish[answer] when you have the complete answer
- Be specific about course codes (e.g., IN2064, MA1001)

Example:

Question: "I want to take Machine Learning. What are the prerequisites?"

Thought: I need to find the official prerequisites for the Machine Learning course at TUM.
Action: Search[Machine Learning IN2064 prerequisites requirements]
Observation: Machine Learning (IN2064) requires: 1) Linear Algebra, 2) Probability Theory, 3) Programming experience in Python.

Thought: Let me check if these prerequisites have their own requirements.
Action: Search[Linear Algebra MA1001 prerequisites TUM]
Observation: Linear Algebra for Informatics (MA1001) is a foundational course with no prerequisites. Typically taken in 1st semester.

Thought: Now check Probability Theory.
Action: Search[Probability Theory MA2009 prerequisites]
Observation: Probability Theory (MA2009) requires Linear Algebra as a prerequisite.

Thought: I now have complete information about the prerequisite chain.
Action: Finish[To take Machine Learning (IN2064) at TUM, you need:

**Prerequisites:**
1. Linear Algebra for Informatics (MA1001) - 8 ECTS
   - No prerequisites
   - Typically 1st semester

2. Probability Theory (MA2009) - 6 ECTS
   - Requires: Linear Algebra
   - Typically 2nd semester

3. Programming Skills
   - Python proficiency required
   - Can be self-taught or via Python courses

**Study Path:**
Semester 1: Linear Algebra + Python
Semester 2: Probability Theory
Semester 3: Machine Learning

Total preparation time: 2-3 semesters depending on your programming background.]

---

Now answer the following question with similar reasoning.
"""

# GRPO参考答案：只生成答案（无推理链）
SYSTEM_PROMPT_GRPO = """You are an expert assistant for TUM (Technical University of Munich) course consultation.

Provide a direct, comprehensive, and accurate answer to the question.

Requirements:
- Be specific and factual
- Include relevant course codes (e.g., IN2064, MA1001)
- Provide complete information
- Use clear structure (bullet points, numbers)
- Support both English and German questions

Do NOT include reasoning steps like "Thought:", "Action:", or "Observation:".
Only provide the final answer.

Example:

Question: "I want to take Machine Learning. What are the prerequisites?"

Answer:
To take Machine Learning (IN2064) at TUM, you need:

**Prerequisites:**
1. Linear Algebra for Informatics (MA1001) - 8 ECTS
   - No prerequisites
   - Typically 1st semester

2. Probability Theory (MA2009) - 6 ECTS
   - Requires: Linear Algebra
   - Typically 2nd semester

3. Programming Skills
   - Python proficiency required
   - Can be self-taught or via Python courses

**Study Path:**
Semester 1: Linear Algebra + Python
Semester 2: Probability Theory
Semester 3: Machine Learning

Total preparation time: 2-3 semesters depending on your programming background.

---

Now answer the following question.
"""

# ===================== 数据分配 =====================

def assign_data_splits(questions: List[Dict],
                       num_sft: int,
                       num_grpo: int,
                       strategy: str = 'random') -> List[Dict]:
    """
    为问题分配用途：SFT训练 vs GRPO参考

    strategy:
    - 'random': 随机分配（推荐，避免偏差）
    - 'first': 前N个SFT，后M个GRPO
    - 'balanced': 保持每类问题的比例一致
    """

    total = len(questions)

    if total != num_sft + num_grpo:
        print(f"⚠️  警告: 问题总数({total}) != SFT({num_sft}) + GRPO({num_grpo})")
        print(f"   将调整为: SFT={num_sft}, GRPO={min(num_grpo, total-num_sft)}")

    if strategy == 'random':
        print("🎲 随机分配数据集...")

        # 随机打乱
        shuffled = questions.copy()
        random.shuffle(shuffled)

        # 分配
        for i, q in enumerate(shuffled):
            if i < num_sft:
                q['data_split'] = 'sft'
            else:
                q['data_split'] = 'grpo'

    elif strategy == 'first':
        print("📋 按顺序分配...")
        for i, q in enumerate(questions):
            q['data_split'] = 'sft' if i < num_sft else 'grpo'

    elif strategy == 'balanced':
        # 如果问题有类别标签，保持每类的比例
        print("⚖️  平衡分配（保持类别比例）...")

        # 按类别分组
        categories = {}
        for q in questions:
            cat = q.get('category', 'default')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(q)

        # 每个类别按比例分配
        sft_ratio = num_sft / total

        sft_assigned = 0
        for cat, cat_questions in categories.items():
            random.shuffle(cat_questions)
            cat_sft_count = int(len(cat_questions) * sft_ratio)

            for i, q in enumerate(cat_questions):
                if i < cat_sft_count and sft_assigned < num_sft:
                    q['data_split'] = 'sft'
                    sft_assigned += 1
                else:
                    q['data_split'] = 'grpo'

    # 统计
    sft_count = sum(1 for q in questions if q.get('data_split') == 'sft')
    grpo_count = sum(1 for q in questions if q.get('data_split') == 'grpo')

    print(f"✅ 分配完成:")
    print(f"   - SFT训练: {sft_count} 条（推理链）")
    print(f"   - GRPO+测试: {grpo_count} 条（参考答案）")

    return questions


# ===================== 生成函数 =====================

async def generate_answer_async(
    session: aiohttp.ClientSession,
    question_data: Dict,
    semaphore: asyncio.Semaphore
) -> Dict:
    """异步生成答案（根据data_split选择prompt）"""
    async with semaphore:
        question_text = question_data.get('question', question_data.get('text'))
        data_split = question_data.get('data_split', 'sft')

        # 选择prompt
        if data_split == 'sft':
            system_prompt = SYSTEM_PROMPT_SFT
            max_tokens = 1000  # 推理链需要更多tokens
        else:  # grpo
            system_prompt = SYSTEM_PROMPT_GRPO
            max_tokens = 500   # 只有答案，较短

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_text}
        ]

        # 重试逻辑
        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    openai.ChatCompletion.create,
                    model=MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=max_tokens
                )

                answer = response.choices[0].message.content

                result = {
                    **question_data,
                    'answer': answer,
                    'data_split': data_split,
                    'model': MODEL,
                    'timestamp': time.time()
                }

                return result

            except openai.error.RateLimitError:
                wait_time = (2 ** attempt) * RATE_LIMIT_DELAY
                await asyncio.sleep(wait_time)

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {
                        **question_data,
                        'answer': None,
                        'error': str(e),
                        'data_split': data_split
                    }
                await asyncio.sleep(2 ** attempt)

        return {
            **question_data,
            'answer': None,
            'error': 'Max retries exceeded',
            'data_split': data_split
        }


async def process_batch(
    questions_batch: List[Dict],
    checkpoint_file: str,
    processed_ids: set,
    semaphore: asyncio.Semaphore
):
    """处理一批问题，根据data_split写入不同文件"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            generate_answer_async(session, q, semaphore)
            for q in questions_batch
        ]

        results = await asyncio.gather(*tasks)

        # 按data_split分别保存
        for result in results:
            data_split = result.get('data_split', 'sft')

            if data_split == 'sft':
                output_file = OUTPUT_SFT
            else:
                output_file = OUTPUT_GRPO

            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

            question_id = result.get('id', result.get('question'))
            processed_ids.add(question_id)

        # 保存checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({'processed_ids': list(processed_ids)}, f)


async def main_async(questions: List[Dict], checkpoint_file: str):
    """主函数"""
    # 加载checkpoint
    processed_ids = set()
    if Path(checkpoint_file).exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            processed_ids = set(checkpoint.get('processed_ids', []))
        print(f"📍 恢复checkpoint：已处理 {len(processed_ids)} 个")

    # 过滤未处理的
    remaining = [
        q for q in questions
        if q.get('id', q.get('question')) not in processed_ids
    ]

    print(f"🚀 需要处理 {len(remaining)} 个问题")

    if len(remaining) == 0:
        print("✅ 所有问题已处理！")
        return

    semaphore = asyncio.Semaphore(BATCH_SIZE)

    with tqdm(total=len(remaining), desc="生成数据") as pbar:
        for i in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[i:i + BATCH_SIZE]

            await process_batch(
                batch,
                checkpoint_file,
                processed_ids,
                semaphore
            )

            pbar.update(len(batch))

            if i + BATCH_SIZE < len(remaining):
                await asyncio.sleep(RATE_LIMIT_DELAY)

    print(f"✅ 完成！")
    print(f"   - SFT训练数据: {OUTPUT_SFT}")
    print(f"   - GRPO参考数据: {OUTPUT_GRPO}")


# ===================== 成本估算 =====================

def estimate_cost(num_sft: int, num_grpo: int, model: str = "gpt-4o-mini"):
    """估算成本"""
    prices = {
        "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    }

    price = prices.get(model, prices["gpt-4o-mini"])

    # SFT：推理链（更多tokens）
    sft_input = 200   # 长system prompt
    sft_output = 600  # 完整推理链

    # GRPO：只有答案
    grpo_input = 150
    grpo_output = 300

    cost_sft = num_sft * (sft_input * price["input"] + sft_output * price["output"])
    cost_grpo = num_grpo * (grpo_input * price["input"] + grpo_output * price["output"])

    total_cost = cost_sft + cost_grpo

    print(f"\n💰 成本估算 ({model}):")
    print(f"\n   📚 SFT训练数据 ({num_sft}条，带推理链):")
    print(f"      输入: {num_sft * sft_input:,} tokens → ${num_sft * sft_input * price['input']:.2f}")
    print(f"      输出: {num_sft * sft_output:,} tokens → ${num_sft * sft_output * price['output']:.2f}")
    print(f"      小计: ${cost_sft:.2f}")

    print(f"\n   🎯 GRPO参考数据 ({num_grpo}条，只有答案):")
    print(f"      输入: {num_grpo * grpo_input:,} tokens → ${num_grpo * grpo_input * price['input']:.2f}")
    print(f"      输出: {num_grpo * grpo_output:,} tokens → ${num_grpo * grpo_output * price['output']:.2f}")
    print(f"      小计: ${cost_grpo:.2f}")

    print(f"\n   💵 总成本: ${total_cost:.2f}")
    print(f"\n   ⏱️  预计时间: {(num_sft + num_grpo) / BATCH_SIZE * RATE_LIMIT_DELAY / 3600:.1f} 小时")


# ===================== 后处理：分割GRPO数据 =====================

def split_grpo_data(grpo_file: str, train_count: int, test_count: int):
    """将GRPO数据分割为训练集和测试集"""
    print(f"\n📂 分割GRPO数据...")

    # 读取所有GRPO数据
    with open(grpo_file, 'r', encoding='utf-8') as f:
        grpo_data = [json.loads(line) for line in f]

    # 只保留成功的
    grpo_data = [d for d in grpo_data if d.get('answer')]

    # 随机打乱
    random.shuffle(grpo_data)

    # 分割
    grpo_train = grpo_data[:train_count]
    grpo_test = grpo_data[train_count:train_count + test_count]

    # 保存
    with open('grpo_train.jsonl', 'w', encoding='utf-8') as f:
        for item in grpo_train:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open('grpo_test.jsonl', 'w', encoding='utf-8') as f:
        for item in grpo_test:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ GRPO数据分割完成:")
    print(f"   - 训练集: {len(grpo_train)} 条 → grpo_train.jsonl")
    print(f"   - 测试集: {len(grpo_test)} 条 → grpo_test.jsonl")


# ===================== 主程序 =====================

def main():
    if not OPENAI_API_KEY:
        print("❌ 请设置 OPENAI_API_KEY 环境变量")
        return

    if not Path(INPUT_FILE).exists():
        print(f"❌ 输入文件不存在: {INPUT_FILE}")
        return

    # 加载问题
    print("📂 加载问题...")
    questions = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))

    print(f"✅ 加载了 {len(questions)} 个问题")

    # 估算成本
    estimate_cost(NUM_SFT_TRAINING, NUM_GRPO_REFERENCE, MODEL)

    # 选择分配策略
    print("\n选择数据分配策略:")
    print("1. 随机分配（推荐）- 避免数据偏差")
    print("2. 按顺序分配 - 前10000个SFT，后5000个GRPO")
    print("3. 平衡分配 - 保持每类问题比例一致")

    choice = input("\n请选择 (1/2/3): ").strip()
    strategy_map = {'1': 'random', '2': 'first', '3': 'balanced'}
    strategy = strategy_map.get(choice, 'random')

    # 分配数据集
    questions = assign_data_splits(
        questions,
        NUM_SFT_TRAINING,
        NUM_GRPO_REFERENCE,
        strategy
    )

    # 确认继续
    print(f"\n是否继续生成？(y/n): ", end='')
    if input().strip().lower() != 'y':
        print("已取消")
        return

    # 开始生成
    asyncio.run(main_async(questions, CHECKPOINT_FILE))

    # 分割GRPO数据
    if Path(OUTPUT_GRPO).exists():
        split_grpo_data(OUTPUT_GRPO, NUM_GRPO_TRAIN, NUM_GRPO_TEST)

    # 统计结果
    print("\n" + "="*60)
    print("📊 最终统计:")
    print("="*60)

    if Path(OUTPUT_SFT).exists():
        with open(OUTPUT_SFT, 'r') as f:
            sft_data = [json.loads(line) for line in f]
        sft_success = sum(1 for d in sft_data if d.get('answer'))
        print(f"\n✅ SFT训练数据: {len(sft_data)} 条")
        print(f"   成功: {sft_success} ({sft_success/len(sft_data)*100:.1f}%)")

    if Path('grpo_train.jsonl').exists():
        with open('grpo_train.jsonl', 'r') as f:
            grpo_train = [json.loads(line) for line in f]
        print(f"\n✅ GRPO训练数据: {len(grpo_train)} 条")

    if Path('grpo_test.jsonl').exists():
        with open('grpo_test.jsonl', 'r') as f:
            grpo_test = [json.loads(line) for line in f]
        print(f"\n✅ 测试数据: {len(grpo_test)} 条")

    print("\n" + "="*60)
    print("🎉 全部完成！")
    print("="*60)
    print("\n下一步:")
    print("1. 将 sft_training_data.jsonl 转换为LLaMA-Factory格式")
    print("2. 运行SFT训练")
    print("3. 用 grpo_train.jsonl 进行GRPO训练")
    print("4. 用 grpo_test.jsonl 评估模型")


if __name__ == "__main__":
    main()
