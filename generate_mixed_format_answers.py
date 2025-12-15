#!/usr/bin/env python3
"""
混合模式生成：10000条推理链 + 5000条直接答案
支持智能分类或手动分配
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
OUTPUT_FILE = "questions_with_mixed_answers.jsonl"
CHECKPOINT_FILE = "checkpoint_mixed.json"

# 数量配置
TOTAL_QUESTIONS = 15000
NUM_WITH_REASONING = 10000  # 生成推理链的数量
NUM_DIRECT_ANSWER = 5000    # 只生成答案的数量

# 批处理配置
BATCH_SIZE = 50
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.0

# ===================== Prompt模板 =====================

# 模式1：生成推理链（Reasoning Chain）
SYSTEM_PROMPT_REASONING = """You are an expert assistant for TUM (Technical University of Munich) course consultation.

Answer the question using step-by-step reasoning with the following format:

Thought: [Your reasoning about what to do next]
Action: Search[query] OR Finish[answer]
Observation: [Simulated search results OR final answer]

Important:
- Use Search[query] when you need specific TUM course information (prerequisites, ECTS, schedules)
- Use Finish[answer] when you have the complete answer
- Show your reasoning process clearly

Example:

Question: "What courses do I need before Machine Learning?"

Thought: I need to find the prerequisites for Machine Learning course at TUM.
Action: Search[Machine Learning IN2064 prerequisites TUM]
Observation: Machine Learning (IN2064) requires: Linear Algebra, Probability Theory, and Python programming experience.

Thought: Let me check if these prerequisites have their own requirements.
Action: Search[Linear Algebra prerequisites TUM Informatics]
Observation: Linear Algebra (MA1001) is a first-semester course with no prerequisites.

Thought: Now check Probability Theory.
Action: Search[Probability Theory MA2009 prerequisites]
Observation: Probability Theory requires Linear Algebra.

Thought: I have complete information now.
Action: Finish[To take Machine Learning (IN2064) at TUM, you need to complete:
1. Linear Algebra (MA1001) - no prerequisites, typically 1st semester
2. Probability Theory (MA2009) - requires Linear Algebra
3. Python programming experience

So the sequence is: Linear Algebra → Probability Theory → Machine Learning. This typically takes 2-3 semesters.]
"""

# 模式2：直接答案（Direct Answer）
SYSTEM_PROMPT_DIRECT = """You are an expert assistant for TUM (Technical University of Munich) course consultation.

Answer the question directly and concisely. Provide accurate information without showing intermediate reasoning steps.

Keep your answer clear, factual, and to the point."""

# ===================== 问题分类 =====================

def classify_question_complexity(question: str) -> str:
    """
    判断问题是否需要推理链

    需要推理链的特征：
    - 包含"path", "plan", "how to", "step"等词
    - 包含"before", "after", "then"等顺序词
    - 包含"difference", "compare", "vs"等对比词
    - 包含"if", "whether", "can I"等条件词
    - 问题较长（>50字符）

    直接答案的特征：
    - 询问具体事实（ECTS, 时间, 代码）
    - 是/否问题
    - 定义性问题（what is X）
    - 问题较短
    """
    question_lower = question.lower()

    # 需要推理链的关键词
    reasoning_keywords = [
        'path', 'plan', 'how to', 'step', 'sequence', 'order',
        'before', 'after', 'then', 'first', 'next',
        'difference', 'compare', 'vs', 'versus', 'between',
        'if', 'whether', 'can i', 'should i', 'which one',
        'route', 'way to', 'strategy', 'approach'
    ]

    # 德语关键词
    reasoning_keywords_de = [
        'weg', 'pfad', 'reihenfolge', 'zuerst', 'dann',
        'unterschied', 'vergleich', 'oder', 'welche',
        'sollte ich', 'kann ich', 'wie kann'
    ]

    # 直接答案的关键词
    direct_keywords = [
        'how many ects', 'when is', 'what time', 'course code',
        'credit', 'semester', 'who teaches', 'what is',
        'wie viele ects', 'wann', 'wer', 'was ist'
    ]

    # 检查直接答案关键词（优先级高）
    for keyword in direct_keywords:
        if keyword in question_lower:
            return 'direct'

    # 检查推理链关键词
    for keyword in reasoning_keywords + reasoning_keywords_de:
        if keyword in question_lower:
            return 'reasoning'

    # 基于长度判断
    if len(question) < 40:
        return 'direct'
    elif len(question) > 80:
        return 'reasoning'

    # 默认：中等复杂度 → 随机分配
    return random.choice(['reasoning', 'direct'])


def assign_question_modes(questions: List[Dict],
                          num_reasoning: int,
                          num_direct: int,
                          strategy: str = 'smart') -> List[Dict]:
    """
    为问题分配生成模式

    strategy:
    - 'smart': 智能分类（基于问题复杂度）
    - 'random': 随机分配
    - 'first': 前N个用推理链，后M个直接答案
    """
    total = len(questions)

    if strategy == 'smart':
        # 智能分类
        print("🧠 智能分类问题...")

        # 第一步：根据复杂度分类
        for q in tqdm(questions, desc="分类"):
            complexity = classify_question_complexity(
                q.get('question', q.get('text', ''))
            )
            q['suggested_mode'] = complexity

        # 统计
        suggested_reasoning = sum(1 for q in questions if q['suggested_mode'] == 'reasoning')
        suggested_direct = sum(1 for q in questions if q['suggested_mode'] == 'direct')

        print(f"   智能分类结果：")
        print(f"   - 推荐推理链: {suggested_reasoning}")
        print(f"   - 推荐直接答案: {suggested_direct}")

        # 第二步：调整到目标数量
        reasoning_questions = [q for q in questions if q['suggested_mode'] == 'reasoning']
        direct_questions = [q for q in questions if q['suggested_mode'] == 'direct']

        # 如果推理链数量不足，从直接答案中抽取
        if len(reasoning_questions) < num_reasoning:
            needed = num_reasoning - len(reasoning_questions)
            additional = random.sample(direct_questions, min(needed, len(direct_questions)))
            reasoning_questions.extend(additional)
            for q in additional:
                direct_questions.remove(q)

        # 如果推理链数量过多，移动到直接答案
        elif len(reasoning_questions) > num_reasoning:
            excess = len(reasoning_questions) - num_reasoning
            moved = random.sample(reasoning_questions, excess)
            direct_questions.extend(moved)
            for q in moved:
                reasoning_questions.remove(q)

        # 分配模式
        for q in reasoning_questions:
            q['generation_mode'] = 'reasoning'
        for q in direct_questions[:num_direct]:
            q['generation_mode'] = 'direct'

    elif strategy == 'random':
        # 随机分配
        print("🎲 随机分配模式...")
        indices = list(range(total))
        random.shuffle(indices)

        reasoning_indices = set(indices[:num_reasoning])

        for i, q in enumerate(questions):
            q['generation_mode'] = 'reasoning' if i in reasoning_indices else 'direct'

    elif strategy == 'first':
        # 前N个推理链，后M个直接答案
        print("📋 按顺序分配模式...")
        for i, q in enumerate(questions):
            q['generation_mode'] = 'reasoning' if i < num_reasoning else 'direct'

    # 统计
    final_reasoning = sum(1 for q in questions if q.get('generation_mode') == 'reasoning')
    final_direct = sum(1 for q in questions if q.get('generation_mode') == 'direct')

    print(f"✅ 最终分配：")
    print(f"   - 推理链: {final_reasoning}")
    print(f"   - 直接答案: {final_direct}")

    return questions


# ===================== 生成函数 =====================

async def generate_answer_async(
    session: aiohttp.ClientSession,
    question_data: Dict,
    semaphore: asyncio.Semaphore
) -> Dict:
    """异步生成答案（根据mode选择prompt）"""
    async with semaphore:
        question_text = question_data.get('question', question_data.get('text'))
        mode = question_data.get('generation_mode', 'reasoning')

        # 选择prompt
        system_prompt = SYSTEM_PROMPT_REASONING if mode == 'reasoning' else SYSTEM_PROMPT_DIRECT

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
                    max_tokens=800 if mode == 'reasoning' else 300
                )

                answer = response.choices[0].message.content

                result = {
                    **question_data,
                    'answer': answer,
                    'answer_type': mode,
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
                        'answer_type': mode
                    }
                await asyncio.sleep(2 ** attempt)

        return {
            **question_data,
            'answer': None,
            'error': 'Max retries exceeded',
            'answer_type': mode
        }


async def process_batch(
    questions_batch: List[Dict],
    output_file: str,
    checkpoint_file: str,
    processed_ids: set,
    semaphore: asyncio.Semaphore
):
    """处理一批问题"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            generate_answer_async(session, q, semaphore)
            for q in questions_batch
        ]

        results = await asyncio.gather(*tasks)

        for result in results:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

            question_id = result.get('id', result.get('question'))
            processed_ids.add(question_id)

        # 保存checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({'processed_ids': list(processed_ids)}, f)


async def main_async(questions: List[Dict], output_file: str, checkpoint_file: str):
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

    with tqdm(total=len(remaining), desc="生成答案") as pbar:
        for i in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[i:i + BATCH_SIZE]

            await process_batch(
                batch,
                output_file,
                checkpoint_file,
                processed_ids,
                semaphore
            )

            pbar.update(len(batch))

            if i + BATCH_SIZE < len(remaining):
                await asyncio.sleep(RATE_LIMIT_DELAY)

    print(f"✅ 完成！输出文件: {output_file}")


# ===================== 成本估算 =====================

def estimate_mixed_cost(num_reasoning: int, num_direct: int, model: str = "gpt-4o-mini"):
    """估算混合模式成本"""
    prices = {
        "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    }

    price = prices.get(model, prices["gpt-4o-mini"])

    # 推理链：更多tokens
    reasoning_input = 150  # system prompt更长
    reasoning_output = 400  # 包含Thought/Action/Observation

    # 直接答案：较少tokens
    direct_input = 80
    direct_output = 150

    cost_reasoning = num_reasoning * (
        reasoning_input * price["input"] + reasoning_output * price["output"]
    )

    cost_direct = num_direct * (
        direct_input * price["input"] + direct_output * price["output"]
    )

    total_cost = cost_reasoning + cost_direct

    print(f"\n💰 成本估算 (混合模式 - {model}):")
    print(f"   推理链答案: {num_reasoning:,} 条")
    print(f"     - 输入: {num_reasoning * reasoning_input:,} tokens → ${num_reasoning * reasoning_input * price['input']:.2f}")
    print(f"     - 输出: {num_reasoning * reasoning_output:,} tokens → ${num_reasoning * reasoning_output * price['output']:.2f}")
    print(f"     - 小计: ${cost_reasoning:.2f}")
    print(f"\n   直接答案: {num_direct:,} 条")
    print(f"     - 输入: {num_direct * direct_input:,} tokens → ${num_direct * direct_input * price['input']:.2f}")
    print(f"     - 输出: {num_direct * direct_output:,} tokens → ${num_direct * direct_output * price['output']:.2f}")
    print(f"     - 小计: ${cost_direct:.2f}")
    print(f"\n   💵 总成本: ${total_cost:.2f}")

    # 对比全部生成推理链
    all_reasoning_cost = (num_reasoning + num_direct) * (
        reasoning_input * price["input"] + reasoning_output * price["output"]
    )
    savings = all_reasoning_cost - total_cost

    print(f"\n   📊 对比全部生成推理链:")
    print(f"     - 全推理链成本: ${all_reasoning_cost:.2f}")
    print(f"     - 混合模式成本: ${total_cost:.2f}")
    print(f"     - 💰 节省: ${savings:.2f} ({savings/all_reasoning_cost*100:.1f}%)")

    # 时间估算
    print(f"\n   ⏱️  预计时间: {(num_reasoning + num_direct) / BATCH_SIZE * RATE_LIMIT_DELAY / 3600:.1f} 小时")


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

    # 先估算成本
    estimate_mixed_cost(NUM_WITH_REASONING, NUM_DIRECT_ANSWER, MODEL)

    # 询问是否继续
    print("\n选择问题分配策略:")
    print("1. 智能分类（推荐）- 根据问题复杂度自动分配")
    print("2. 随机分配 - 完全随机")
    print("3. 按顺序分配 - 前10000个推理链，后5000个直接答案")

    choice = input("\n请选择 (1/2/3): ").strip()

    strategy_map = {'1': 'smart', '2': 'random', '3': 'first'}
    strategy = strategy_map.get(choice, 'smart')

    # 分配模式
    questions = assign_question_modes(
        questions,
        NUM_WITH_REASONING,
        NUM_DIRECT_ANSWER,
        strategy
    )

    # 确认继续
    print(f"\n是否继续生成？(y/n): ", end='')
    if input().strip().lower() != 'y':
        print("已取消")
        return

    # 开始生成
    asyncio.run(main_async(questions, OUTPUT_FILE, CHECKPOINT_FILE))

    # 统计结果
    print("\n📊 生成统计:")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]

    reasoning_count = sum(1 for r in results if r.get('answer_type') == 'reasoning')
    direct_count = sum(1 for r in results if r.get('answer_type') == 'direct')
    success_count = sum(1 for r in results if r.get('answer'))

    print(f"   总数: {len(results)}")
    print(f"   推理链: {reasoning_count}")
    print(f"   直接答案: {direct_count}")
    print(f"   成功: {success_count} ({success_count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
