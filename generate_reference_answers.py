#!/usr/bin/env python3
"""
批量使用OpenAI API生成参考答案
适用于15000个问题的大规模数据生成
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

# ===================== 配置 =====================

# OpenAI API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 从环境变量读取
openai.api_key = OPENAI_API_KEY

# 模型选择
MODEL = "gpt-4o-mini"  # 推荐：便宜且质量好
# MODEL = "gpt-4o"  # 更贵但质量更高
# MODEL = "gpt-3.5-turbo"  # 最便宜，但质量较低

# 文件路径
INPUT_FILE = "questions.jsonl"  # 输入：15000个问题
OUTPUT_FILE = "questions_with_answers.jsonl"  # 输出：问题+答案
CHECKPOINT_FILE = "checkpoint.json"  # 断点续传

# 批处理配置
BATCH_SIZE = 50  # 每批处理多少个（异步并发）
MAX_RETRIES = 3  # 失败重试次数
RATE_LIMIT_DELAY = 1.0  # API调用间隔（秒）

# Prompt模板（针对TUM课程咨询）
SYSTEM_PROMPT = """You are an expert assistant for TUM (Technical University of Munich) course consultation.
You help students with course planning, prerequisites, ECTS credits, and study path questions.

Answer the following question clearly and concisely. If you need specific course information that you don't have, indicate what information would be needed."""

# ===================== 核心函数 =====================

def load_questions(input_file: str) -> List[Dict]:
    """加载问题列表"""
    questions = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            questions.append(data)
    print(f"✅ 加载了 {len(questions)} 个问题")
    return questions


def load_checkpoint(checkpoint_file: str) -> set:
    """加载已处理的问题ID（断点续传）"""
    if not Path(checkpoint_file).exists():
        return set()

    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)

    processed_ids = set(checkpoint.get('processed_ids', []))
    print(f"📍 从checkpoint恢复：已处理 {len(processed_ids)} 个问题")
    return processed_ids


def save_checkpoint(checkpoint_file: str, processed_ids: set):
    """保存checkpoint"""
    with open(checkpoint_file, 'w') as f:
        json.dump({'processed_ids': list(processed_ids)}, f)


def append_result(output_file: str, result: Dict):
    """追加结果到输出文件"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


async def generate_answer_async(
    session: aiohttp.ClientSession,
    question_data: Dict,
    semaphore: asyncio.Semaphore
) -> Dict:
    """异步生成单个答案"""
    async with semaphore:  # 限制并发数
        question_id = question_data.get('id', question_data.get('question'))
        question_text = question_data.get('question', question_data.get('text'))

        # 构建API请求
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_text}
        ]

        # 重试逻辑
        for attempt in range(MAX_RETRIES):
            try:
                # 使用OpenAI Python SDK（异步）
                response = await asyncio.to_thread(
                    openai.ChatCompletion.create,
                    model=MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )

                answer = response.choices[0].message.content

                # 返回结果
                result = {
                    **question_data,  # 保留原始数据
                    'reference_answer': answer,
                    'model': MODEL,
                    'timestamp': time.time()
                }

                return result

            except openai.error.RateLimitError as e:
                wait_time = (2 ** attempt) * RATE_LIMIT_DELAY
                print(f"⚠️  Rate limit reached, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)

            except openai.error.APIError as e:
                print(f"⚠️  API Error for question {question_id}: {e}")
                if attempt == MAX_RETRIES - 1:
                    return {
                        **question_data,
                        'reference_answer': None,
                        'error': str(e)
                    }
                await asyncio.sleep(2 ** attempt)

            except Exception as e:
                print(f"❌ Unexpected error for question {question_id}: {e}")
                return {
                    **question_data,
                    'reference_answer': None,
                    'error': str(e)
                }

        # 所有重试失败
        return {
            **question_data,
            'reference_answer': None,
            'error': 'Max retries exceeded'
        }


async def process_batch(
    questions_batch: List[Dict],
    output_file: str,
    checkpoint_file: str,
    processed_ids: set,
    semaphore: asyncio.Semaphore
):
    """异步处理一批问题"""
    async with aiohttp.ClientSession() as session:
        tasks = [
            generate_answer_async(session, q, semaphore)
            for q in questions_batch
        ]

        results = await asyncio.gather(*tasks)

        # 保存结果
        for result in results:
            append_result(output_file, result)
            question_id = result.get('id', result.get('question'))
            processed_ids.add(question_id)

        # 保存checkpoint
        save_checkpoint(checkpoint_file, processed_ids)


async def main_async(
    questions: List[Dict],
    output_file: str,
    checkpoint_file: str
):
    """主异步函数"""
    # 加载checkpoint
    processed_ids = load_checkpoint(checkpoint_file)

    # 过滤未处理的问题
    remaining_questions = [
        q for q in questions
        if q.get('id', q.get('question')) not in processed_ids
    ]

    print(f"🚀 需要处理 {len(remaining_questions)} 个问题")

    if len(remaining_questions) == 0:
        print("✅ 所有问题已处理完毕！")
        return

    # 信号量：控制最大并发数
    semaphore = asyncio.Semaphore(BATCH_SIZE)

    # 分批处理
    total_batches = (len(remaining_questions) + BATCH_SIZE - 1) // BATCH_SIZE

    with tqdm(total=len(remaining_questions), desc="生成答案") as pbar:
        for i in range(0, len(remaining_questions), BATCH_SIZE):
            batch = remaining_questions[i:i + BATCH_SIZE]

            await process_batch(
                batch,
                output_file,
                checkpoint_file,
                processed_ids,
                semaphore
            )

            pbar.update(len(batch))

            # 批次间延迟（避免rate limit）
            if i + BATCH_SIZE < len(remaining_questions):
                await asyncio.sleep(RATE_LIMIT_DELAY)

    print(f"✅ 完成！已生成 {len(remaining_questions)} 个答案")
    print(f"📁 输出文件: {output_file}")


# ===================== 同步版本（简单但慢）=====================

def generate_answer_sync(question_data: Dict) -> Dict:
    """同步生成单个答案（简单但慢）"""
    question_text = question_data.get('question', question_data.get('text'))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text}
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )

            answer = response.choices[0].message.content

            return {
                **question_data,
                'reference_answer': answer,
                'model': MODEL
            }

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"❌ Failed after {MAX_RETRIES} attempts: {e}")
                return {
                    **question_data,
                    'reference_answer': None,
                    'error': str(e)
                }
            time.sleep(2 ** attempt)


def main_sync(questions: List[Dict], output_file: str):
    """同步主函数（简单版本）"""
    print("⚠️  使用同步模式（较慢，建议用异步模式）")

    for i, question in enumerate(tqdm(questions, desc="生成答案")):
        result = generate_answer_sync(question)
        append_result(output_file, result)

        # API调用间隔
        if i < len(questions) - 1:
            time.sleep(RATE_LIMIT_DELAY)

    print(f"✅ 完成！输出文件: {output_file}")


# ===================== 入口函数 =====================

def main():
    """主函数"""
    # 检查API key
    if not OPENAI_API_KEY:
        print("❌ 错误: 请设置环境变量 OPENAI_API_KEY")
        print("   export OPENAI_API_KEY='your-api-key'")
        return

    # 检查输入文件
    if not Path(INPUT_FILE).exists():
        print(f"❌ 错误: 输入文件不存在: {INPUT_FILE}")
        print("   请准备格式为 JSONL 的问题文件，每行一个JSON对象")
        print("   示例: {'id': 1, 'question': 'What is ML?'}")
        return

    # 加载问题
    questions = load_questions(INPUT_FILE)

    # 询问用户选择模式
    print("\n选择运行模式:")
    print("1. 异步模式（推荐，快速）")
    print("2. 同步模式（简单，较慢）")
    choice = input("请选择 (1/2): ").strip()

    if choice == '1':
        # 异步模式
        asyncio.run(main_async(questions, OUTPUT_FILE, CHECKPOINT_FILE))
    else:
        # 同步模式
        main_sync(questions, OUTPUT_FILE)


# ===================== 成本估算 =====================

def estimate_cost(num_questions: int, model: str = "gpt-4o-mini"):
    """估算API成本"""
    # OpenAI定价（2024年1月）
    prices = {
        "gpt-4o-mini": {
            "input": 0.150 / 1_000_000,   # $0.150 per 1M tokens
            "output": 0.600 / 1_000_000   # $0.600 per 1M tokens
        },
        "gpt-4o": {
            "input": 2.50 / 1_000_000,
            "output": 10.00 / 1_000_000
        },
        "gpt-3.5-turbo": {
            "input": 0.50 / 1_000_000,
            "output": 1.50 / 1_000_000
        }
    }

    # 假设平均每个问题：
    # - Input: 100 tokens (system prompt + question)
    # - Output: 200 tokens (answer)
    avg_input_tokens = 100
    avg_output_tokens = 200

    price = prices.get(model, prices["gpt-4o-mini"])

    total_input_tokens = num_questions * avg_input_tokens
    total_output_tokens = num_questions * avg_output_tokens

    cost_input = total_input_tokens * price["input"]
    cost_output = total_output_tokens * price["output"]
    total_cost = cost_input + cost_output

    print(f"\n💰 成本估算 (使用 {model}):")
    print(f"   问题数量: {num_questions:,}")
    print(f"   预计输入tokens: {total_input_tokens:,}")
    print(f"   预计输出tokens: {total_output_tokens:,}")
    print(f"   输入成本: ${cost_input:.2f}")
    print(f"   输出成本: ${cost_output:.2f}")
    print(f"   总成本: ${total_cost:.2f}")
    print(f"\n   ⏱️  预计时间: {num_questions * RATE_LIMIT_DELAY / 3600:.1f} 小时 (同步模式)")
    print(f"   ⏱️  预计时间: {num_questions * RATE_LIMIT_DELAY / BATCH_SIZE / 3600:.1f} 小时 (异步模式，批次={BATCH_SIZE})")


if __name__ == "__main__":
    # 先估算成本
    estimate_cost(15000, MODEL)

    # 询问是否继续
    print("\n是否继续？(y/n): ", end='')
    if input().strip().lower() == 'y':
        main()
    else:
        print("已取消")
