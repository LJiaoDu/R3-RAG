#!/usr/bin/env python3
"""
R3-RAG数据生成（参考原论文）
- 8000条推理链（带真实检索）
- 5000条GRPO参考答案
- 2000条测试集答案
"""

import openai
import json
import re
from pathlib import Path
from tqdm import tqdm
import os
import random
from typing import List, Dict, Tuple
import numpy as np

# 假设你有一个检索系统（FAISS）
try:
    from retrieval_system import TUMCourseRetriever  # 你需要实现这个
except:
    print("⚠️  检索系统未导入，使用模拟模式")

# ===================== 配置 =====================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

MODEL = "gpt-4o-mini"

# 数据量配置
NUM_SFT = 8000       # 推理链
NUM_GRPO = 5000      # GRPO训练
NUM_TEST = 2000      # 测试集

# 原论文参数
NUM_ATTEMPTS_PER_QUESTION = 7  # 每个问题尝试7次
NUM_SEARCH_PER_ATTEMPT = 7     # 每次最多7步检索
NUM_PASSAGES_PER_RETRIEVAL = 3 # 每次检索返回3个文档
TEMPERATURE_FIRST = 0.0        # 第一次temperature
TEMPERATURE_REST = 0.9         # 后续temperature

# ===================== 检索系统 =====================

class MockRetriever:
    """模拟检索系统（实际应该用FAISS）"""

    def __init__(self):
        # 模拟的TUM课程知识库
        self.knowledge_base = {
            "machine learning prerequisites": [
                "Machine Learning (IN2064) requires: Linear Algebra, Probability Theory, and Python programming.",
                "Prerequisites must be completed with passing grade before enrollment.",
                "Recommended preparation: Introduction to AI."
            ],
            "linear algebra prerequisites": [
                "Linear Algebra for Informatics (MA1001) has no prerequisites.",
                "Typically taken in the first semester.",
                "Worth 8 ECTS credits."
            ],
            "probability theory prerequisites": [
                "Probability Theory (MA2009) requires Linear Algebra as prerequisite.",
                "Typically taken in the second semester.",
                "Worth 6 ECTS credits."
            ],
            "computer vision ects": [
                "Computer Vision (IN2128) is worth 6 ECTS credits.",
                "Offered in Winter Semester.",
                "Taught by Professor XYZ."
            ]
        }

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        真实实现应该：
        1. 用embedding model将query转为向量
        2. 在FAISS索引中检索
        3. 返回top_k个最相关文档
        """
        query_lower = query.lower()

        # 简单的关键词匹配（仅用于演示）
        results = []
        for key, docs in self.knowledge_base.items():
            if any(word in query_lower for word in key.split()):
                results.extend(docs)

        # 如果没找到，返回通用回复
        if not results:
            results = ["No specific information found in the database."]

        return results[:top_k]


# ===================== Prompt模板 =====================

SYSTEM_PROMPT_REASONING = """You are an expert assistant for TUM (Technical University of Munich) course consultation.

You have access to a TUM course database through a Search function.

Answer questions using step-by-step reasoning:

Thought: [Your reasoning about what to do next]
Action: Search[query] OR Finish[answer]
Observation: [Will be provided after Search]

Rules:
- Use Search[query] when you need specific course information
- After Search, you will receive relevant documents
- Show clear reasoning at each step
- Use Finish[answer] when you have the complete answer
- Be specific about course codes (e.g., IN2064, MA1001)

Example:

Question: What prerequisites do I need for Machine Learning?

Thought: I need to find the official prerequisites for Machine Learning at TUM.
Action: Search[Machine Learning IN2064 prerequisites requirements]
Observation: [Documents will appear here]

[Continue until you have enough information to answer]
"""

SYSTEM_PROMPT_DIRECT = """You are an expert assistant for TUM course consultation.

Provide a direct, accurate, and complete answer to the question.
Do NOT show reasoning steps or intermediate thoughts.
"""

# ===================== 推理链生成（SFT数据）=====================

def generate_reasoning_chain(
    question: str,
    retriever: MockRetriever,
    temperature: float = 0.7,
    max_steps: int = 7
) -> Tuple[bool, Dict]:
    """
    生成单个推理链（带真实检索交互）

    返回: (是否成功, 数据字典)
    """

    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT_REASONING},
        {"role": "user", "content": f"Question: {question}"}
    ]

    reasoning_chain = []
    current_step = 0

    while current_step < max_steps:
        try:
            # GPT-4生成推理
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=conversation_history,
                temperature=temperature,
                max_tokens=500
            )

            assistant_message = response.choices[0].message.content
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            # 解析Action
            if "Action: Search[" in assistant_message:
                # 提取查询
                match = re.search(r'Action: Search\[(.*?)\]', assistant_message)
                if match:
                    query = match.group(1)

                    # 真实检索！
                    search_results = retriever.search(query, top_k=NUM_PASSAGES_PER_RETRIEVAL)

                    # 格式化检索结果
                    observation = "\n".join([
                        f"[{i+1}] {doc}"
                        for i, doc in enumerate(search_results)
                    ])

                    # 保存这一步
                    reasoning_chain.append({
                        "step": current_step + 1,
                        "thought": extract_thought(assistant_message),
                        "action": f"Search[{query}]",
                        "observation": observation
                    })

                    # 将检索结果反馈给GPT-4
                    conversation_history.append({
                        "role": "user",
                        "content": f"Observation: {observation}\n\nContinue reasoning..."
                    })

                    current_step += 1

            elif "Action: Finish[" in assistant_message:
                # 提取最终答案
                match = re.search(r'Action: Finish\[(.*?)\]', assistant_message, re.DOTALL)
                if match:
                    final_answer = match.group(1).strip()

                    reasoning_chain.append({
                        "step": current_step + 1,
                        "thought": extract_thought(assistant_message),
                        "action": "Finish",
                        "answer": final_answer
                    })

                    # 成功生成完整轨迹
                    return True, {
                        "question": question,
                        "reasoning_chain": reasoning_chain,
                        "answer": final_answer,
                        "num_steps": len(reasoning_chain),
                        "temperature": temperature
                    }

            else:
                # 格式错误，重新提示
                conversation_history.append({
                    "role": "user",
                    "content": "Please use the format: Thought: ... Action: Search[...] OR Action: Finish[...]"
                })

        except Exception as e:
            print(f"❌ Error: {e}")
            return False, {}

    # 超过最大步数
    return False, {"error": "Max steps exceeded"}


def extract_thought(text: str) -> str:
    """提取Thought部分"""
    match = re.search(r'Thought: (.*?)(?=Action:|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def generate_sft_data(questions: List[str], retriever: MockRetriever, num_target: int):
    """
    生成SFT训练数据（推理链）

    原论文策略：
    - 每个问题尝试7次
    - 第1次 temperature=0.0
    - 第2-7次 temperature=0.9
    - 保留成功的轨迹
    """

    print(f"\n🔄 生成SFT数据（目标{num_target}条）...")

    sft_data = []
    question_idx = 0
    pbar = tqdm(total=num_target, desc="生成推理链")

    while len(sft_data) < num_target and question_idx < len(questions):
        question = questions[question_idx]

        # 尝试7次
        for attempt in range(NUM_ATTEMPTS_PER_QUESTION):
            temp = TEMPERATURE_FIRST if attempt == 0 else TEMPERATURE_REST

            success, data = generate_reasoning_chain(
                question,
                retriever,
                temperature=temp,
                max_steps=NUM_SEARCH_PER_ATTEMPT
            )

            if success:
                sft_data.append(data)
                pbar.update(1)

                # 达到目标数量
                if len(sft_data) >= num_target:
                    break

        question_idx += 1

    pbar.close()

    print(f"✅ 生成了 {len(sft_data)} 条SFT数据")

    # 保存
    with open('sft_training_data.jsonl', 'w', encoding='utf-8') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return sft_data


# ===================== 参考答案生成（GRPO+TEST）=====================

def generate_reference_answer(question: str, temperature: float = 0.7) -> str:
    """生成参考答案（无推理链）"""

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
                {"role": "user", "content": question}
            ],
            temperature=temperature,
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def generate_reference_data(questions: List[str], num_grpo: int, num_test: int):
    """生成GRPO训练数据和测试数据"""

    print(f"\n🎯 生成参考答案数据...")
    print(f"   - GRPO训练: {num_grpo}")
    print(f"   - 测试集: {num_test}")

    total = num_grpo + num_test
    reference_data = []

    for question in tqdm(questions[:total], desc="生成参考答案"):
        answer = generate_reference_answer(question, temperature=0.7)

        if answer:
            reference_data.append({
                "question": question,
                "reference_answer": answer
            })

    # 分割GRPO和TEST
    random.shuffle(reference_data)
    grpo_data = reference_data[:num_grpo]
    test_data = reference_data[num_grpo:num_grpo + num_test]

    # 保存
    with open('grpo_train.jsonl', 'w', encoding='utf-8') as f:
        for item in grpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open('test_data.jsonl', 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ GRPO训练数据: {len(grpo_data)} 条")
    print(f"✅ 测试数据: {len(test_data)} 条")

    return grpo_data, test_data


# ===================== 主函数 =====================

def main():
    print("="*60)
    print("R3-RAG数据生成（原论文方法）")
    print("="*60)

    # 1. 初始化检索系统
    print("\n📂 初始化检索系统...")
    retriever = MockRetriever()  # 替换为真实的FAISS检索器
    print("✅ 检索系统就绪")

    # 2. 加载问题
    print("\n📂 加载问题...")
    # 假设你有questions.jsonl
    if not Path("questions.jsonl").exists():
        print("❌ questions.jsonl不存在，创建示例...")
        create_sample_questions()

    questions = []
    with open("questions.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            questions.append(data.get('question', data.get('text')))

    print(f"✅ 加载了 {len(questions)} 个问题")

    # 随机打乱
    random.shuffle(questions)

    # 3. 分配问题
    sft_questions = questions[:NUM_SFT]
    ref_questions = questions[NUM_SFT:NUM_SFT + NUM_GRPO + NUM_TEST]

    print(f"\n📊 数据分配:")
    print(f"   - SFT（推理链）: {len(sft_questions)} 个问题")
    print(f"   - GRPO+TEST（参考答案）: {len(ref_questions)} 个问题")

    # 4. 生成SFT数据（推理链）
    sft_data = generate_sft_data(
        sft_questions,
        retriever,
        num_target=NUM_SFT
    )

    # 5. 生成参考答案数据
    grpo_data, test_data = generate_reference_data(
        ref_questions,
        num_grpo=NUM_GRPO,
        num_test=NUM_TEST
    )

    # 6. 统计
    print("\n" + "="*60)
    print("📊 最终统计")
    print("="*60)
    print(f"✅ SFT训练数据: {len(sft_data)} 条（带推理链）")
    print(f"✅ GRPO训练数据: {len(grpo_data)} 条（参考答案）")
    print(f"✅ 测试数据: {len(test_data)} 条（参考答案）")

    # 统计SFT数据的推理步数
    steps = [d['num_steps'] for d in sft_data if 'num_steps' in d]
    if steps:
        print(f"\n📈 SFT数据统计:")
        print(f"   平均推理步数: {np.mean(steps):.1f}")
        print(f"   最少步数: {min(steps)}")
        print(f"   最多步数: {max(steps)}")

    print("\n🎉 数据生成完成！")
    print("\n文件:")
    print("   - sft_training_data.jsonl")
    print("   - grpo_train.jsonl")
    print("   - test_data.jsonl")


def create_sample_questions():
    """创建示例问题（如果没有questions.jsonl）"""
    sample_questions = [
        {"id": 1, "question": "What courses do I need before taking Machine Learning?"},
        {"id": 2, "question": "How many ECTS is Computer Vision?"},
        {"id": 3, "question": "What's the prerequisite chain for Advanced Deep Learning?"},
        # 添加更多...
    ]

    with open("questions.jsonl", 'w', encoding='utf-8') as f:
        for q in sample_questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    print("✅ 创建了示例questions.jsonl")


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("❌ 请设置OPENAI_API_KEY环境变量")
        exit(1)

    main()
