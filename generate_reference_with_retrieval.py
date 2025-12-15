#!/usr/bin/env python3
"""
生成Reference Answer（配合检索器）
专门用于生成高质量的参考答案，作为后续推理链验证的Golden Answer

核心改进：
1. 结合检索器（OpenAI + Retriever）
2. 优化Prompt设计，提高答案质量
3. 多层质量控制
"""

import openai
import json
import os
from typing import List, Dict, Optional

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# ===================== Prompt设计（核心） =====================

# 【关键】Prompt模板：高质量Reference Answer
REFERENCE_ANSWER_SYSTEM_PROMPT = """You are an expert academic advisor at Technical University of Munich (TUM).

Your role: Generate REFERENCE ANSWERS that will serve as GOLD STANDARD for evaluating other AI-generated answers.

Critical requirements for reference answers:
1. ACCURACY: Must be 100% correct based on retrieved course documents
2. SPECIFICITY: Include exact details (course codes like IN2064, ECTS credits, prerequisites)
3. COMPLETENESS: Fully answer the question, including all relevant aspects
4. CLARITY: Use clear, structured language that students can easily understand
5. VERIFIABILITY: Every claim must come from the retrieved documents

Quality standards:
- If information is not in the documents, say "This information is not available in the course catalog"
- Never speculate or add information beyond the documents
- For multi-hop questions, connect information from multiple documents logically
- Use specific course codes, not just course names
- Include ECTS credits when relevant
- Mention prerequisites explicitly when they exist"""

REFERENCE_ANSWER_USER_PROMPT_TEMPLATE = """Student Question:
{question}

Retrieved Course Documents:
{retrieved_docs}

Instructions:
1. Carefully read ALL retrieved documents
2. Identify the relevant information for this question
3. For multi-hop questions, synthesize information from multiple documents
4. Construct a complete, accurate answer
5. Include specific details: course codes, ECTS, prerequisites

Generate a reference answer that will serve as the gold standard for evaluating other answers.

Reference Answer:"""


# 【改进版】更严格的Prompt（适用于关键数据）
STRICT_REFERENCE_ANSWER_USER_PROMPT_TEMPLATE = """Student Question:
{question}

Retrieved Course Documents:
{retrieved_docs}

⚠️ CRITICAL: This answer will be used as GOLD STANDARD for training data quality control.

Step-by-step process:
1. Extract Facts: List all relevant facts from the documents
2. Verify Accuracy: Ensure each fact is directly from a document
3. Multi-hop Reasoning: If the question requires multiple documents, connect them logically
4. Construct Answer: Build a complete answer with specific details
5. Quality Check: Verify the answer is clear, specific, and complete

Answer Format Requirements:
[Main Answer]
- Start with a direct answer to the question
- Include specific course codes (e.g., IN2064, MA1001)
- Include ECTS credits if relevant
- Include prerequisites if they exist

[Supporting Details]
- Add any important constraints or conditions
- Note any special requirements (e.g., "Must pass Linear Algebra before taking Machine Learning")

Generate the reference answer:"""


# ===================== 检索文档格式化 =====================

def format_retrieved_documents(documents: List[Dict]) -> str:
    """
    格式化检索到的文档，使其更易于GPT理解

    Args:
        documents: [
            {
                'doc_id': 'doc_123',
                'course_code': 'IN2064',
                'course_name': 'Machine Learning',
                'content': '...',
                'metadata': {...}
            }
        ]

    Returns:
        格式化的文档字符串
    """

    if not documents:
        return "No documents retrieved."

    formatted = ""
    for i, doc in enumerate(documents, 1):
        formatted += f"--- Document {i} ---\n"

        # 课程代码和名称（如果有）
        if 'course_code' in doc:
            formatted += f"Course Code: {doc['course_code']}\n"
        if 'course_name' in doc:
            formatted += f"Course Name: {doc['course_name']}\n"

        # 文档ID
        if 'doc_id' in doc:
            formatted += f"Document ID: {doc['doc_id']}\n"

        # 内容
        content = doc.get('content', doc.get('text', 'N/A'))
        formatted += f"Content:\n{content}\n"

        # Metadata（可选）
        if 'metadata' in doc and doc['metadata']:
            formatted += f"Metadata: {json.dumps(doc['metadata'])}\n"

        # 相关性分数（可选）
        if 'score' in doc:
            formatted += f"Relevance Score: {doc['score']:.3f}\n"

        formatted += "\n"

    return formatted


# ===================== 生成Reference Answer =====================

def generate_reference_answer(
    question: str,
    retrieved_docs: List[Dict],
    model: str = "gpt-4",
    temperature: float = 0.0,  # 推荐0.0，确保稳定性
    use_strict_prompt: bool = True,
    max_tokens: int = 800
) -> Dict:
    """
    生成Reference Answer（配合检索器）

    Args:
        question: 学生问题
        retrieved_docs: 检索到的文档列表
        model: OpenAI模型
        temperature: 生成温度（推荐0.0确保一致性）
        use_strict_prompt: 是否使用严格版prompt
        max_tokens: 最大token数

    Returns:
        {
            'question': str,
            'reference_answer': str,
            'retrieved_docs': List[Dict],
            'model': str,
            'temperature': float,
            'tokens_used': int,
            'success': bool
        }
    """

    print(f"\n{'='*60}")
    print(f"生成Reference Answer")
    print(f"{'='*60}")
    print(f"问题: {question}")
    print(f"检索文档数: {len(retrieved_docs)}")

    # Step 1: 格式化文档
    formatted_docs = format_retrieved_documents(retrieved_docs)

    # Step 2: 选择Prompt模板
    if use_strict_prompt:
        user_prompt = STRICT_REFERENCE_ANSWER_USER_PROMPT_TEMPLATE.format(
            question=question,
            retrieved_docs=formatted_docs
        )
        print(f"使用: 严格版Prompt")
    else:
        user_prompt = REFERENCE_ANSWER_USER_PROMPT_TEMPLATE.format(
            question=question,
            retrieved_docs=formatted_docs
        )
        print(f"使用: 标准Prompt")

    # Step 3: 调用OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": REFERENCE_ANSWER_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        reference_answer = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens

        print(f"✅ 生成成功")
        print(f"Tokens使用: {tokens_used}")
        print(f"\n--- Reference Answer ---")
        print(reference_answer)
        print(f"{'='*60}\n")

        return {
            'question': question,
            'reference_answer': reference_answer,
            'retrieved_docs': retrieved_docs,
            'model': model,
            'temperature': temperature,
            'tokens_used': tokens_used,
            'success': True
        }

    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return {
            'question': question,
            'reference_answer': None,
            'retrieved_docs': retrieved_docs,
            'model': model,
            'temperature': temperature,
            'error': str(e),
            'success': False
        }


# ===================== 质量检查 =====================

def quality_check(result: Dict) -> Dict:
    """
    对生成的Reference Answer进行质量检查

    检查维度：
    1. 长度合理性（不能太短或太长）
    2. 包含课程代码（课程咨询必须）
    3. 包含具体数字（ECTS等）
    4. 基于检索文档（不能凭空捏造）
    5. 无模糊表述（必须明确）
    6. 语言清晰度
    """

    print(f"\n{'='*60}")
    print(f"质量检查")
    print(f"{'='*60}")

    answer = result.get('reference_answer', '')
    docs = result.get('retrieved_docs', [])

    if not answer:
        print(f"❌ 没有生成答案")
        return {**result, 'quality_score': 0.0, 'quality_checks': {}}

    import re

    checks = {}

    # 1. 长度检查
    words = answer.split()
    word_count = len(words)
    checks['length_ok'] = 30 <= word_count <= 300
    status = '✅' if checks['length_ok'] else '⚠️'
    print(f"{status} 长度: {word_count} words (建议30-300)")

    # 2. 课程代码检查
    course_codes = re.findall(r'\b[A-Z]{2,3}\d{4}\b', answer)
    checks['has_course_codes'] = len(course_codes) > 0
    status = '✅' if checks['has_course_codes'] else '⚠️'
    print(f"{status} 课程代码: {course_codes if course_codes else '无'}")

    # 3. 具体数字（ECTS等）
    ects_mentions = re.findall(r'\d+\s*ECTS', answer, re.IGNORECASE)
    checks['has_ects'] = len(ects_mentions) > 0
    status = '✅' if checks['has_ects'] else '⚠️'
    print(f"{status} ECTS信息: {ects_mentions if ects_mentions else '无'}")

    # 4. 基于文档（检查答案中的课程是否在文档中）
    doc_content = ' '.join([doc.get('content', '') for doc in docs])
    doc_codes = re.findall(r'\b[A-Z]{2,3}\d{4}\b', doc_content)

    grounded = any(code in doc_codes for code in course_codes) if course_codes else False
    checks['grounded_in_docs'] = grounded or len(course_codes) == 0
    status = '✅' if checks['grounded_in_docs'] else '❌'
    print(f"{status} 基于文档: {'是' if grounded else '否'}")

    # 5. 无模糊表述
    vague_phrases = ['maybe', 'probably', 'might', 'perhaps', 'i think', 'not sure', 'unclear']
    has_vague = any(phrase in answer.lower() for phrase in vague_phrases)
    checks['no_vague_language'] = not has_vague
    status = '✅' if checks['no_vague_language'] else '⚠️'
    print(f"{status} 明确性: {'明确' if not has_vague else '含有模糊表述'}")

    # 6. 包含必要信息（prerequisites, requirements等关键词）
    info_keywords = ['prerequisite', 'requirement', 'before', 'after', 'must', 'need']
    has_info_keywords = any(kw in answer.lower() for kw in info_keywords)
    checks['has_structural_info'] = has_info_keywords
    status = '✅' if has_info_keywords else 'ℹ️'
    print(f"{status} 结构化信息: {'包含' if has_info_keywords else '不包含前置条件等关键词'}")

    # 7. 不包含不应该出现的内容
    bad_phrases = ['i don\'t know', 'i cannot answer', 'no information', 'not in the documents']
    has_bad = any(phrase in answer.lower() for phrase in bad_phrases)
    checks['no_inability_statements'] = not has_bad
    status = '✅' if not has_bad else '⚠️'
    print(f"{status} 答案完整性: {'完整' if not has_bad else '可能信息不足'}")

    # 计算总分
    # 权重：基于文档(0.3) + 其他(0.7/6)
    grounded_weight = 0.3
    other_weight = 0.7 / 6

    score = (
        checks['grounded_in_docs'] * grounded_weight +
        checks['length_ok'] * other_weight +
        checks['has_course_codes'] * other_weight +
        checks['has_ects'] * other_weight +
        checks['no_vague_language'] * other_weight +
        checks['has_structural_info'] * other_weight +
        checks['no_inability_statements'] * other_weight
    )

    print(f"\n{'='*60}")
    print(f"总体质量评分: {score:.1%}")

    if score >= 0.8:
        print(f"✅ 质量优秀，可以作为Golden Answer")
    elif score >= 0.6:
        print(f"⚠️  质量一般，建议优化检索或重新生成")
    else:
        print(f"❌ 质量不足，必须重新生成")

    print(f"{'='*60}\n")

    return {
        **result,
        'quality_score': score,
        'quality_checks': checks
    }


# ===================== 完整流程示例 =====================

def example_with_retrieval():
    """
    完整示例：检索 + 生成Reference Answer + 质量检查
    """

    # 示例问题
    question = "What are the prerequisites for Machine Learning (IN2064)?"

    # 模拟检索结果（实际使用时替换为真实检索器）
    retrieved_docs = [
        {
            'doc_id': 'course_IN2064',
            'course_code': 'IN2064',
            'course_name': 'Machine Learning',
            'content': 'Machine Learning (IN2064) is an 8 ECTS course offered by the Department of Informatics. Prerequisites: Linear Algebra (MA1001), Probability Theory (MA2009). Students must pass both prerequisites before enrolling in this course.',
            'score': 0.95
        },
        {
            'doc_id': 'course_MA1001',
            'course_code': 'MA1001',
            'course_name': 'Linear Algebra',
            'content': 'Linear Algebra (MA1001) is a 6 ECTS foundational mathematics course. No prerequisites required. Offered in both Winter and Summer semesters.',
            'score': 0.82
        },
        {
            'doc_id': 'course_MA2009',
            'course_code': 'MA2009',
            'course_name': 'Probability Theory',
            'content': 'Probability Theory (MA2009) is an 8 ECTS course. Prerequisite: Linear Algebra (MA1001). Recommended to take before advanced informatics courses.',
            'score': 0.78
        }
    ]

    print("="*60)
    print("完整示例：生成Reference Answer")
    print("="*60)

    # Step 1: 生成Reference Answer
    result = generate_reference_answer(
        question=question,
        retrieved_docs=retrieved_docs,
        model="gpt-4",
        temperature=0.0,
        use_strict_prompt=True
    )

    # Step 2: 质量检查
    if result['success']:
        result = quality_check(result)

    # Step 3: 保存结果
    output_file = "example_reference_answer.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"结果已保存到: {output_file}")

    return result


# ===================== 批量处理 =====================

def batch_generate_with_retriever(
    questions_with_docs: List[Dict],
    output_file: str = "reference_answers.jsonl",
    **kwargs
) -> List[Dict]:
    """
    批量生成Reference Answers

    Args:
        questions_with_docs: [
            {
                'question': 'What is ML?',
                'retrieved_docs': [...]
            },
            ...
        ]
        output_file: 输出文件
        **kwargs: 传递给generate_reference_answer的参数

    Returns:
        结果列表
    """

    print(f"\n{'='*60}")
    print(f"批量生成Reference Answers")
    print(f"{'='*60}")
    print(f"总问题数: {len(questions_with_docs)}")

    results = []
    success_count = 0
    high_quality_count = 0

    for i, item in enumerate(questions_with_docs, 1):
        print(f"\n[{i}/{len(questions_with_docs)}]")

        # 生成
        result = generate_reference_answer(
            question=item['question'],
            retrieved_docs=item.get('retrieved_docs', []),
            **kwargs
        )

        # 质量检查
        if result['success']:
            result = quality_check(result)
            success_count += 1

            if result.get('quality_score', 0) >= 0.8:
                high_quality_count += 1

        results.append(result)

        # 实时保存
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print(f"批量生成完成")
    print(f"{'='*60}")
    print(f"总数: {len(questions_with_docs)}")
    print(f"成功: {success_count} ({success_count/len(questions_with_docs):.1%})")
    print(f"高质量 (≥80%): {high_quality_count} ({high_quality_count/len(questions_with_docs):.1%})")
    print(f"保存到: {output_file}")

    return results


# ===================== 使用说明 =====================

def print_usage():
    """打印使用说明"""

    print("""
╔═══════════════════════════════════════════════════════════╗
║       生成Reference Answer - 使用说明                     ║
╚═══════════════════════════════════════════════════════════╝

📋 完整流程：

1. 准备数据
   - 问题列表
   - 为每个问题检索相关文档

2. 调用函数生成
   result = generate_reference_answer(
       question="What is Machine Learning?",
       retrieved_docs=[...],  # 你的检索器返回的文档
       model="gpt-4",
       temperature=0.0,       # 推荐0.0确保稳定
       use_strict_prompt=True
   )

3. 质量检查
   result = quality_check(result)

4. 使用结果
   - 作为Golden Answer验证推理链
   - 计算F1 score
   - 实体匹配

═══════════════════════════════════════════════════════════

🎯 Prompt设计要点（已内置）：

✅ 明确角色：TUM学术顾问专家
✅ 强调用途：作为Gold Standard
✅ 要求准确性：必须基于检索文档
✅ 要求具体性：课程代码、ECTS、前置条件
✅ 要求完整性：全面回答问题
✅ 禁止猜测：没有信息就说没有

═══════════════════════════════════════════════════════════

💡 关键参数建议：

- model: "gpt-4" (推荐) 或 "gpt-4-turbo"
  * 不推荐gpt-3.5，质量可能不稳定

- temperature: 0.0 (推荐)
  * Reference Answer需要稳定一致
  * 不需要创造性

- use_strict_prompt: True (推荐)
  * 更严格的质量要求
  * 适合作为Golden Answer

═══════════════════════════════════════════════════════════

📊 质量评分标准：

- ≥80%: 优秀，可直接作为Golden Answer
- 60-80%: 一般，建议重新生成或优化检索
- <60%: 不合格，必须重新生成

检查维度：
1. 基于文档 (权重30%)
2. 长度合理
3. 包含课程代码
4. 包含ECTS信息
5. 无模糊表述
6. 包含结构化信息
7. 答案完整

═══════════════════════════════════════════════════════════
    """)


# ===================== 主函数 =====================

if __name__ == "__main__":
    print_usage()

    print("\n运行示例...\n")
    example_with_retrieval()

    print("\n" + "="*60)
    print("提示")
    print("="*60)
    print("""
接下来你需要：

1. 集成你的检索器
   在你的代码中调用检索器，获取相关文档

2. 批量生成
   questions_with_docs = [
       {
           'question': '...',
           'retrieved_docs': your_retriever.search(question, top_k=5)
       }
       for question in your_questions
   ]

   results = batch_generate_with_retriever(
       questions_with_docs,
       model="gpt-4",
       temperature=0.0,
       use_strict_prompt=True
   )

3. 保存结果作为Golden Answers
   用于后续生成推理链时的验证
    """)
