#!/usr/bin/env python3
"""
生成Reference Answer - 改进版（防止脏数据）

关键改进：
1. 不再使用"GOLD STANDARD"等强制确定性的表述
2. 结构化输出（JSON格式，包含引用）
3. 检索结果带可引用的锚点ID
4. 条件化质量要求（文档有才要求，没有则标记unknown）
5. 允许诚实承认信息不足
6. 每条事实强制带引用
"""

import openai
import json
import os
from typing import List, Dict, Optional

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# ===================== 改进的Prompt设计 =====================

# 【改进】System Prompt - 不再强调"GOLD STANDARD"
IMPROVED_SYSTEM_PROMPT = """You are an expert academic advisor at Technical University of Munich (TUM).

Your task: Generate evidence-grounded reference answers for course consultation questions.

IMPORTANT PRINCIPLES:
1. HONESTY: If information is not in the retrieved documents, explicitly say so
2. EVIDENCE-BASED: Every factual claim must cite specific document chunks
3. STRUCTURED: Output in JSON format for verifiability
4. NO SPECULATION: Never infer or add information beyond what's explicitly stated
5. UNCERTAINTY-AWARE: Mark information gaps in the "unknown" field

Output format requirements:
- Use structured JSON (not free-form text)
- Every fact must include citation(s)
- Clearly separate known facts from unknown information
- Use exact citation IDs from the retrieved documents"""


# 【改进】User Prompt - 结构化输出要求
IMPROVED_USER_PROMPT_TEMPLATE = """Student Question:
{question}

Retrieved Documents (cite using [doc_id#chunk_id]):
{formatted_docs_with_ids}

REQUIRED OUTPUT FORMAT (JSON):
{{
  "answer_summary": "Brief direct answer to the question",
  "facts": [
    {{
      "claim": "Specific factual statement",
      "citations": ["doc_1#chunk_2", "doc_3#chunk_1"],
      "confidence": "high|medium|low"
    }}
  ],
  "unknown": [
    "What information is missing or unclear"
  ],
  "used_documents": ["doc_1", "doc_3"]
}}

CRITICAL RULES:
1. Every claim in "facts" MUST have at least one citation
2. Only include information that is explicitly stated in the documents
3. If a commonly expected detail (like ECTS, prerequisites) is not in documents, list it in "unknown"
4. Do NOT fabricate course codes, ECTS credits, or prerequisites to "complete" the answer
5. Use citation IDs exactly as shown: [doc_id#chunk_id]

Generate the structured reference answer:"""


# ===================== 改进的文档格式化 =====================

def format_documents_with_citation_ids(documents: List[Dict]) -> str:
    """
    格式化文档，每个chunk都有可引用的锚点ID

    Args:
        documents: [
            {
                'doc_id': 'course_IN2064',
                'chunks': [
                    {
                        'chunk_id': 'chunk_1',
                        'content': '...'
                    }
                ]
            }
        ]

    Returns:
        格式化的文档字符串，每个chunk有 [doc_id#chunk_id]
    """

    if not documents:
        return "No documents retrieved."

    formatted = ""

    for doc in documents:
        doc_id = doc.get('doc_id', 'unknown_doc')

        # 如果有chunks（推荐格式）
        if 'chunks' in doc:
            for chunk in doc['chunks']:
                chunk_id = chunk.get('chunk_id', 'unknown_chunk')
                content = chunk.get('content', '')

                # 格式：[doc_id#chunk_id] content
                formatted += f"[{doc_id}#{chunk_id}]\n{content}\n\n"

        # 如果没有chunks（整个文档）
        else:
            content = doc.get('content', doc.get('text', ''))
            formatted += f"[{doc_id}#full]\n{content}\n\n"

    return formatted


# ===================== 改进的生成函数 =====================

def generate_improved_reference_answer(
    question: str,
    retrieved_docs: List[Dict],
    model: str = "gpt-4",
    temperature: float = 0.0,
    max_tokens: int = 1000
) -> Dict:
    """
    生成改进的Reference Answer（防止脏数据）

    Args:
        question: 学生问题
        retrieved_docs: 检索文档（必须包含doc_id和chunk_id）
        model: OpenAI模型
        temperature: 0.0（确保稳定）
        max_tokens: 最大token数

    Returns:
        {
            'question': str,
            'structured_answer': {
                'answer_summary': str,
                'facts': [{'claim', 'citations', 'confidence'}],
                'unknown': [...],
                'used_documents': [...]
            },
            'raw_response': str,
            'success': bool
        }
    """

    print(f"\n{'='*60}")
    print(f"生成Reference Answer（改进版）")
    print(f"{'='*60}")
    print(f"问题: {question}")
    print(f"检索文档数: {len(retrieved_docs)}")

    # Step 1: 格式化文档（带引用ID）
    formatted_docs = format_documents_with_citation_ids(retrieved_docs)

    # Step 2: 构建prompt
    user_prompt = IMPROVED_USER_PROMPT_TEMPLATE.format(
        question=question,
        formatted_docs_with_ids=formatted_docs
    )

    print(f"使用改进的Prompt（结构化输出）")

    # Step 3: 调用OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": IMPROVED_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        raw_response = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens

        print(f"✅ 生成成功")
        print(f"Tokens使用: {tokens_used}")

        # Step 4: 解析JSON响应
        try:
            # 提取JSON（可能在markdown代码块中）
            if "```json" in raw_response:
                json_str = raw_response.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_response:
                json_str = raw_response.split("```")[1].split("```")[0].strip()
            else:
                json_str = raw_response

            structured_answer = json.loads(json_str)

            print(f"\n--- Structured Answer ---")
            print(json.dumps(structured_answer, indent=2, ensure_ascii=False))
            print(f"{'='*60}\n")

            return {
                'question': question,
                'structured_answer': structured_answer,
                'raw_response': raw_response,
                'retrieved_docs': retrieved_docs,
                'model': model,
                'temperature': temperature,
                'tokens_used': tokens_used,
                'success': True
            }

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON解析失败: {e}")
            print(f"原始响应:\n{raw_response}")

            return {
                'question': question,
                'structured_answer': None,
                'raw_response': raw_response,
                'retrieved_docs': retrieved_docs,
                'error': f"JSON decode error: {e}",
                'success': False
            }

    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return {
            'question': question,
            'structured_answer': None,
            'error': str(e),
            'success': False
        }


# ===================== 改进的质量检查 =====================

def improved_quality_check(result: Dict) -> Dict:
    """
    改进的质量检查（防止脏数据）

    检查维度：
    1. 结构完整性（必须有facts和unknown）
    2. 引用完整性（每个fact必须有citations）
    3. 引用有效性（citations必须存在于检索文档中）
    4. 信息诚实性（允许unknown不为空）
    5. 无编造信息（所有claim必须可追溯）
    """

    print(f"\n{'='*60}")
    print(f"质量检查（改进版）")
    print(f"{'='*60}")

    structured_answer = result.get('structured_answer')
    docs = result.get('retrieved_docs', [])

    if not structured_answer:
        print(f"❌ 没有结构化答案")
        return {**result, 'quality_score': 0.0, 'quality_checks': {}}

    checks = {}

    # 1. 结构完整性
    required_fields = ['answer_summary', 'facts', 'unknown']
    has_all_fields = all(field in structured_answer for field in required_fields)
    checks['structure_complete'] = has_all_fields
    status = '✅' if has_all_fields else '❌'
    print(f"{status} 结构完整性: {'完整' if has_all_fields else '缺少必要字段'}")

    # 2. 引用完整性
    facts = structured_answer.get('facts', [])
    all_facts_have_citations = all(
        'citations' in fact and len(fact.get('citations', [])) > 0
        for fact in facts
    )
    checks['citations_complete'] = all_facts_have_citations
    status = '✅' if all_facts_have_citations else '❌'
    print(f"{status} 引用完整性: {len(facts)}个facts，{'全部有引用' if all_facts_have_citations else '部分缺少引用'}")

    # 3. 引用有效性（检查citations是否存在于文档中）
    valid_citation_ids = set()
    for doc in docs:
        doc_id = doc.get('doc_id', '')
        if 'chunks' in doc:
            for chunk in doc['chunks']:
                chunk_id = chunk.get('chunk_id', '')
                valid_citation_ids.add(f"{doc_id}#{chunk_id}")
        else:
            valid_citation_ids.add(f"{doc_id}#full")

    invalid_citations = []
    for fact in facts:
        for cite in fact.get('citations', []):
            if cite not in valid_citation_ids:
                invalid_citations.append(cite)

    checks['citations_valid'] = len(invalid_citations) == 0
    status = '✅' if len(invalid_citations) == 0 else '⚠️'
    print(f"{status} 引用有效性: {'全部有效' if len(invalid_citations) == 0 else f'{len(invalid_citations)}个无效引用'}")

    # 4. 信息诚实性（允许unknown不为空）
    unknown = structured_answer.get('unknown', [])
    has_unknown = len(unknown) > 0
    checks['has_unknown'] = has_unknown
    status = 'ℹ️'
    print(f"{status} 信息诚实性: {'承认了{len(unknown)}个未知信息' if has_unknown else '所有信息都已知'}")

    # 5. 答案非空
    answer_summary = structured_answer.get('answer_summary', '')
    checks['answer_not_empty'] = len(answer_summary) > 10
    status = '✅' if len(answer_summary) > 10 else '⚠️'
    print(f"{status} 答案非空: {len(answer_summary)} 字符")

    # 6. Facts数量合理
    num_facts = len(facts)
    checks['facts_count_reasonable'] = 1 <= num_facts <= 20
    status = '✅' if 1 <= num_facts <= 20 else '⚠️'
    print(f"{status} Facts数量: {num_facts} (建议1-20)")

    # 计算总分（所有检查项等权重）
    # 注意：has_unknown不计分（允许有unknown）
    scored_checks = [
        checks['structure_complete'],
        checks['citations_complete'],
        checks['citations_valid'],
        checks['answer_not_empty'],
        checks['facts_count_reasonable']
    ]

    score = sum(scored_checks) / len(scored_checks)

    print(f"\n{'='*60}")
    print(f"总体质量评分: {score:.1%}")

    if score >= 0.8:
        print(f"✅ 质量优秀")
    elif score >= 0.6:
        print(f"⚠️  质量一般")
    else:
        print(f"❌ 质量不足")

    # 详细报告
    if invalid_citations:
        print(f"\n⚠️  无效引用: {invalid_citations}")

    if unknown:
        print(f"\nℹ️  未知信息 (正常): {unknown}")

    print(f"{'='*60}\n")

    return {
        **result,
        'quality_score': score,
        'quality_checks': checks,
        'invalid_citations': invalid_citations
    }


# ===================== 示例 =====================

def example_improved_version():
    """
    改进版本的完整示例
    """

    # 示例问题
    question = "What are the prerequisites for Machine Learning?"

    # 模拟检索结果（改进格式：带chunk_id）
    retrieved_docs = [
        {
            'doc_id': 'course_IN2064',
            'chunks': [
                {
                    'chunk_id': 'prereq',
                    'content': 'Machine Learning (IN2064) is an 8 ECTS course. Prerequisites: Linear Algebra (MA1001), Probability Theory (MA2009).'
                },
                {
                    'chunk_id': 'description',
                    'content': 'The course covers supervised learning, unsupervised learning, and neural networks.'
                }
            ]
        },
        {
            'doc_id': 'course_MA1001',
            'chunks': [
                {
                    'chunk_id': 'info',
                    'content': 'Linear Algebra (MA1001) is a 6 ECTS foundational mathematics course. No prerequisites.'
                }
            ]
        },
        {
            'doc_id': 'course_MA2009',
            'chunks': [
                {
                    'chunk_id': 'info',
                    'content': 'Probability Theory (MA2009) is 8 ECTS. Prerequisite: Linear Algebra (MA1001).'
                }
            ]
        }
    ]

    print("="*60)
    print("改进版Reference Answer生成示例")
    print("="*60)

    # 生成
    result = generate_improved_reference_answer(
        question=question,
        retrieved_docs=retrieved_docs,
        model="gpt-4",
        temperature=0.0
    )

    # 质量检查
    if result['success']:
        result = improved_quality_check(result)

    # 保存
    output_file = "improved_reference_answer_example.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"结果已保存到: {output_file}")

    return result


# ===================== 期望的输出格式示例 =====================

EXPECTED_OUTPUT_EXAMPLE = """
期望的输出格式示例：

{
  "answer_summary": "The prerequisites for Machine Learning (IN2064) are Linear Algebra (MA1001) and Probability Theory (MA2009). These must be taken in sequence as Probability Theory itself requires Linear Algebra.",

  "facts": [
    {
      "claim": "Machine Learning (IN2064) is an 8 ECTS course",
      "citations": ["course_IN2064#prereq"],
      "confidence": "high"
    },
    {
      "claim": "Prerequisites for IN2064 are Linear Algebra (MA1001) and Probability Theory (MA2009)",
      "citations": ["course_IN2064#prereq"],
      "confidence": "high"
    },
    {
      "claim": "Linear Algebra (MA1001) is 6 ECTS with no prerequisites",
      "citations": ["course_MA1001#info"],
      "confidence": "high"
    },
    {
      "claim": "Probability Theory (MA2009) is 8 ECTS and requires MA1001",
      "citations": ["course_MA2009#info"],
      "confidence": "high"
    }
  ],

  "unknown": [
    "Exam format for Machine Learning",
    "Number of assignments",
    "Whether courses can be taken concurrently"
  ],

  "used_documents": ["course_IN2064", "course_MA1001", "course_MA2009"]
}

优点：
✅ 每个事实都有明确的引用
✅ 可以自动验证引用的有效性
✅ 明确列出未知信息（诚实）
✅ 结构化，便于后续处理
✅ 不会为了"完整"而编造信息
"""


# ===================== 使用说明 =====================

def print_usage():
    """打印使用说明"""

    print("""
╔═══════════════════════════════════════════════════════════╗
║       改进版Reference Answer生成 - 防止脏数据            ║
╚═══════════════════════════════════════════════════════════╝

🔧 关键改进：

1. ❌ 去掉 "GOLD STANDARD / 100% correct"
   ✅ 改为 "Evidence-grounded Answer / Silver Label"
   → 防止模型"装确定"

2. ❌ 自由文本输出
   ✅ 结构化JSON输出（facts + citations + unknown）
   → 可验证、可追溯

3. ❌ "--- Document 1 ---"
   ✅ "[doc_id#chunk_id]"锚点ID
   → 精确引用、自动验证

4. ❌ 硬要求"必须有课程代码/ECTS"
   ✅ 条件触发：文档有就写，没有进unknown
   → 防止编造

5. ❌ "no information"判为bad
   ✅ 允许并鼓励诚实承认信息不足
   → 信息诚实性

6. ❌ 事实混在正文
   ✅ 每个fact必须带citation
   → 强制引用

═══════════════════════════════════════════════════════════

📋 使用方法：

# 1. 准备文档（必须有doc_id和chunk_id）
docs = [
    {
        'doc_id': 'course_IN2064',
        'chunks': [
            {'chunk_id': 'prereq', 'content': '...'},
            {'chunk_id': 'description', 'content': '...'}
        ]
    }
]

# 2. 生成
result = generate_improved_reference_answer(
    question="What are the prerequisites for ML?",
    retrieved_docs=docs,
    model="gpt-4",
    temperature=0.0
)

# 3. 质量检查
if result['success']:
    result = improved_quality_check(result)

# 4. 使用结构化输出
structured = result['structured_answer']
for fact in structured['facts']:
    print(f"Claim: {fact['claim']}")
    print(f"Citations: {fact['citations']}")

═══════════════════════════════════════════════════════════

⚠️  重要提示：

1. 检索器必须返回带doc_id和chunk_id的文档
2. 不要在prompt里强调"必须正确"或"必须完整"
3. 允许unknown列表不为空（这是好事）
4. 每次生成后都要验证引用的有效性
5. 如果模型不返回JSON，需要重试或调整prompt

═══════════════════════════════════════════════════════════
    """)


# ===================== 主函数 =====================

if __name__ == "__main__":
    print_usage()

    print("\n运行改进版示例...\n")
    example_improved_version()

    print(EXPECTED_OUTPUT_EXAMPLE)
