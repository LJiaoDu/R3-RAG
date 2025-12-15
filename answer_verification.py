#!/usr/bin/env python3
"""
答案一致性验证模块
确保同一问题的多条推理链答案一致
"""

import re
from typing import List, Dict, Set, Tuple
from collections import Counter
import difflib

# ===================== 答案验证（有Gold Answer）=====================

def verify_answer_with_gold(generated: str, gold: str, threshold: float = 0.7) -> bool:
    """
    验证生成的答案是否与标准答案一致

    Args:
        generated: 生成的答案
        gold: 标准答案（gold answer）
        threshold: F1分数阈值

    Returns:
        True if答案正确，False otherwise
    """

    # 方法1: Exact Match（完全匹配）
    if normalize_text(generated) == normalize_text(gold):
        return True

    # 方法2: F1 Score（token重叠度）
    f1 = compute_f1_score(generated, gold)
    if f1 >= threshold:
        return True

    # 方法3: Entity Match（实体匹配）
    if entity_match(generated, gold, threshold=0.8):
        return True

    return False


def normalize_text(text: str) -> str:
    """标准化文本"""
    # 转小写
    text = text.lower()
    # 移除标点
    text = re.sub(r'[.,!?;:]', '', text)
    # 移除多余空格
    text = ' '.join(text.split())
    return text.strip()


def compute_f1_score(generated: str, gold: str) -> float:
    """
    计算F1分数（token level）

    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    gen_tokens = set(normalize_text(generated).split())
    gold_tokens = set(normalize_text(gold).split())

    if len(gen_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    # 交集
    common = gen_tokens & gold_tokens

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(gen_tokens)
    recall = len(common) / len(gold_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def entity_match(generated: str, gold: str, threshold: float = 0.8) -> bool:
    """
    检查关键实体是否匹配

    例如：
    gold: "Linear Algebra and Probability Theory"
    generated: "You need LA and Prob Theory"
    → 检测到"LA"是"Linear Algebra"的缩写 → True
    """
    # 提取实体（课程名称）
    gen_entities = extract_course_entities(generated)
    gold_entities = extract_course_entities(gold)

    if len(gold_entities) == 0:
        return False

    # 计算实体重叠
    matched = 0
    for gold_ent in gold_entities:
        for gen_ent in gen_entities:
            if is_same_entity(gen_ent, gold_ent):
                matched += 1
                break

    coverage = matched / len(gold_entities)
    return coverage >= threshold


def extract_course_entities(text: str) -> Set[str]:
    """
    提取课程实体

    例如：
    "You need Linear Algebra and Probability Theory"
    → {"Linear Algebra", "Probability Theory"}
    """
    entities = set()

    # 常见课程名称模式
    patterns = [
        r'Linear Algebra',
        r'Probability Theory',
        r'Machine Learning',
        r'Deep Learning',
        r'Computer Vision',
        r'Natural Language Processing',
        r'NLP',
        # 课程代码
        r'[A-Z]{2}\d{4}',  # 如 IN2064
        r'MA\d{4}',        # 如 MA1001
    ]

    text_lower = text.lower()

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.add(match.group(0))

    # 缩写检测
    abbreviations = {
        'LA': 'Linear Algebra',
        'Prob': 'Probability',
        'Probability': 'Probability Theory',
        'ML': 'Machine Learning',
        'DL': 'Deep Learning',
        'CV': 'Computer Vision',
    }

    for abbr, full in abbreviations.items():
        if abbr.lower() in text_lower:
            entities.add(full)

    return entities


def is_same_entity(entity1: str, entity2: str) -> bool:
    """检查两个实体是否相同（考虑缩写和变体）"""

    e1 = entity1.lower().strip()
    e2 = entity2.lower().strip()

    # 完全匹配
    if e1 == e2:
        return True

    # 部分匹配（如"Probability"和"Probability Theory"）
    if e1 in e2 or e2 in e1:
        return True

    # 相似度匹配
    similarity = difflib.SequenceMatcher(None, e1, e2).ratio()
    if similarity > 0.8:
        return True

    return False


# ===================== 答案验证（无Gold Answer）=====================

def verify_answers_consistency(chains: List[Dict], threshold: float = 0.7) -> List[Dict]:
    """
    验证多条链的答案一致性（无标准答案时使用）

    使用多数投票和一致性检查

    Args:
        chains: 生成的推理链列表
        threshold: 一致性阈值

    Returns:
        一致的推理链列表
    """

    if len(chains) <= 1:
        return chains  # 只有1条，无法验证

    # 方法1: 多数投票（Majority Voting）
    majority_chains = majority_voting(chains)

    # 方法2: 一致性检查（Consistency Check）
    consistent_chains = consistency_check(chains, threshold)

    # 取交集（同时满足多数投票和一致性检查）
    final_chains = [
        chain for chain in chains
        if chain in majority_chains and chain in consistent_chains
    ]

    return final_chains if final_chains else majority_chains  # 至少返回多数投票结果


def majority_voting(chains: List[Dict]) -> List[Dict]:
    """
    多数投票：保留答案与多数一致的链
    """

    # 提取所有答案的实体
    all_entities = []
    for chain in chains:
        entities = extract_course_entities(chain.get('answer', ''))
        all_entities.append(entities)

    # 统计实体出现次数
    entity_counts = Counter()
    for entities in all_entities:
        for ent in entities:
            entity_counts[ent] += 1

    # 找出多数实体（出现在>50%的答案中）
    majority_threshold = len(chains) * 0.5
    majority_entities = {
        ent for ent, count in entity_counts.items()
        if count > majority_threshold
    }

    if not majority_entities:
        # 如果没有多数实体，降低阈值到30%
        majority_threshold = len(chains) * 0.3
        majority_entities = {
            ent for ent, count in entity_counts.items()
            if count > majority_threshold
        }

    # 保留包含多数实体的链
    majority_chains = []
    for i, entities in enumerate(all_entities):
        # 计算覆盖率
        if majority_entities:
            coverage = len(entities & majority_entities) / len(majority_entities)
            if coverage >= 0.5:  # 至少包含50%的多数实体
                majority_chains.append(chains[i])

    return majority_chains


def consistency_check(chains: List[Dict], threshold: float = 0.7) -> List[Dict]:
    """
    一致性检查：保留与其他答案相似度高的链
    """

    answers = [chain.get('answer', '') for chain in chains]

    # 计算每个答案与其他答案的平均相似度
    avg_similarity = []
    for i in range(len(answers)):
        similarities = []
        for j in range(len(answers)):
            if i != j:
                sim = compute_f1_score(answers[i], answers[j])
                similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        avg_similarity.append(avg_sim)

    # 保留平均相似度 >= threshold的
    consistent_chains = [
        chain for i, chain in enumerate(chains)
        if avg_similarity[i] >= threshold
    ]

    return consistent_chains


# ===================== 完整验证流程 =====================

def validate_generated_chains(
    question: str,
    chains: List[Dict],
    gold_answer: str = None,
    mode: str = 'auto'
) -> Tuple[List[Dict], Dict]:
    """
    完整的链验证流程

    Args:
        question: 问题
        chains: 生成的推理链
        gold_answer: 标准答案（可选）
        mode: 验证模式 ('gold', 'consistency', 'auto')

    Returns:
        (validated_chains, statistics)
    """

    stats = {
        'total_attempts': len(chains),
        'valid_chains': 0,
        'invalid_chains': 0,
        'validation_method': ''
    }

    # 自动选择验证方法
    if mode == 'auto':
        mode = 'gold' if gold_answer else 'consistency'

    valid_chains = []

    if mode == 'gold' and gold_answer:
        # 使用标准答案验证
        stats['validation_method'] = 'gold_answer'

        for chain in chains:
            answer = chain.get('answer', '')
            if verify_answer_with_gold(answer, gold_answer):
                valid_chains.append(chain)
                stats['valid_chains'] += 1
            else:
                stats['invalid_chains'] += 1

    elif mode == 'consistency':
        # 使用一致性验证
        stats['validation_method'] = 'consistency_check'

        valid_chains = verify_answers_consistency(chains)
        stats['valid_chains'] = len(valid_chains)
        stats['invalid_chains'] = len(chains) - len(valid_chains)

    return valid_chains, stats


# ===================== 使用示例 =====================

def example_usage():
    """示例：验证同一问题的多条链"""

    question = "What courses do I need before Machine Learning?"
    gold_answer = "Linear Algebra and Probability Theory"

    # 模拟7次生成的结果
    chains = [
        {
            "attempt": 1,
            "temperature": 0.0,
            "answer": "You need Linear Algebra and Probability Theory before Machine Learning.",
            "num_steps": 3
        },
        {
            "attempt": 2,
            "temperature": 0.9,
            "answer": "Prerequisites: 1. Linear Algebra (MA1001) 2. Probability Theory (MA2009) 3. Python",
            "num_steps": 4
        },
        {
            "attempt": 3,
            "temperature": 0.9,
            "answer": "You need some foundational math courses.",
            "num_steps": 2
        },
        {
            "attempt": 4,
            "temperature": 0.9,
            "answer": "Before ML, complete Linear Algebra and Probability courses.",
            "num_steps": 5
        },
        {
            "attempt": 5,
            "temperature": 0.9,
            "answer": None,  # 失败
            "error": "Timeout"
        },
        {
            "attempt": 6,
            "temperature": 0.9,
            "answer": "ML requires LA and Prob Theory. Path: Sem1 LA → Sem2 Prob → Sem3 ML",
            "num_steps": 3
        },
        {
            "attempt": 7,
            "temperature": 0.9,
            "answer": "Machine Learning is worth 8 ECTS.",
            "num_steps": 4
        }
    ]

    # 过滤失败的
    chains = [c for c in chains if c.get('answer')]

    print("="*60)
    print("答案验证示例")
    print("="*60)
    print(f"\n问题: {question}")
    print(f"标准答案: {gold_answer}")
    print(f"\n尝试次数: 7")
    print(f"成功生成: {len(chains)}条")

    # 验证
    valid_chains, stats = validate_generated_chains(
        question,
        chains,
        gold_answer,
        mode='gold'
    )

    print(f"\n验证结果:")
    print(f"  验证方法: {stats['validation_method']}")
    print(f"  有效链: {stats['valid_chains']}条")
    print(f"  无效链: {stats['invalid_chains']}条")

    print(f"\n保留的链:")
    for i, chain in enumerate(valid_chains):
        print(f"\n  链{i+1} (尝试{chain['attempt']}, temp={chain['temperature']}):")
        print(f"    步数: {chain['num_steps']}")
        print(f"    答案: {chain['answer'][:100]}...")

        # 计算F1分数
        f1 = compute_f1_score(chain['answer'], gold_answer)
        print(f"    F1分数: {f1:.2f}")


if __name__ == "__main__":
    example_usage()
