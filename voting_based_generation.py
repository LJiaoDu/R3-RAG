#!/usr/bin/env python3
"""
基于投票的数据生成方案
- 每个问题生成6条推理链
- 组内投票决定正确答案
- 去除偏差大的链
- 保留一致的4-5条
"""

import openai
import json
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter
from answer_verification import extract_course_entities, compute_f1_score
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# ===================== 配置 =====================

CONFIG = {
    "num_attempts_per_question": 6,  # 每个问题生成6条
    "temperature_anchor": 0.0,        # 第1条用0.0（anchor）
    "temperature_exploratory": 0.9,   # 第2-6条用0.9（探索）
    "min_consensus": 3,               # 至少3条（50%）一致
    "deviation_threshold": 0.5,       # 偏差阈值
    "min_valid_chains": 2,            # 至少保留2条
}

# ===================== 核心功能 =====================

def generate_6_chains(question: str) -> List[Dict]:
    """
    为一个问题生成6条推理链

    策略：
    - 第1条: temperature=0.0（最保守，作为anchor）
    - 第2-6条: temperature=0.9（探索多样性）
    """

    chains = []

    # 第1条：anchor
    print(f"  生成anchor（temp=0.0）...")
    anchor = generate_single_chain(question, temperature=0.0)
    if anchor:
        anchor['role'] = 'anchor'
        chains.append(anchor)

    # 第2-6条：exploratory
    for i in range(5):
        print(f"  生成exploratory {i+1}/5（temp=0.9）...")
        chain = generate_single_chain(question, temperature=0.9)
        if chain:
            chain['role'] = 'exploratory'
            chains.append(chain)

    return chains


def generate_single_chain(question: str, temperature: float) -> Dict:
    """生成单条推理链（简化版，实际应该包含检索交互）"""

    try:
        # 这里简化了，实际应该是多轮交互
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a TUM course consultant. Answer with clear reasoning."},
                {"role": "user", "content": question}
            ],
            temperature=temperature,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        return {
            "question": question,
            "answer": answer,
            "temperature": temperature
        }

    except Exception as e:
        print(f"    ❌ 生成失败: {e}")
        return None


# ===================== 投票机制 =====================

def majority_voting(chains: List[Dict]) -> Tuple[Set[str], Dict]:
    """
    多数投票：确定正确答案

    Returns:
        (majority_entities, vote_stats)
    """

    if len(chains) == 0:
        return set(), {}

    # 提取所有答案的实体
    all_entities = []
    for chain in chains:
        entities = extract_course_entities(chain.get('answer', ''))
        all_entities.append(entities)

    # 统计实体出现次数
    entity_counter = Counter()
    for entities in all_entities:
        for ent in entities:
            entity_counter[ent] += 1

    # 多数阈值：至少出现在50%的答案中
    min_votes = len(chains) * 0.5
    majority_entities = {
        ent for ent, count in entity_counter.items()
        if count >= min_votes
    }

    # 统计信息
    vote_stats = {
        "total_chains": len(chains),
        "unique_entities": len(entity_counter),
        "majority_entities": len(majority_entities),
        "entity_votes": dict(entity_counter.most_common(10))
    }

    return majority_entities, vote_stats


# ===================== 偏差计算 =====================

def compute_deviation(chain: Dict, majority_entities: Set[str]) -> float:
    """
    计算答案与多数的偏差

    偏差 = 1 - 覆盖度

    Returns:
        偏差值（0-1，越小越好）
    """

    if not majority_entities:
        return 0.0  # 没有多数实体，无法计算

    answer = chain.get('answer', '')
    answer_entities = extract_course_entities(answer)

    # 计算覆盖度
    coverage = len(answer_entities & majority_entities) / len(majority_entities)

    # 偏差
    deviation = 1 - coverage

    return deviation


def filter_outliers(chains: List[Dict], majority_entities: Set[str],
                    threshold: float = 0.5) -> Tuple[List[Dict], List[Dict]]:
    """
    过滤偏差大的链

    Args:
        chains: 推理链列表
        majority_entities: 多数实体
        threshold: 偏差阈值

    Returns:
        (valid_chains, outlier_chains)
    """

    valid_chains = []
    outlier_chains = []

    for chain in chains:
        deviation = compute_deviation(chain, majority_entities)
        chain['deviation'] = deviation

        if deviation <= threshold:
            valid_chains.append(chain)
        else:
            outlier_chains.append(chain)

    return valid_chains, outlier_chains


# ===================== 质量检查 =====================

def quality_check(chains: List[Dict], anchor: Dict) -> Dict:
    """
    质量检查

    检查项：
    1. 是否有足够的一致性
    2. exploratory是否与anchor偏差过大
    3. 是否需要人工review
    """

    result = {
        "pass": True,
        "warnings": [],
        "needs_review": False
    }

    # 检查1：一致性
    if len(chains) < CONFIG["min_valid_chains"]:
        result["warnings"].append(f"保留链数太少: {len(chains)}")
        result["needs_review"] = True

    # 检查2：探索性链与anchor的偏差
    if anchor:
        anchor_answer = anchor.get('answer', '')
        exploratory_chains = [c for c in chains if c.get('role') == 'exploratory']

        deviations = []
        for chain in exploratory_chains:
            f1 = compute_f1_score(chain.get('answer', ''), anchor_answer)
            deviations.append(1 - f1)  # deviation = 1 - similarity

        if deviations:
            avg_deviation = np.mean(deviations)
            if avg_deviation > 0.7:  # 平均偏差>70%
                result["warnings"].append(f"探索性链与anchor偏差过大: {avg_deviation:.2f}")
                result["needs_review"] = True

    # 检查3：是否所有链都被过滤了
    if len(chains) == 0:
        result["pass"] = False
        result["warnings"].append("所有链都被过滤，可能有问题")
        result["needs_review"] = True

    return result


# ===================== 主流程 =====================

def process_question_with_voting(question: str) -> Dict:
    """
    完整流程：生成、投票、过滤

    Returns:
        {
            "question": str,
            "valid_chains": List[Dict],
            "outlier_chains": List[Dict],
            "majority_entities": Set[str],
            "vote_stats": Dict,
            "quality_check": Dict
        }
    """

    print(f"\n{'='*60}")
    print(f"处理问题: {question}")
    print(f"{'='*60}")

    # Step 1: 生成6条链
    print("\n[Step 1] 生成6条推理链...")
    chains = generate_6_chains(question)
    print(f"✅ 成功生成 {len(chains)}/6 条")

    if len(chains) < 2:
        print("❌ 生成失败过多，跳过")
        return None

    # Step 2: 多数投票
    print("\n[Step 2] 多数投票...")
    majority_entities, vote_stats = majority_voting(chains)
    print(f"✅ 多数实体: {majority_entities}")
    print(f"   投票统计: {vote_stats['entity_votes']}")

    # Step 3: 过滤偏差大的
    print("\n[Step 3] 过滤偏差大的链...")
    valid_chains, outlier_chains = filter_outliers(
        chains,
        majority_entities,
        threshold=CONFIG["deviation_threshold"]
    )

    print(f"✅ 保留 {len(valid_chains)} 条有效链")
    print(f"❌ 过滤 {len(outlier_chains)} 条偏差大的链")

    # 显示偏差信息
    for i, chain in enumerate(valid_chains):
        print(f"   有效链{i+1}: 偏差={chain['deviation']:.2f}, temp={chain['temperature']}")

    for i, chain in enumerate(outlier_chains):
        print(f"   异常链{i+1}: 偏差={chain['deviation']:.2f}, temp={chain['temperature']}")

    # Step 4: 质量检查
    print("\n[Step 4] 质量检查...")
    anchor = next((c for c in chains if c.get('role') == 'anchor'), None)
    quality = quality_check(valid_chains, anchor)

    if quality["warnings"]:
        print(f"⚠️  警告:")
        for warning in quality["warnings"]:
            print(f"   - {warning}")

    if quality["needs_review"]:
        print(f"🔍 需要人工review")
    else:
        print(f"✅ 质量检查通过")

    return {
        "question": question,
        "valid_chains": valid_chains,
        "outlier_chains": outlier_chains,
        "majority_entities": majority_entities,
        "vote_stats": vote_stats,
        "quality_check": quality
    }


# ===================== 批量处理 =====================

def batch_process_with_voting(questions: List[str], output_file: str):
    """批量处理问题"""

    results = []
    stats = {
        "total": len(questions),
        "success": 0,
        "needs_review": 0,
        "failed": 0,
        "avg_valid_chains": []
    }

    for i, question in enumerate(questions):
        print(f"\n{'#'*60}")
        print(f"进度: {i+1}/{len(questions)}")
        print(f"{'#'*60}")

        result = process_question_with_voting(question)

        if result:
            results.append(result)
            stats["success"] += 1
            stats["avg_valid_chains"].append(len(result["valid_chains"]))

            if result["quality_check"]["needs_review"]:
                stats["needs_review"] += 1
        else:
            stats["failed"] += 1

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # 统计
    print(f"\n{'='*60}")
    print(f"批量处理完成")
    print(f"{'='*60}")
    print(f"总问题数: {stats['total']}")
    print(f"成功: {stats['success']}")
    print(f"需要review: {stats['needs_review']}")
    print(f"失败: {stats['failed']}")

    if stats['avg_valid_chains']:
        avg = np.mean(stats['avg_valid_chains'])
        print(f"平均保留链数: {avg:.1f}")

    print(f"\n输出文件: {output_file}")

    return results, stats


# ===================== 示例 =====================

def example_usage():
    """使用示例"""

    # 测试单个问题
    question = "What courses do I need before Machine Learning?"

    result = process_question_with_voting(question)

    if result:
        print(f"\n{'='*60}")
        print(f"最终结果")
        print(f"{'='*60}")
        print(f"问题: {result['question']}")
        print(f"多数实体: {result['majority_entities']}")
        print(f"保留链数: {len(result['valid_chains'])}")
        print(f"异常链数: {len(result['outlier_chains'])}")

        print(f"\n保留的链:")
        for i, chain in enumerate(result['valid_chains']):
            print(f"\n  链{i+1}:")
            print(f"    答案: {chain['answer'][:100]}...")
            print(f"    偏差: {chain['deviation']:.2f}")
            print(f"    温度: {chain['temperature']}")


if __name__ == "__main__":
    print("投票式数据生成方案\n")

    # 运行示例
    example_usage()
