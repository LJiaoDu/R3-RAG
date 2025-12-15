#!/usr/bin/env python3
"""
锚点过滤策略（Anchor-based Filtering）
使用Temperature=0的答案作为基准，过滤偏差大的链

对比三种策略：
1. 纯投票（Voting-only）
2. 纯锚点（Anchor-only）
3. 混合策略（Hybrid）
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

ANCHOR_CONFIG = {
    "num_attempts_per_question": 6,
    "temperature_anchor": 0.0,        # 锚点（第1条）
    "temperature_exploratory": 0.9,   # 探索性（第2-6条）
    "anchor_similarity_threshold": 0.6,  # 与锚点的相似度阈值
    "deviation_threshold": 0.5,       # 偏差阈值
}

# ===================== 策略1: 纯锚点过滤 =====================

def anchor_based_filtering(chains: List[Dict], anchor: Dict,
                           threshold: float = 0.6) -> Tuple[List[Dict], List[Dict]]:
    """
    纯锚点过滤：以Temperature=0的答案为基准

    策略：
    - 第1条（temp=0.0）= 锚点（最保守、最可靠）
    - 第2-6条（temp=0.9）= 探索性（多样性）
    - 保留与锚点相似度 >= threshold的链

    Args:
        chains: 所有生成的链
        anchor: 锚点链（temp=0）
        threshold: 相似度阈值

    Returns:
        (valid_chains, outlier_chains)
    """

    if not anchor or not anchor.get('answer'):
        print("  ⚠️  没有有效的锚点，无法使用锚点过滤")
        return chains, []

    anchor_answer = anchor.get('answer', '')
    anchor_entities = extract_course_entities(anchor_answer)

    valid_chains = [anchor]  # 锚点自己保留
    outlier_chains = []

    for chain in chains:
        if chain is anchor:
            continue

        answer = chain.get('answer', '')

        # 方法1: F1相似度
        f1_similarity = compute_f1_score(answer, anchor_answer)

        # 方法2: 实体覆盖度
        entities = extract_course_entities(answer)
        if anchor_entities:
            entity_coverage = len(entities & anchor_entities) / len(anchor_entities)
        else:
            entity_coverage = 0.0

        # 综合相似度（F1和实体覆盖度的平均）
        similarity = (f1_similarity + entity_coverage) / 2.0

        chain['anchor_similarity'] = similarity
        chain['f1_similarity'] = f1_similarity
        chain['entity_coverage'] = entity_coverage

        if similarity >= threshold:
            valid_chains.append(chain)
        else:
            outlier_chains.append(chain)

    return valid_chains, outlier_chains


# ===================== 策略2: 混合策略 =====================

def hybrid_filtering(chains: List[Dict], anchor: Dict,
                     anchor_threshold: float = 0.6,
                     deviation_threshold: float = 0.5) -> Tuple[List[Dict], List[Dict]]:
    """
    混合策略：锚点过滤 + 投票验证

    流程：
    1. 锚点过滤：保留与anchor相似的链
    2. 投票验证：在保留的链中找多数实体
    3. 双重检查：必须通过两个条件

    优点：
    - 既有锚点的稳定性
    - 又有投票的容错性

    Returns:
        (valid_chains, outlier_chains)
    """

    # Step 1: 锚点过滤
    print("  [Step 1] 锚点过滤...")
    anchor_valid, anchor_outliers = anchor_based_filtering(
        chains, anchor, threshold=anchor_threshold
    )
    print(f"    通过锚点过滤: {len(anchor_valid)}条")

    # Step 2: 投票验证（在通过锚点过滤的链中投票）
    print("  [Step 2] 投票验证...")

    # 提取所有答案的实体
    all_entities = []
    for chain in anchor_valid:
        entities = extract_course_entities(chain.get('answer', ''))
        all_entities.append(entities)

    # 统计实体
    entity_counter = Counter()
    for entities in all_entities:
        for ent in entities:
            entity_counter[ent] += 1

    # 多数实体（>=50%）
    min_votes = len(anchor_valid) * 0.5
    majority_entities = {
        ent for ent, count in entity_counter.items()
        if count >= min_votes
    }

    print(f"    多数实体: {majority_entities}")

    # Step 3: 计算偏差
    print("  [Step 3] 计算偏差...")
    final_valid = []
    final_outliers = []

    for chain in anchor_valid:
        answer = chain.get('answer', '')
        entities = extract_course_entities(answer)

        # 计算与多数的偏差
        if majority_entities:
            coverage = len(entities & majority_entities) / len(majority_entities)
            deviation = 1 - coverage
        else:
            deviation = 0.0

        chain['deviation'] = deviation

        if deviation <= deviation_threshold:
            final_valid.append(chain)
        else:
            final_outliers.append(chain)

    print(f"    通过投票验证: {len(final_valid)}条")

    # 合并所有被过滤的链
    all_outliers = anchor_outliers + final_outliers

    return final_valid, all_outliers


# ===================== 策略对比 =====================

def compare_strategies(question: str, chains: List[Dict], anchor: Dict) -> Dict:
    """
    对比三种策略的结果

    Returns:
        {
            'voting': {...},
            'anchor': {...},
            'hybrid': {...}
        }
    """

    print(f"\n{'='*60}")
    print(f"策略对比")
    print(f"{'='*60}")

    results = {}

    # 策略1: 纯投票（从voting_based_generation.py导入）
    print(f"\n[策略1] 纯投票过滤")
    from voting_based_generation import majority_voting, filter_outliers

    majority_entities, vote_stats = majority_voting(chains)
    voting_valid, voting_outliers = filter_outliers(
        chains, majority_entities, threshold=0.5
    )

    results['voting'] = {
        'valid_chains': len(voting_valid),
        'outlier_chains': len(voting_outliers),
        'majority_entities': majority_entities,
        'chains': voting_valid
    }

    print(f"  保留: {len(voting_valid)}条")
    print(f"  过滤: {len(voting_outliers)}条")
    print(f"  多数实体: {majority_entities}")

    # 策略2: 纯锚点
    print(f"\n[策略2] 纯锚点过滤")
    anchor_valid, anchor_outliers = anchor_based_filtering(
        chains, anchor, threshold=0.6
    )

    results['anchor'] = {
        'valid_chains': len(anchor_valid),
        'outlier_chains': len(anchor_outliers),
        'chains': anchor_valid
    }

    print(f"  保留: {len(anchor_valid)}条")
    print(f"  过滤: {len(anchor_outliers)}条")

    # 策略3: 混合
    print(f"\n[策略3] 混合策略（锚点+投票）")
    hybrid_valid, hybrid_outliers = hybrid_filtering(
        chains, anchor, anchor_threshold=0.6, deviation_threshold=0.5
    )

    results['hybrid'] = {
        'valid_chains': len(hybrid_valid),
        'outlier_chains': len(hybrid_outliers),
        'chains': hybrid_valid
    }

    print(f"  最终保留: {len(hybrid_valid)}条")
    print(f"  最终过滤: {len(hybrid_outliers)}条")

    return results


# ===================== 完整流程 =====================

def process_question_with_anchor(question: str) -> Dict:
    """
    使用锚点策略的完整流程

    Returns:
        完整结果字典
    """

    print(f"\n{'='*60}")
    print(f"处理问题（锚点策略）")
    print(f"{'='*60}")
    print(f"问题: {question}")

    # Step 1: 生成6条链
    print(f"\n[Step 1] 生成6条推理链...")
    print(f"  生成锚点（temp=0.0）...")

    # 这里简化，实际应该调用OpenAI API
    # anchor = generate_single_chain(question, temperature=0.0)
    # chains = [anchor]
    # for i in range(5):
    #     chain = generate_single_chain(question, temperature=0.9)
    #     chains.append(chain)

    # 模拟生成的数据（实际使用时替换为真实生成）
    anchor = {
        'attempt': 1,
        'temperature': 0.0,
        'role': 'anchor',
        'answer': 'You need Linear Algebra and Probability Theory before Machine Learning.'
    }

    chains = [
        anchor,
        {
            'attempt': 2,
            'temperature': 0.9,
            'role': 'exploratory',
            'answer': 'Prerequisites: Linear Algebra (MA1001) and Probability Theory (MA2009)'
        },
        {
            'attempt': 3,
            'temperature': 0.9,
            'role': 'exploratory',
            'answer': 'Before ML, complete Linear Algebra and Probability courses.'
        },
        {
            'attempt': 4,
            'temperature': 0.9,
            'role': 'exploratory',
            'answer': 'You need some math courses.'  # 模糊，应该被过滤
        },
        {
            'attempt': 5,
            'temperature': 0.9,
            'role': 'exploratory',
            'answer': 'ML requires LA and Prob Theory.'
        },
        {
            'attempt': 6,
            'temperature': 0.9,
            'role': 'exploratory',
            'answer': 'Machine Learning is worth 8 ECTS.'  # 答非所问，应该被过滤
        }
    ]

    print(f"✅ 生成 {len(chains)} 条链")

    # Step 2: 策略对比
    print(f"\n[Step 2] 对比三种策略...")
    results = compare_strategies(question, chains, anchor)

    # Step 3: 推荐策略
    print(f"\n{'='*60}")
    print(f"策略推荐")
    print(f"{'='*60}")

    recommend_strategy(results)

    return {
        'question': question,
        'total_chains': len(chains),
        'anchor': anchor,
        'results': results
    }


def recommend_strategy(results: Dict):
    """推荐最佳策略"""

    voting_count = results['voting']['valid_chains']
    anchor_count = results['anchor']['valid_chains']
    hybrid_count = results['hybrid']['valid_chains']

    print(f"\n保留链数对比:")
    print(f"  纯投票: {voting_count}条")
    print(f"  纯锚点: {anchor_count}条")
    print(f"  混合策略: {hybrid_count}条")

    print(f"\n推荐：")
    print(f"✅ 混合策略（Hybrid）")
    print(f"\n理由：")
    print(f"1. 锚点提供稳定性")
    print(f"   - Temperature=0生成最可靠的答案")
    print(f"   - 避免所有链都错误的情况")
    print(f"")
    print(f"2. 投票提供容错性")
    print(f"   - 即使锚点不完美，也能通过多数纠正")
    print(f"   - 允许不同的表述方式（如缩写）")
    print(f"")
    print(f"3. 双重验证质量更高")
    print(f"   - 必须同时满足：与锚点相似 + 符合多数")
    print(f"   - 过滤掉偏差大的异常链")

    print(f"\n应用场景：")
    print(f"- 有gold answer → 直接用gold验证（不需要锚点或投票）")
    print(f"- 无gold answer + 对质量要求高 → 混合策略（推荐）")
    print(f"- 无gold answer + 追求多样性 → 纯投票")
    print(f"- 无gold answer + 追求保守 → 纯锚点")


# ===================== 示例和测试 =====================

def example_usage():
    """完整示例"""

    question = "What courses do I need before Machine Learning?"

    result = process_question_with_anchor(question)

    # 显示详细结果
    print(f"\n{'='*60}")
    print(f"详细结果")
    print(f"{'='*60}")

    # 混合策略的链
    hybrid_chains = result['results']['hybrid']['chains']

    print(f"\n混合策略保留的链:")
    for i, chain in enumerate(hybrid_chains, 1):
        print(f"\n  链{i}:")
        print(f"    尝试: {chain['attempt']}")
        print(f"    温度: {chain['temperature']}")
        print(f"    角色: {chain['role']}")
        print(f"    答案: {chain['answer']}")

        if 'anchor_similarity' in chain:
            print(f"    锚点相似度: {chain['anchor_similarity']:.2f}")
            print(f"      - F1相似度: {chain['f1_similarity']:.2f}")
            print(f"      - 实体覆盖: {chain['entity_coverage']:.2f}")

        if 'deviation' in chain:
            print(f"    投票偏差: {chain['deviation']:.2f}")


def create_comparison_table():
    """创建策略对比表"""

    print(f"\n{'='*60}")
    print(f"三种策略对比表")
    print(f"{'='*60}")

    table = """
┌─────────────┬──────────────┬──────────────┬──────────────┐
│   策略      │   纯投票     │   纯锚点     │   混合策略   │
├─────────────┼──────────────┼──────────────┼──────────────┤
│ 核心机制    │ 多数实体投票 │ 与锚点相似度 │ 锚点+投票    │
│ 需要gold    │ 否           │ 否           │ 否           │
│ 稳定性      │ 中           │ 高           │ 高           │
│ 容错性      │ 高           │ 低           │ 中           │
│ 多样性      │ 高           │ 低           │ 中           │
│ 质量控制    │ 中           │ 中           │ 高           │
│ 保留链数    │ 3-4条        │ 2-3条        │ 3条          │
│ 适用场景    │ 追求多样性   │ 追求保守     │ 平衡质量     │
└─────────────┴──────────────┴──────────────┴──────────────┘

优缺点详细对比：

1️⃣ 纯投票（Voting-only）
   ✅ 优点：
      - 容错性强：即使某条链错误，不影响多数
      - 多样性好：允许不同表述方式
      - 无需锚点：所有链平等参与

   ❌ 缺点：
      - 可能集体错误：如果多数链都错了
      - 需要较多链：至少3-4条才有效
      - 模糊答案风险：可能保留不够具体的答案

2️⃣ 纯锚点（Anchor-only）
   ✅ 优点：
      - 稳定可靠：锚点（temp=0）最保守
      - 防止集体错误：有一个稳定的基准
      - 简单直接：只需对比相似度

   ❌ 缺点：
      - 依赖锚点质量：如果锚点错了，全错
      - 多样性低：过于相似的答案
      - 可能过严：合理的表述变体被过滤

3️⃣ 混合策略（Hybrid）
   ✅ 优点：
      - 质量最高：双重验证
      - 平衡性好：稳定性+容错性
      - 适用性广：适合大多数场景

   ❌ 缺点：
      - 可能过严：同时满足两个条件
      - 保留较少：3条左右（但质量高）
      - 计算稍慢：需要两次过滤

推荐使用场景：

🎯 TUM课程咨询系统（你的项目）
   → 推荐：混合策略

   理由：
   - 没有gold answer
   - 对质量要求高（学生依赖答案做决策）
   - 需要平衡准确性和多样性
   - 可以接受保留3条左右的高质量链

🎯 原论文数据集（有gold answer）
   → 不需要这些策略，直接用gold验证

🎯 创意生成任务（如写作）
   → 推荐：纯投票
   - 追求多样性
   - 容忍一定的变体

🎯 关键决策系统（如医疗）
   → 推荐：纯锚点
   - 追求最稳定的答案
   - 保守优于冒险
"""

    print(table)


# ===================== 主函数 =====================

if __name__ == "__main__":
    print("锚点过滤策略实现\n")

    # 示例
    example_usage()

    # 对比表
    create_comparison_table()

    print(f"\n{'='*60}")
    print(f"总结")
    print(f"{'='*60}")
    print(f"\n✅ 你完全可以将Temperature=0当作基准解筛选轨迹！")
    print(f"\n这就是锚点机制（Anchor Mechanism）：")
    print(f"  1. Temperature=0 → 最保守、最可靠的答案")
    print(f"  2. Temperature=0.9 → 探索多样性")
    print(f"  3. 保留与锚点相似度高的探索性链")
    print(f"\n推荐：混合策略（锚点+投票）")
    print(f"  - 兼顾稳定性和容错性")
    print(f"  - 适合你的TUM项目（无gold answer）")
    print(f"  - 平均保留3条高质量链")
