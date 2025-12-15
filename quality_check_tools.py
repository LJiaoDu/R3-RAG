#!/usr/bin/env python3
"""
Reference Answer质量抽检工具集
包含所有自动化检查、抽样、验证功能
"""

import json
import random
import re
from typing import List, Dict, Tuple
from collections import defaultdict
from pathlib import Path


# ===================== 工具1: 自动化检查 =====================

def load_reference_answers(file_path: str) -> List[Dict]:
    """加载所有Reference Answers"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            results.append(data)
    return results


def automatic_check_all(results: List[Dict]) -> Dict:
    """
    对所有结果进行自动检查
    """

    report = {
        'total': len(results),
        'issues': defaultdict(list),
        'statistics': {}
    }

    for i, result in enumerate(results):
        question = result.get('question', f'Question_{i}')

        # 检查1: 生成成功
        if not result.get('success'):
            report['issues']['failed_generation'].append({
                'index': i,
                'question': question,
                'error': result.get('error', 'Unknown error')
            })
            continue

        structured = result.get('structured_answer')
        if not structured:
            report['issues']['no_structured_answer'].append({
                'index': i,
                'question': question
            })
            continue

        # 检查2: 必需字段
        required_fields = ['answer_summary', 'facts', 'unknown']
        missing_fields = [f for f in required_fields if f not in structured]
        if missing_fields:
            report['issues']['missing_fields'].append({
                'index': i,
                'question': question,
                'missing': missing_fields
            })

        # 检查3: 引用完整性
        facts = structured.get('facts', [])
        for j, fact in enumerate(facts):
            if 'citations' not in fact or len(fact['citations']) == 0:
                report['issues']['missing_citations'].append({
                    'index': i,
                    'question': question,
                    'fact_index': j,
                    'claim': fact.get('claim', 'Unknown')
                })

        # 检查4: 引用有效性
        invalid_citations = result.get('invalid_citations', [])
        if len(invalid_citations) > 0:
            report['issues']['invalid_citations'].append({
                'index': i,
                'question': question,
                'invalid': invalid_citations
            })

        # 检查5: 质量分数
        quality_score = result.get('quality_score', 0)
        if quality_score < 0.6:
            report['issues']['low_quality'].append({
                'index': i,
                'question': question,
                'score': quality_score
            })

        # 检查6: Facts为空
        if len(facts) == 0:
            report['issues']['no_facts'].append({
                'index': i,
                'question': question
            })

        # 检查7: Answer summary为空
        answer_summary = structured.get('answer_summary', '')
        if len(answer_summary.strip()) < 10:
            report['issues']['empty_answer'].append({
                'index': i,
                'question': question
            })

    # 统计信息
    successful_results = [r for r in results if r.get('success')]
    report['statistics'] = {
        'success_rate': len(successful_results) / len(results) if results else 0,
        'avg_quality_score': sum(r.get('quality_score', 0) for r in successful_results) / len(successful_results) if successful_results else 0,
        'avg_facts_count': sum(len(r.get('structured_answer', {}).get('facts', [])) for r in successful_results) / len(successful_results) if successful_results else 0,
        'avg_unknown_count': sum(len(r.get('structured_answer', {}).get('unknown', [])) for r in successful_results) / len(successful_results) if successful_results else 0,
        'questions_with_issues': sum(len(v) for v in report['issues'].values())
    }

    return report


def print_automatic_check_report(report: Dict):
    """打印自动检查报告"""

    print(f"\n{'='*60}")
    print(f"自动化检查报告")
    print(f"{'='*60}")

    print(f"\n📊 统计信息:")
    stats = report['statistics']
    print(f"  总数: {report['total']}")
    print(f"  成功率: {stats['success_rate']:.1%}")
    print(f"  平均质量分: {stats['avg_quality_score']:.2f}")
    print(f"  平均Facts数: {stats['avg_facts_count']:.1f}")
    print(f"  平均Unknown数: {stats['avg_unknown_count']:.1f}")
    print(f"  有问题的数量: {stats['questions_with_issues']}")

    print(f"\n🔍 问题详情:")

    issue_names = {
        'failed_generation': '生成失败',
        'no_structured_answer': '无结构化答案',
        'missing_fields': '缺少必需字段',
        'missing_citations': '缺少引用',
        'invalid_citations': '无效引用',
        'low_quality': '低质量',
        'no_facts': '无Facts',
        'empty_answer': '答案为空'
    }

    for issue_type, issues in report['issues'].items():
        if len(issues) > 0:
            issue_name = issue_names.get(issue_type, issue_type)
            print(f"\n  [{issue_name}] {len(issues)}个问题")

            # 显示前3个示例
            for issue in issues[:3]:
                question = issue.get('question', 'Unknown')
                if len(question) > 50:
                    question = question[:50] + '...'
                print(f"    - 问题 {issue['index']}: {question}")

            if len(issues) > 3:
                print(f"    ... 还有 {len(issues) - 3} 个")

    print(f"\n{'='*60}")


# ===================== 工具2: 分层抽样 =====================

def stratified_sampling(results: List[Dict], sample_size: int = 500) -> Dict[str, List[Dict]]:
    """
    分层抽样
    """

    # 过滤成功的结果
    successful_results = [r for r in results if r.get('success')]

    if len(successful_results) < sample_size:
        print(f"⚠️  成功结果数({len(successful_results)})少于抽样数({sample_size})，调整为{len(successful_results)}")
        sample_size = len(successful_results)

    samples = {
        'random': [],
        'low_quality': [],
        'high_unknown': [],
        'complex': [],
        'simple': []
    }

    # 1. 随机抽样（30%）
    random_sample_size = int(sample_size * 0.3)
    samples['random'] = random.sample(successful_results, min(random_sample_size, len(successful_results)))

    # 2. 低质量样本（30%）
    low_quality = sorted(successful_results, key=lambda x: x.get('quality_score', 1.0))
    samples['low_quality'] = low_quality[:int(sample_size * 0.3)]

    # 3. 高unknown样本（20%）
    high_unknown = sorted(
        successful_results,
        key=lambda x: len(x.get('structured_answer', {}).get('unknown', [])),
        reverse=True
    )
    samples['high_unknown'] = high_unknown[:int(sample_size * 0.2)]

    # 4. 复杂问题（10%）
    complex_qs = sorted(
        successful_results,
        key=lambda x: len(x.get('structured_answer', {}).get('facts', [])),
        reverse=True
    )
    samples['complex'] = complex_qs[:int(sample_size * 0.1)]

    # 5. 简单问题（10%）
    simple_qs = sorted(
        successful_results,
        key=lambda x: len(x.get('structured_answer', {}).get('facts', []))
    )
    samples['simple'] = simple_qs[:int(sample_size * 0.1)]

    return samples


def save_samples_for_review(samples: Dict[str, List[Dict]], output_dir: str = 'review'):
    """保存抽样结果供人工检查"""
    Path(output_dir).mkdir(exist_ok=True)

    category_names = {
        'random': '随机抽样',
        'low_quality': '低质量',
        'high_unknown': '高未知信息',
        'complex': '复杂问题',
        'simple': '简单问题'
    }

    for category, sample_list in samples.items():
        output_file = f"{output_dir}/{category}_samples.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in sample_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        name = category_names.get(category, category)
        print(f"✅ {name}: {len(sample_list)}个样本 → {output_file}")


# ===================== 工具3: Citation验证 =====================

def verify_claim_against_citation(claim: str, citation_id: str, retrieved_docs: List[Dict]) -> Dict:
    """验证claim是否被citation支持"""

    # 1. 提取citation对应的chunk内容
    try:
        doc_id, chunk_id = citation_id.split('#')
    except ValueError:
        return {
            'valid': False,
            'citation_content': None,
            'match_score': 0.0,
            'issues': ['Citation格式错误']
        }

    chunk_content = None
    for doc in retrieved_docs:
        if doc.get('doc_id') == doc_id:
            if 'chunks' in doc:
                for chunk in doc['chunks']:
                    if chunk.get('chunk_id') == chunk_id:
                        chunk_content = chunk.get('content', '')
                        break
            else:
                # 整个文档
                if chunk_id == 'full':
                    chunk_content = doc.get('content', '')
            break

    if chunk_content is None:
        return {
            'valid': False,
            'citation_content': None,
            'match_score': 0.0,
            'issues': [f'Citation不存在: {citation_id}']
        }

    # 2. 提取关键信息
    # 课程代码
    claim_codes = set(re.findall(r'\b[A-Z]{2,3}\d{4}\b', claim))
    content_codes = set(re.findall(r'\b[A-Z]{2,3}\d{4}\b', chunk_content))

    # ECTS
    claim_ects = set(re.findall(r'(\d+)\s*ECTS', claim, re.IGNORECASE))
    content_ects = set(re.findall(r'(\d+)\s*ECTS', chunk_content, re.IGNORECASE))

    # 关键词
    claim_keywords = set(w.lower() for w in re.findall(r'\b\w+\b', claim) if len(w) > 3)
    content_keywords = set(w.lower() for w in re.findall(r'\b\w+\b', chunk_content) if len(w) > 3)

    # 3. 计算匹配度
    issues = []
    match_score = 0.0
    score_components = []

    # 课程代码匹配
    if claim_codes:
        code_match = len(claim_codes & content_codes) / len(claim_codes)
        score_components.append(('code', code_match, 0.4))
        if code_match < 1.0:
            issues.append(f"课程代码不完全匹配: claim={claim_codes}, content={content_codes}")

    # ECTS匹配
    if claim_ects:
        ects_match = len(claim_ects & content_ects) / len(claim_ects)
        score_components.append(('ects', ects_match, 0.3))
        if ects_match < 1.0:
            issues.append(f"ECTS不匹配: claim={claim_ects}, content={content_ects}")

    # 关键词匹配
    if claim_keywords:
        keyword_match = len(claim_keywords & content_keywords) / len(claim_keywords)
        score_components.append(('keywords', keyword_match, 0.3))

    # 计算总分
    if score_components:
        total_weight = sum(w for _, _, w in score_components)
        match_score = sum(score * weight for _, score, weight in score_components) / total_weight
    else:
        match_score = 0.0

    # 4. 判断
    valid = match_score >= 0.6 and len(issues) == 0

    return {
        'valid': valid,
        'citation_content': chunk_content[:200] + '...' if len(chunk_content) > 200 else chunk_content,
        'match_score': match_score,
        'score_components': score_components,
        'issues': issues
    }


def batch_verify_citations(results: List[Dict]) -> Dict:
    """批量验证所有citations"""

    report = {
        'total_facts': 0,
        'total_citations': 0,
        'valid_citations': 0,
        'invalid_citations': [],
        'suspicious_citations': []
    }

    for i, result in enumerate(results):
        if not result.get('success'):
            continue

        structured = result.get('structured_answer', {})
        facts = structured.get('facts', [])
        retrieved_docs = result.get('retrieved_docs', [])

        for fact_idx, fact in enumerate(facts):
            report['total_facts'] += 1
            claim = fact.get('claim', '')
            citations = fact.get('citations', [])

            for citation in citations:
                report['total_citations'] += 1

                verification = verify_claim_against_citation(
                    claim, citation, retrieved_docs
                )

                if verification['valid']:
                    report['valid_citations'] += 1
                elif verification['match_score'] >= 0.4:
                    # 可疑但不完全无效
                    report['suspicious_citations'].append({
                        'index': i,
                        'question': result['question'],
                        'fact_index': fact_idx,
                        'claim': claim,
                        'citation': citation,
                        'score': verification['match_score'],
                        'issues': verification['issues']
                    })
                else:
                    # 完全无效
                    report['invalid_citations'].append({
                        'index': i,
                        'question': result['question'],
                        'fact_index': fact_idx,
                        'claim': claim,
                        'citation': citation,
                        'score': verification['match_score'],
                        'issues': verification['issues']
                    })

    if report['total_citations'] > 0:
        report['valid_rate'] = report['valid_citations'] / report['total_citations']
        report['suspicious_rate'] = len(report['suspicious_citations']) / report['total_citations']
        report['invalid_rate'] = len(report['invalid_citations']) / report['total_citations']
    else:
        report['valid_rate'] = 0
        report['suspicious_rate'] = 0
        report['invalid_rate'] = 0

    return report


def print_citation_verification_report(report: Dict):
    """打印Citation验证报告"""

    print(f"\n{'='*60}")
    print(f"Citation验证报告")
    print(f"{'='*60}")

    print(f"\n📊 统计:")
    print(f"  总Facts数: {report['total_facts']}")
    print(f"  总Citations数: {report['total_citations']}")
    print(f"  有效Citations: {report['valid_citations']} ({report['valid_rate']:.1%})")
    print(f"  可疑Citations: {len(report['suspicious_citations'])} ({report['suspicious_rate']:.1%})")
    print(f"  无效Citations: {len(report['invalid_citations'])} ({report['invalid_rate']:.1%})")

    if report['invalid_citations']:
        print(f"\n❌ 无效Citations示例（前3个）:")
        for item in report['invalid_citations'][:3]:
            print(f"\n  问题 {item['index']}: {item['question'][:50]}...")
            print(f"  Claim: {item['claim'][:80]}...")
            print(f"  Citation: {item['citation']}")
            print(f"  匹配分: {item['score']:.2f}")
            print(f"  问题: {', '.join(item['issues'])}")

    if report['suspicious_citations']:
        print(f"\n⚠️  可疑Citations示例（前3个）:")
        for item in report['suspicious_citations'][:3]:
            print(f"\n  问题 {item['index']}: {item['question'][:50]}...")
            print(f"  Claim: {item['claim'][:80]}...")
            print(f"  Citation: {item['citation']}")
            print(f"  匹配分: {item['score']:.2f}")

    print(f"\n{'='*60}")


# ===================== 工具4: 综合报告 =====================

def generate_comprehensive_report(
    results: List[Dict],
    auto_check_report: Dict,
    citation_report: Dict
) -> Dict:
    """生成综合质量报告"""

    total = len(results)
    successful = sum(1 for r in results if r.get('success'))

    # 质量分布
    quality_bins = {
        '优秀 (≥0.9)': 0,
        '良好 (0.8-0.9)': 0,
        '一般 (0.6-0.8)': 0,
        '差 (<0.6)': 0
    }

    for r in results:
        if not r.get('success'):
            continue
        score = r.get('quality_score', 0)
        if score >= 0.9:
            quality_bins['优秀 (≥0.9)'] += 1
        elif score >= 0.8:
            quality_bins['良好 (0.8-0.9)'] += 1
        elif score >= 0.6:
            quality_bins['一般 (0.6-0.8)'] += 1
        else:
            quality_bins['差 (<0.6)'] += 1

    # 建议
    recommendations = []

    if auto_check_report['statistics']['success_rate'] < 0.95:
        recommendations.append('❗ 成功率较低(<95%)，检查API调用和检索质量')

    if auto_check_report['statistics']['avg_quality_score'] < 0.8:
        recommendations.append('❗ 平均质量分较低(<0.8)，考虑优化Prompt或检索')

    if quality_bins['差 (<0.6)'] > total * 0.1:
        recommendations.append(f"❗ 低质量样本过多({quality_bins['差 (<0.6)']}个，{quality_bins['差 (<0.6)']/total:.1%})，需要重新生成")

    if citation_report['invalid_rate'] > 0.05:
        recommendations.append(f"❗ 无效引用率较高({citation_report['invalid_rate']:.1%})，检查检索器和citation格式")

    if not recommendations:
        recommendations.append('✅ 数据质量良好，可以用于训练')

    report = {
        'overview': {
            'total': total,
            'successful': successful,
            'success_rate': successful / total if total > 0 else 0,
            'avg_quality_score': auto_check_report['statistics']['avg_quality_score'],
            'avg_facts': auto_check_report['statistics']['avg_facts_count'],
            'avg_unknown': auto_check_report['statistics']['avg_unknown_count']
        },
        'quality_distribution': quality_bins,
        'issues_summary': {
            'total_issues': auto_check_report['statistics']['questions_with_issues'],
            'by_type': {k: len(v) for k, v in auto_check_report['issues'].items()}
        },
        'citation_quality': {
            'valid_rate': citation_report['valid_rate'],
            'suspicious_count': len(citation_report['suspicious_citations']),
            'invalid_count': len(citation_report['invalid_citations'])
        },
        'recommendations': recommendations
    }

    return report


def print_comprehensive_report(report: Dict):
    """打印综合报告"""

    print(f"\n{'='*60}")
    print(f"Reference Answer 综合质量报告")
    print(f"{'='*60}")

    # 概览
    print(f"\n📊 概览:")
    overview = report['overview']
    print(f"  总问题数: {overview['total']}")
    print(f"  成功生成: {overview['successful']} ({overview['success_rate']:.1%})")
    print(f"  平均质量分: {overview['avg_quality_score']:.2f}")
    print(f"  平均Facts数: {overview['avg_facts']:.1f}")
    print(f"  平均Unknown数: {overview['avg_unknown']:.1f}")

    # 质量分布
    print(f"\n📈 质量分布:")
    for category, count in report['quality_distribution'].items():
        pct = count / overview['successful'] * 100 if overview['successful'] > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {category:20s} {count:4d} ({pct:5.1f}%) {bar}")

    # 问题总结
    print(f"\n🔍 问题总结:")
    print(f"  有问题的样本: {report['issues_summary']['total_issues']}")
    for issue_type, count in report['issues_summary']['by_type'].items():
        if count > 0:
            print(f"    - {issue_type}: {count}")

    # Citation质量
    print(f"\n📎 Citation质量:")
    cit = report['citation_quality']
    print(f"  有效率: {cit['valid_rate']:.1%}")
    print(f"  可疑数: {cit['suspicious_count']}")
    print(f"  无效数: {cit['invalid_count']}")

    # 建议
    print(f"\n💡 建议:")
    for rec in report['recommendations']:
        print(f"  {rec}")

    print(f"\n{'='*60}")


# ===================== 主函数 =====================

def main():
    """主函数 - 运行完整的质量检查流程"""

    import sys

    if len(sys.argv) < 2:
        print("用法: python quality_check_tools.py <reference_answers.jsonl>")
        print("\n示例:")
        print("  python quality_check_tools.py reference_answers.jsonl")
        return

    input_file = sys.argv[1]

    print(f"{'='*60}")
    print(f"Reference Answer 质量检查工具")
    print(f"{'='*60}")
    print(f"输入文件: {input_file}\n")

    # 加载数据
    print("正在加载数据...")
    try:
        results = load_reference_answers(input_file)
        print(f"✅ 加载了 {len(results)} 条记录\n")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 步骤1: 自动化检查
    print("步骤1: 自动化检查...")
    auto_check_report = automatic_check_all(results)
    print_automatic_check_report(auto_check_report)

    # 步骤2: 分层抽样
    print("\n步骤2: 分层抽样...")
    sample_size = min(500, len(results) // 10)  # 最多500个或10%
    samples = stratified_sampling(results, sample_size)
    save_samples_for_review(samples)
    print(f"\n✅ 总抽样数: {sum(len(v) for v in samples.values())}")

    # 步骤3: Citation验证
    print("\n步骤3: Citation验证...")
    citation_report = batch_verify_citations(results)
    print_citation_verification_report(citation_report)

    # 步骤4: 综合报告
    print("\n步骤4: 生成综合报告...")
    comprehensive_report = generate_comprehensive_report(
        results, auto_check_report, citation_report
    )
    print_comprehensive_report(comprehensive_report)

    # 保存报告
    output_dir = Path('review')
    output_dir.mkdir(exist_ok=True)

    report_file = output_dir / 'quality_check_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'auto_check': auto_check_report,
            'citation_verification': citation_report,
            'comprehensive': comprehensive_report
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 详细报告已保存到: {report_file}")

    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"{'='*60}")
    print(f"\n下一步:")
    print(f"1. 查看 review/ 目录下的抽样文件")
    print(f"2. 进行人工抽检")
    print(f"3. 根据建议修复问题")
    print(f"4. 重新生成低质量样本")


if __name__ == "__main__":
    main()
