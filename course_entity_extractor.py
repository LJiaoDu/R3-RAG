#!/usr/bin/env python3
"""
自动提取课程实体（支持60+门课程）
使用课程列表 + 模糊匹配
"""

import json
import re
from typing import Set, List, Dict
from fuzzywuzzy import fuzz
from pathlib import Path

# ===================== 课程数据 =====================

# 示例：TUM信息学硕士常见课程（实际应该有60+门）
COURSES_DATABASE = [
    {
        "code": "IN2064",
        "name": "Machine Learning",
        "department": "Informatics",
        "ects": 8,
        "aliases": ["ML", "Machine Learning", "机器学习"]
    },
    {
        "code": "IN2346",
        "name": "Introduction to Deep Learning",
        "department": "Informatics",
        "ects": 6,
        "aliases": ["Intro to Deep Learning", "Intro DL", "Introduction to DL", "深度学习入门"]
    },
    {
        "code": "IN2375",
        "name": "Advanced Deep Learning",
        "department": "Informatics",
        "ects": 6,
        "aliases": ["Advanced DL", "Adv Deep Learning"]
    },
    {
        "code": "IN2128",
        "name": "Computer Vision",
        "department": "Informatics",
        "ects": 6,
        "aliases": ["CV", "Computer Vision", "计算机视觉"]
    },
    {
        "code": "IN2361",
        "name": "Natural Language Processing",
        "department": "Informatics",
        "ects": 6,
        "aliases": ["NLP", "Natural Language Processing", "自然语言处理"]
    },
    {
        "code": "IN2370",
        "name": "Reinforcement Learning",
        "department": "Informatics",
        "ects": 6,
        "aliases": ["RL", "Reinforcement Learning", "强化学习"]
    },
    {
        "code": "MA1001",
        "name": "Linear Algebra",
        "department": "Mathematics",
        "ects": 8,
        "aliases": ["LA", "Linear Algebra", "线性代数", "LinAlg", "Linear Alg"]
    },
    {
        "code": "MA2009",
        "name": "Probability Theory",
        "department": "Mathematics",
        "ects": 6,
        "aliases": ["Prob", "Probability", "Probability Theory", "概率论", "Prob Theory"]
    },
    {
        "code": "MA2001",
        "name": "Analysis",
        "department": "Mathematics",
        "ects": 8,
        "aliases": ["Analysis", "Mathematical Analysis", "数学分析"]
    },
    {
        "code": "IN2339",
        "name": "Database Systems",
        "department": "Informatics",
        "ects": 6,
        "aliases": ["DB", "Databases", "Database Systems", "数据库"]
    },
    # ... 实际应该有60+门课程
    # 可以从TUMonline API或课程手册爬取
]

# ===================== 实体提取器 =====================

class CourseEntityExtractor:
    """课程实体提取器"""

    def __init__(self, courses: List[Dict]):
        """
        初始化

        Args:
            courses: 课程列表
        """
        self.courses = courses

        # 构建索引
        self.code_to_name = {c['code']: c['name'] for c in courses}
        self.alias_to_name = {}

        for course in courses:
            name = course['name']
            # 课程名本身
            self.alias_to_name[name.lower()] = name
            # 所有别名
            for alias in course.get('aliases', []):
                self.alias_to_name[alias.lower()] = name

    def extract(self, text: str, fuzzy_threshold: int = 85) -> Set[str]:
        """
        从文本中提取课程实体

        Args:
            text: 输入文本
            fuzzy_threshold: 模糊匹配阈值（0-100）

        Returns:
            课程名称集合
        """

        entities = set()
        text_lower = text.lower()

        # 方法1: 精确匹配课程代码（如IN2064）
        codes = self._extract_course_codes(text)
        for code in codes:
            if code in self.code_to_name:
                entities.add(self.code_to_name[code])

        # 方法2: 精确匹配课程名和别名
        for alias, name in self.alias_to_name.items():
            # 使用词边界，避免部分匹配
            # 例如："ML"不会匹配"HTML"
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_lower):
                entities.add(name)

        # 方法3: 模糊匹配（处理拼写错误、变体）
        # 注意：这个比较慢，可选
        if fuzzy_threshold < 100:
            entities.update(self._fuzzy_match(text, fuzzy_threshold))

        return entities

    def _extract_course_codes(self, text: str) -> List[str]:
        """
        提取课程代码

        TUM课程代码格式：
        - 2个大写字母 + 4个数字
        - 例如：IN2064, MA1001, CIT5230
        """
        pattern = r'\b[A-Z]{2,3}\d{4}\b'
        codes = re.findall(pattern, text)
        return codes

    def _fuzzy_match(self, text: str, threshold: int) -> Set[str]:
        """
        模糊匹配

        用于处理：
        - 拼写错误："Machne Learning" → "Machine Learning"
        - 部分匹配："Deep Learning course" → "Deep Learning"
        """

        entities = set()
        text_lower = text.lower()

        for alias, name in self.alias_to_name.items():
            # 使用partial_ratio（部分匹配）
            similarity = fuzz.partial_ratio(alias, text_lower)

            if similarity >= threshold:
                # 双重检查：确保不是误匹配
                if len(alias) > 2:  # 避免太短的别名（如"ML"）误匹配
                    entities.add(name)

        return entities

    def batch_extract(self, texts: List[str]) -> List[Set[str]]:
        """批量提取"""
        return [self.extract(text) for text in texts]


# ===================== 辅助函数 =====================

def load_courses_from_json(filepath: str) -> List[Dict]:
    """从JSON文件加载课程列表"""

    if not Path(filepath).exists():
        print(f"⚠️  文件不存在: {filepath}")
        print(f"   使用内置课程数据")
        return COURSES_DATABASE

    with open(filepath, 'r', encoding='utf-8') as f:
        courses = json.load(f)

    print(f"✅ 从 {filepath} 加载了 {len(courses)} 门课程")
    return courses


def save_courses_to_json(courses: List[Dict], filepath: str):
    """保存课程列表到JSON"""

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(courses, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存 {len(courses)} 门课程到 {filepath}")


# ===================== 示例和测试 =====================

def test_extractor():
    """测试提取器"""

    print("="*60)
    print("课程实体提取器测试")
    print("="*60)

    # 初始化
    extractor = CourseEntityExtractor(COURSES_DATABASE)
    print(f"\n已加载 {len(COURSES_DATABASE)} 门课程\n")

    # 测试用例
    test_cases = [
        "You need Linear Algebra and Probability Theory before Machine Learning.",
        "Prerequisites: LA, Prob, and IN2064",
        "Complete MA1001 and MA2009 first",
        "I recommend taking ML, CV, and NLP together",
        "先学线性代数和概率论，然后学机器学习",
        "Take Introduction to Deep Learning after completing Machine Learning",
        "The required courses are IN2064, IN2346, and IN2128",
        "You need some foundational math courses",  # 模糊
        "Deep Learning is important",  # 部分匹配
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"测试 {i}:")
        print(f"  输入: {text}")

        entities = extractor.extract(text, fuzzy_threshold=85)

        if entities:
            print(f"  提取: {entities}")
        else:
            print(f"  提取: (无)")
        print()


def example_usage_with_voting():
    """结合投票机制的完整示例"""

    print("\n" + "="*60)
    print("完整示例：提取实体 + 投票")
    print("="*60)

    # 初始化提取器
    extractor = CourseEntityExtractor(COURSES_DATABASE)

    # 模拟6条生成的答案
    answers = [
        "You need Linear Algebra and Probability Theory",
        "Prerequisites: LA and Prob",
        "Complete Linear Algebra (MA1001) and Probability Theory (MA2009)",
        "You need LA, Prob, and optionally Analysis",
        "Some foundational math courses are required",
        "Machine Learning requires Linear Algebra"
    ]

    print(f"\n问题: What prerequisites for Machine Learning?")
    print(f"生成了 {len(answers)} 条答案\n")

    # Step 1: 提取每条答案的实体
    print("[Step 1] 提取实体:")
    all_entities = []
    for i, answer in enumerate(answers, 1):
        entities = extractor.extract(answer)
        all_entities.append(entities)
        print(f"  答案{i}: {entities}")

    # Step 2: 投票
    print(f"\n[Step 2] 投票统计:")
    from collections import Counter

    entity_counter = Counter()
    for entities in all_entities:
        for entity in entities:
            entity_counter[entity] += 1

    for entity, count in entity_counter.most_common():
        print(f"  {entity}: {count}票")

    # Step 3: 确定多数实体
    threshold = len(answers) * 0.5
    majority_entities = {
        entity for entity, count in entity_counter.items()
        if count >= threshold
    }

    print(f"\n[Step 3] 多数实体（≥{threshold}票）:")
    print(f"  {majority_entities}")

    # Step 4: 计算偏差
    print(f"\n[Step 4] 计算偏差:")
    for i, entities in enumerate(all_entities, 1):
        if majority_entities:
            coverage = len(entities & majority_entities) / len(majority_entities)
            deviation = 1 - coverage
        else:
            deviation = 0.0

        status = "✅" if deviation <= 0.5 else "❌"
        print(f"  答案{i}: 偏差={deviation:.2f} {status}")


# ===================== 课程数据准备 =====================

def prepare_course_database():
    """
    准备完整的60门课程数据

    实际应该：
    1. 从TUMonline API爬取
    2. 或者从课程手册PDF提取
    3. 或者手动整理
    """

    print("\n提示：准备完整的课程数据")
    print("="*60)
    print("方法1: 从TUMonline API爬取")
    print("  - 访问 https://campus.tum.de/tumonline/")
    print("  - 提取所有Informatics Master课程")
    print()
    print("方法2: 从课程手册提取")
    print("  - 下载TUM Informatics课程手册PDF")
    print("  - 使用PDF解析库提取课程信息")
    print()
    print("方法3: 手动整理")
    print("  - 根据你的知识手动创建课程列表")
    print("  - 包含：代码、名称、别名、ECTS等")
    print()
    print("示例格式:")
    print(json.dumps(COURSES_DATABASE[0], indent=2, ensure_ascii=False))
    print()
    print(f"当前内置了 {len(COURSES_DATABASE)} 门示例课程")
    print("实际使用时应该扩展到60+门")


# ===================== 主函数 =====================

if __name__ == "__main__":
    # 测试提取器
    test_extractor()

    # 完整示例
    example_usage_with_voting()

    # 准备课程数据的说明
    prepare_course_database()

    # 保存示例课程数据
    save_courses_to_json(COURSES_DATABASE, "courses_example.json")

    print("\n" + "="*60)
    print("提示：安装依赖")
    print("="*60)
    print("pip install fuzzywuzzy python-Levenshtein")
