#!/usr/bin/env python3
"""
准备多语言课程语料库
从PDF/文本文件提取内容并分块
"""

import os
import json
from typing import List, Dict
from pathlib import Path
import hashlib


def detect_language_by_course(course_code: str) -> str:
    """
    根据课程代码判断语言

    规则:
    - IN/CS/EI等 → 英语
    - MA/PH等 → 德语 (TUM数学/物理通常德语授课)
    - 可以根据实际情况调整
    """
    english_prefixes = ['IN', 'CS', 'EI', 'MW']
    german_prefixes = ['MA', 'PH', 'CH']

    prefix = course_code[:2]

    if prefix in english_prefixes:
        return 'en'
    elif prefix in german_prefixes:
        return 'de'
    else:
        # 默认英语，但最好手动确认
        return 'en'


def split_into_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    将文本分块

    Args:
        text: 原始文本
        chunk_size: 每块大小(字符数)
        overlap: 块之间的重叠(字符数)

    Returns:
        文本块列表
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # 尝试在句子边界处分割
        if end < len(text):
            # 查找最后一个句号、问号或换行符
            for separator in ['. ', '? ', '! ', '\n']:
                last_sep = chunk.rfind(separator)
                if last_sep > chunk_size * 0.7:  # 至少包含70%的内容
                    chunk = text[start:start + last_sep + len(separator)]
                    break

        chunks.append(chunk.strip())
        start = start + len(chunk) - overlap

    return chunks


def process_course_folder(
    course_folder: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[Dict]:
    """
    处理课程文件夹，提取所有文档并分块

    Args:
        course_folder: 课程文件夹路径
        chunk_size: 分块大小
        overlap: 重叠大小

    Returns:
        文档列表
    """
    documents = []
    course_path = Path(course_folder)

    # 假设文件夹结构:
    # courses/
    #   IN2064_MachineLearning/
    #     syllabus.txt
    #     lecture_1.txt
    #   MA1001_LinearAlgebra/
    #     skript.txt

    for course_dir in course_path.iterdir():
        if not course_dir.is_dir():
            continue

        # 提取课程代码
        course_code = course_dir.name.split('_')[0]
        course_name = course_dir.name

        # 检测语言
        lang = detect_language_by_course(course_code)

        print(f"Processing {course_name} (Language: {lang})")

        # 读取该课程的所有文本文件
        for file_path in course_dir.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # 分块
            chunks = split_into_chunks(text, chunk_size, overlap)

            # 为每个块创建文档
            for i, chunk in enumerate(chunks):
                # 生成唯一ID
                chunk_id = hashlib.md5(
                    f"{course_code}_{file_path.name}_{i}".encode()
                ).hexdigest()[:16]

                documents.append({
                    "id": chunk_id,
                    "text": chunk,
                    "course": course_code,
                    "lang": lang,
                    "metadata": {
                        "course_name": course_name,
                        "source_file": file_path.name,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })

    print(f"\n✅ Processed {len(documents)} chunks from courses")
    return documents


def save_documents(documents: List[Dict], output_file: str):
    """保存文档到JSONL文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')
    print(f"✅ Saved {len(documents)} documents to {output_file}")


def load_documents(input_file: str) -> List[Dict]:
    """从JSONL文件加载文档"""
    documents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            documents.append(json.loads(line))
    print(f"✅ Loaded {len(documents)} documents from {input_file}")
    return documents


# ===================== 使用示例 =====================

def example_prepare_corpus():
    """示例：准备语料库"""

    # 假设你的课程材料在这个文件夹
    course_folder = "./courses"

    # 如果文件夹不存在，创建示例数据
    if not os.path.exists(course_folder):
        print("⚠️  Course folder not found. Creating sample data...")
        create_sample_courses(course_folder)

    # 处理所有课程
    documents = process_course_folder(
        course_folder,
        chunk_size=512,
        overlap=50
    )

    # 保存到文件
    save_documents(documents, "tum_courses_corpus.jsonl")

    # 统计信息
    from collections import Counter
    lang_counts = Counter(doc['lang'] for doc in documents)
    course_counts = Counter(doc['course'] for doc in documents)

    print("\n📊 Corpus Statistics:")
    print(f"Total documents: {len(documents)}")
    print(f"Languages: {dict(lang_counts)}")
    print(f"Courses: {len(course_counts)}")
    print(f"  English courses: {sum(1 for c in course_counts if c.startswith('IN') or c.startswith('CS'))}")
    print(f"  German courses: {sum(1 for c in course_counts if c.startswith('MA') or c.startswith('PH'))}")


def create_sample_courses(base_folder: str):
    """创建示例课程数据用于测试"""
    os.makedirs(base_folder, exist_ok=True)

    # 英语课程样本
    english_courses = {
        "IN2064_MachineLearning": """
Machine Learning (IN2064) - Winter Semester

Course Description:
This course provides a comprehensive introduction to machine learning, covering both theoretical foundations and practical applications.

Topics Covered:
1. Supervised Learning
   - Linear Regression
   - Logistic Regression
   - Support Vector Machines
   - Decision Trees

2. Unsupervised Learning
   - K-Means Clustering
   - Principal Component Analysis (PCA)
   - Autoencoders

3. Reinforcement Learning
   - Markov Decision Processes
   - Q-Learning
   - Policy Gradients

Prerequisites:
- Linear Algebra (MA1001)
- Probability Theory (MA2009)
- Python Programming

Credits: 8 ECTS
Instructor: Prof. Dr. Smith
""",
        "IN2062_DeepLearning": """
Deep Learning (IN2062) - Summer Semester

Course Description:
Advanced course on deep neural networks and modern architectures.

Topics:
1. Neural Network Fundamentals
2. Convolutional Neural Networks (CNNs)
3. Recurrent Neural Networks (RNNs)
4. Transformers and Attention Mechanisms
5. Generative Models (GANs, VAEs)

Prerequisites:
- Machine Learning (IN2064)
- Strong Python skills

Credits: 6 ECTS
Instructor: Prof. Dr. Johnson
"""
    }

    # 德语课程样本
    german_courses = {
        "MA1001_LineareAlgebra": """
Lineare Algebra (MA1001) - Wintersemester

Kursbeschreibung:
Einführung in die lineare Algebra mit Anwendungen in der Informatik und Physik.

Themen:
1. Vektorräume
2. Lineare Abbildungen
3. Matrizen und Determinanten
4. Eigenwerte und Eigenvektoren
5. Skalarprodukt und Orthogonalität

Voraussetzungen:
Keine (Grundkurs)

Credits: 8 ECTS
Dozent: Prof. Dr. Müller
""",
        "MA2009_Wahrscheinlichkeitstheorie": """
Wahrscheinlichkeitstheorie (MA2009) - Sommersemester

Kursbeschreibung:
Grundlagen der Wahrscheinlichkeitstheorie und Statistik.

Themen:
1. Zufallsvariablen
2. Verteilungen (Normal, Binomial, Poisson)
3. Erwartungswert und Varianz
4. Grenzwertsätze
5. Stochastische Prozesse

Voraussetzungen:
- Analysis I und II

Credits: 8 ECTS
Dozent: Prof. Dr. Schmidt
"""
    }

    # 创建文件
    for course_name, content in {**english_courses, **german_courses}.items():
        course_path = Path(base_folder) / course_name
        course_path.mkdir(exist_ok=True)

        with open(course_path / "syllabus.txt", 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"✅ Created {len(english_courses) + len(german_courses)} sample courses in {base_folder}")


if __name__ == "__main__":
    example_prepare_corpus()
