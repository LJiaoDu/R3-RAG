#!/usr/bin/env python3
"""
多语言检索器实现
支持英语+德语课程语料库
"""

import os
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langdetect import detect


class MultilingualRetriever:
    """
    多语言检索器

    支持:
    - 20门英语课程
    - 10门德语课程
    - 跨语言检索
    - 语言偏好过滤
    """

    def __init__(
        self,
        embedding_model: str = "intfloat/multilingual-e5-large",
        collection_name: str = "tum_courses",
        qdrant_path: str = "./qdrant_storage"
    ):
        """
        初始化检索器

        Args:
            embedding_model: 多语言Embedding模型名称
            collection_name: Qdrant集合名称
            qdrant_path: Qdrant存储路径
        """
        print(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"Connecting to Qdrant: {qdrant_path}")
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name

        # 如果集合不存在，创建它
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            print(f"Creating collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )

    def add_documents(self, documents: List[Dict]):
        """
        添加文档到语料库

        Args:
            documents: 文档列表，每个文档包含:
                {
                    "id": "unique_id",
                    "text": "文档内容",
                    "course": "课程代码",
                    "lang": "en" or "de",
                    "metadata": {...}  # 其他元数据
                }
        """
        print(f"Generating embeddings for {len(documents)} documents...")

        # 批量生成Embeddings
        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)

        # 准备Qdrant数据点
        points = []
        for i, doc in enumerate(documents):
            payload = {
                "text": doc["text"],
                "course": doc["course"],
                "lang": doc["lang"]
            }
            # 添加其他元数据
            if "metadata" in doc:
                payload.update(doc["metadata"])

            points.append(
                PointStruct(
                    id=doc["id"] if isinstance(doc["id"], int) else hash(doc["id"]) % (2**63),
                    vector=embeddings[i].tolist(),
                    payload=payload
                )
            )

        # 批量插入
        print(f"Inserting {len(points)} points into Qdrant...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print("✅ Documents added successfully!")

    def search(
        self,
        query: str,
        top_k: int = 5,
        preferred_lang: Optional[str] = None,
        auto_detect_lang: bool = False
    ) -> List[Dict]:
        """
        检索文档

        Args:
            query: 查询问题
            top_k: 返回Top-K结果
            preferred_lang: 偏好语言 ('en', 'de', 或 None)
            auto_detect_lang: 是否自动检测问题语言并优先返回同语言文档

        Returns:
            检索结果列表
        """
        # 自动检测语言
        if auto_detect_lang and not preferred_lang:
            try:
                detected_lang = detect(query)
                if detected_lang in ['en', 'de']:
                    preferred_lang = detected_lang
                    print(f"🔍 Detected language: {detected_lang}")
            except:
                pass

        # 生成查询向量
        query_vector = self.model.encode(query).tolist()

        # 构建过滤条件
        query_filter = None
        if preferred_lang:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="lang",
                        match=MatchValue(value=preferred_lang)
                    )
                ]
            )

        # 检索
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k
        )

        # 格式化结果
        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result.payload["text"],
                "course": result.payload["course"],
                "lang": result.payload["lang"],
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items()
                           if k not in ["text", "course", "lang"]}
            })

        return formatted_results

    def smart_search(
        self,
        query: str,
        top_k: int = 5,
        same_lang_ratio: float = 0.7
    ) -> List[Dict]:
        """
        智能检索：优先返回同语言文档，但也包含其他语言相关文档

        Args:
            query: 查询问题
            top_k: 总共返回多少结果
            same_lang_ratio: 同语言文档的比例 (0-1)

        Returns:
            混合语言的检索结果
        """
        # 检测问题语言
        try:
            query_lang = detect(query)
            if query_lang not in ['en', 'de']:
                query_lang = None
        except:
            query_lang = None

        if not query_lang:
            # 无法检测语言，直接跨语言检索
            return self.search(query, top_k=top_k)

        # 计算同语言和其他语言的数量
        same_lang_k = int(top_k * same_lang_ratio)
        other_lang_k = top_k - same_lang_k

        # 同语言检索
        same_lang_results = self.search(
            query,
            top_k=same_lang_k,
            preferred_lang=query_lang
        )

        # 其他语言检索
        other_lang_results = self.search(
            query,
            top_k=other_lang_k,
            preferred_lang=None  # 不限语言
        )

        # 去重（避免重复文档）
        seen_texts = {r["text"] for r in same_lang_results}
        unique_other = [r for r in other_lang_results if r["text"] not in seen_texts]

        # 合并结果
        return same_lang_results + unique_other[:other_lang_k]


# ===================== 使用示例 =====================

def example_usage():
    """示例：如何使用多语言检索器"""

    # 1. 初始化检索器
    retriever = MultilingualRetriever(
        embedding_model="intfloat/multilingual-e5-base",  # 用base版本更快
        collection_name="tum_courses_demo",
        qdrant_path="./qdrant_demo"
    )

    # 2. 准备示例文档（模拟30门课程）
    documents = [
        # 英语课程 (20门)
        {
            "id": "IN2064_chunk_1",
            "text": "Machine Learning (IN2064) covers supervised learning, unsupervised learning, and reinforcement learning. Prerequisites: Linear Algebra, Probability Theory.",
            "course": "IN2064",
            "lang": "en",
            "metadata": {"semester": "WS", "credits": 8}
        },
        {
            "id": "IN2062_chunk_1",
            "text": "Deep Learning (IN2062) focuses on neural networks, CNNs, RNNs, and transformers. Prerequisites: Machine Learning (IN2064), Python programming.",
            "course": "IN2062",
            "lang": "en",
            "metadata": {"semester": "SS", "credits": 6}
        },
        {
            "id": "IN2359_chunk_1",
            "text": "Computer Vision (IN2359) teaches image processing, object detection, and semantic segmentation using deep learning techniques.",
            "course": "IN2359",
            "lang": "en",
            "metadata": {"semester": "WS", "credits": 6}
        },

        # 德语课程 (10门)
        {
            "id": "MA1001_chunk_1",
            "text": "Lineare Algebra (MA1001) behandelt Vektorräume, lineare Abbildungen, Eigenwerte und Eigenvektoren. Voraussetzungen: Keine.",
            "course": "MA1001",
            "lang": "de",
            "metadata": {"semester": "WS", "credits": 8}
        },
        {
            "id": "MA2009_chunk_1",
            "text": "Wahrscheinlichkeitstheorie (MA2009) umfasst Zufallsvariablen, Verteilungen, Erwartungswert und Grenzwertsätze. Voraussetzungen: Analysis.",
            "course": "MA2009",
            "lang": "de",
            "metadata": {"semester": "SS", "credits": 8}
        },
        {
            "id": "PH0001_chunk_1",
            "text": "Experimentalphysik I (PH0001) behandelt klassische Mechanik, Schwingungen und Wellen. Voraussetzungen: Abitur Physik.",
            "course": "PH0001",
            "lang": "de",
            "metadata": {"semester": "WS", "credits": 8}
        },
    ]

    # 3. 添加文档到语料库
    retriever.add_documents(documents)

    # 4. 测试检索
    print("\n" + "="*60)
    print("测试1: 英语问题 → 英语文档")
    print("="*60)
    results = retriever.search(
        "What are the prerequisites for Deep Learning?",
        top_k=3,
        auto_detect_lang=True
    )
    for i, r in enumerate(results):
        print(f"\n[{i+1}] Course: {r['course']} | Lang: {r['lang']} | Score: {r['score']:.3f}")
        print(f"Text: {r['text'][:100]}...")

    print("\n" + "="*60)
    print("测试2: 德语问题 → 德语文档")
    print("="*60)
    results = retriever.search(
        "Was sind die Voraussetzungen für Wahrscheinlichkeitstheorie?",
        top_k=3,
        auto_detect_lang=True
    )
    for i, r in enumerate(results):
        print(f"\n[{i+1}] Course: {r['course']} | Lang: {r['lang']} | Score: {r['score']:.3f}")
        print(f"Text: {r['text'][:100]}...")

    print("\n" + "="*60)
    print("测试3: 跨语言检索 (英语问题 → 德语文档)")
    print("="*60)
    results = retriever.search(
        "What topics are covered in Experimentalphysik?",
        top_k=3
    )
    for i, r in enumerate(results):
        print(f"\n[{i+1}] Course: {r['course']} | Lang: {r['lang']} | Score: {r['score']:.3f}")
        print(f"Text: {r['text'][:100]}...")

    print("\n" + "="*60)
    print("测试4: 智能混合检索")
    print("="*60)
    results = retriever.smart_search(
        "What are prerequisites for advanced math courses?",
        top_k=5,
        same_lang_ratio=0.6  # 60%同语言，40%其他语言
    )
    for i, r in enumerate(results):
        print(f"\n[{i+1}] Course: {r['course']} | Lang: {r['lang']} | Score: {r['score']:.3f}")
        print(f"Text: {r['text'][:100]}...")


if __name__ == "__main__":
    example_usage()
