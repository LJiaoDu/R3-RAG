#!/usr/bin/env python3
"""
端到端多语言RAG系统
完整示例：从语料库准备到问答生成
"""

from multilingual_retriever import MultilingualRetriever
from prepare_corpus import process_course_folder, save_documents, load_documents
from openai import OpenAI
from langdetect import detect


class MultilingualRAGSystem:
    """
    多语言RAG系统

    功能:
    1. 多语言检索 (英语+德语)
    2. 自动语言检测
    3. 生成同语言答案
    """

    def __init__(
        self,
        retriever: MultilingualRetriever,
        llm_api_url: str = "https://api.deepseek.com/beta",
        llm_api_key: str = "your_api_key",
        llm_model: str = "deepseek-chat"
    ):
        """
        初始化RAG系统

        Args:
            retriever: 多语言检索器
            llm_api_url: LLM API地址
            llm_api_key: API密钥
            llm_model: 模型名称
        """
        self.retriever = retriever

        # 初始化LLM客户端
        self.llm_client = OpenAI(
            api_key=llm_api_key,
            base_url=llm_api_url
        )
        self.llm_model = llm_model

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        use_smart_search: bool = True
    ) -> Dict:
        """
        回答问题

        Args:
            question: 用户问题
            top_k: 检索文档数量
            use_smart_search: 是否使用智能检索(优先同语言)

        Returns:
            回答结果字典
        """
        # 1. 检测问题语言
        try:
            question_lang = detect(question)
            if question_lang not in ['en', 'de']:
                question_lang = 'en'  # 默认英语
        except:
            question_lang = 'en'

        print(f"🔍 Question language detected: {question_lang}")

        # 2. 检索相关文档
        if use_smart_search:
            retrieved_docs = self.retriever.smart_search(
                question,
                top_k=top_k,
                same_lang_ratio=0.7  # 70%同语言
            )
        else:
            retrieved_docs = self.retriever.search(
                question,
                top_k=top_k,
                auto_detect_lang=True
            )

        print(f"📚 Retrieved {len(retrieved_docs)} documents")

        # 3. 格式化检索结果
        context = self._format_retrieved_docs(retrieved_docs)

        # 4. 构建Prompt
        prompt = self._build_prompt(question, context, question_lang)

        # 5. 调用LLM生成答案
        print(f"🤖 Generating answer using {self.llm_model}...")
        response = self.llm_client.completions.create(
            model=self.llm_model,
            prompt=prompt,
            max_tokens=512,
            temperature=0.7
        )

        answer = response.choices[0].text.strip()

        # 6. 返回结果
        return {
            "question": question,
            "question_lang": question_lang,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "num_docs": len(retrieved_docs)
        }

    def _format_retrieved_docs(self, docs: List[Dict]) -> str:
        """格式化检索到的文档"""
        formatted = []
        for i, doc in enumerate(docs):
            formatted.append(
                f"[Document {i+1}] (Course: {doc['course']}, Language: {doc['lang']})\n"
                f"{doc['text']}\n"
            )
        return "\n".join(formatted)

    def _build_prompt(
        self,
        question: str,
        context: str,
        question_lang: str
    ) -> str:
        """
        构建Prompt

        关键:
        - 如果问题是英语，答案必须是英语
        - 如果问题是德语，答案必须是德语
        """
        if question_lang == 'de':
            # 德语Prompt
            prompt = f"""Du bist ein TUM-Studienberater. Beantworte die Frage des Studenten basierend auf den bereitgestellten Kursinformationen.

WICHTIG:
- Antworte IMMER auf Deutsch
- Beziehe dich nur auf die bereitgestellten Dokumente
- Wenn die Informationen nicht ausreichen, sage es ehrlich
- Gib Kursscodes an, wenn relevant (z.B. IN2064, MA1001)

Bereitgestellte Kursinformationen:
{context}

Frage: {question}

Antwort (auf Deutsch):"""

        else:
            # 英语Prompt
            prompt = f"""You are a TUM course advisor assistant. Answer the student's question based on the provided course information.

IMPORTANT:
- ALWAYS respond in English
- Only use information from the provided documents
- If information is insufficient, say so honestly
- Include course codes when relevant (e.g., IN2064, MA1001)

Retrieved Course Information:
{context}

Question: {question}

Answer (in English):"""

        return prompt


# ===================== 完整流程示例 =====================

def complete_workflow_example():
    """
    完整工作流程示例
    从语料库准备到问答生成
    """

    print("="*60)
    print("完整多语言RAG系统演示")
    print("="*60)

    # ========== 步骤1: 准备语料库 ==========
    print("\n📚 Step 1: Preparing corpus...")

    # 处理课程文件夹 (如果存在)
    import os
    if os.path.exists("./courses"):
        documents = process_course_folder("./courses")
        save_documents(documents, "tum_courses.jsonl")
    else:
        # 使用示例数据
        print("⚠️  Using sample data...")
        from prepare_corpus import create_sample_courses
        create_sample_courses("./courses")
        documents = process_course_folder("./courses")
        save_documents(documents, "tum_courses.jsonl")

    # ========== 步骤2: 初始化检索器 ==========
    print("\n🔧 Step 2: Initializing retriever...")

    retriever = MultilingualRetriever(
        embedding_model="intfloat/multilingual-e5-base",
        collection_name="tum_courses",
        qdrant_path="./qdrant_tum"
    )

    # 添加文档
    retriever.add_documents(documents)

    # ========== 步骤3: 初始化RAG系统 ==========
    print("\n🤖 Step 3: Initializing RAG system...")

    rag_system = MultilingualRAGSystem(
        retriever=retriever,
        llm_api_url="https://api.deepseek.com/beta",  # 改成你的API
        llm_api_key="your_api_key",                   # 改成你的key
        llm_model="deepseek-chat"
    )

    # ========== 步骤4: 测试问答 ==========
    print("\n💬 Step 4: Testing Q&A...")

    test_questions = [
        # 英语问题
        "What are the prerequisites for Machine Learning?",
        "How many ECTS credits is Deep Learning?",

        # 德语问题
        "Was sind die Voraussetzungen für Lineare Algebra?",
        "Wie viele ECTS hat Wahrscheinlichkeitstheorie?",

        # 跨语言问题
        "What topics are covered in Wahrscheinlichkeitstheorie?",
        "Wer ist der Dozent für Machine Learning?"
    ]

    for i, question in enumerate(test_questions):
        print(f"\n{'='*60}")
        print(f"Question {i+1}: {question}")
        print('='*60)

        result = rag_system.answer_question(question, top_k=3)

        print(f"\n🌐 Detected Language: {result['question_lang']}")
        print(f"📚 Used {result['num_docs']} documents")
        print(f"\n💡 Answer:\n{result['answer']}")

        print(f"\n📄 Retrieved Documents:")
        for j, doc in enumerate(result['retrieved_docs']):
            print(f"  [{j+1}] {doc['course']} ({doc['lang']}) - Score: {doc['score']:.3f}")


# ===================== 简化版使用 =====================

def simple_usage():
    """简化版使用示例（假设语料库已准备好）"""

    # 1. 加载已有文档
    documents = load_documents("tum_courses.jsonl")

    # 2. 初始化检索器并添加文档
    retriever = MultilingualRetriever()
    retriever.add_documents(documents)

    # 3. 初始化RAG系统
    rag = MultilingualRAGSystem(
        retriever=retriever,
        llm_api_url="your_llm_api",
        llm_api_key="your_key"
    )

    # 4. 提问
    result = rag.answer_question("What are the prerequisites for Machine Learning?")
    print(result['answer'])


if __name__ == "__main__":
    # 运行完整工作流程
    complete_workflow_example()

    # 或者使用简化版
    # simple_usage()
