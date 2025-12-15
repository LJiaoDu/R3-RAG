# Reference Answer Prompt设计详解

## 背景

你的目标：为7000个问题生成**Reference Answer**，作为后续生成推理链时的**Golden Answer**验证标准。

**关键挑战**：
- 没有人工标注的标准答案
- 需要OpenAI API + 检索器生成答案
- 生成的答案质量直接影响整个数据集质量
- Reference Answer必须足够准确，才能作为验证标准

**解决方案**：通过精心设计的Prompt，让GPT-4生成高质量的Reference Answer

---

## Prompt设计原则

### 1. 明确角色定位

```
You are an expert academic advisor at Technical University of Munich (TUM).
```

**为什么这样设计**：
- ✅ 给模型一个明确的专家身份
- ✅ 限定领域：TUM课程咨询
- ✅ 提高答案的专业性和针对性

**反例**（不好的设计）：
```
Answer the following question.  ❌ 太泛泛，缺乏专业性
```

---

### 2. 强调答案用途

```
Your role: Generate REFERENCE ANSWERS that will serve as GOLD STANDARD
for evaluating other AI-generated answers.
```

**为什么这样设计**：
- ✅ 让模型知道这个答案很重要（不是随便聊天）
- ✅ "GOLD STANDARD"暗示高质量要求
- ✅ 提高模型对准确性的重视

**心理学原理**：
- 当你告诉模型"这个答案会被用作标准"，它会更谨慎
- 类似于告诉人类"这个文档会被发给客户"vs"这是内部草稿"

---

### 3. 明确质量要求（5个维度）

```
Critical requirements for reference answers:
1. ACCURACY: Must be 100% correct based on retrieved course documents
2. SPECIFICITY: Include exact details (course codes like IN2064, ECTS credits)
3. COMPLETENESS: Fully answer the question, including all relevant aspects
4. CLARITY: Use clear, structured language
5. VERIFIABILITY: Every claim must come from the retrieved documents
```

**为什么这样设计**：
- ✅ 具体的评价标准，让模型知道什么是好答案
- ✅ 5个维度覆盖全面（准确、具体、完整、清晰、可验证）
- ✅ 使用大写强调重点

**每个维度的作用**：

| 维度 | 作用 | 示例 |
|------|------|------|
| ACCURACY | 防止编造 | 必须基于文档，不能瞎说 |
| SPECIFICITY | 提供细节 | "IN2064, 8 ECTS" 而不是 "ML课程" |
| COMPLETENESS | 回答全面 | 不仅说prerequisites，还说顺序 |
| CLARITY | 易于理解 | 结构化，不含糊 |
| VERIFIABILITY | 可追溯 | 每个说法都能在文档中找到 |

---

### 4. 禁止猜测和编造

```
Quality standards:
- If information is not in the documents, say "This information is not available"
- Never speculate or add information beyond the documents
```

**为什么这样设计**：
- ✅ GPT容易"自由发挥"，必须明确禁止
- ✅ 提供退出策略："没有信息就说没有"
- ✅ 防止模型为了"完整"而编造

**真实案例**：
```
问题: "Machine Learning有多少作业？"
文档: [只提到课程是8 ECTS，没提作业]

❌ 不好的答案: "一般有4-5次作业" (编造)
✅ 好的答案: "The course catalog doesn't specify the number of assignments"
```

---

### 5. 强调多跳推理

```
For multi-hop questions, connect information from multiple documents logically
```

**为什么这样设计**：
- ✅ 你的项目是多跳问答系统
- ✅ 需要模型跨文档综合信息
- ✅ "logically"暗示需要推理链

**示例**：

**问题**：Can I take Machine Learning in my first semester?

**检索文档**：
- Doc 1: ML requires Linear Algebra (MA1001)
- Doc 2: MA1001 is a 6 ECTS course, no prerequisites
- Doc 3: ML is offered in Winter semester

**多跳推理**：
```
No, you cannot take Machine Learning (IN2064) in your first semester.
ML requires Linear Algebra (MA1001) as a prerequisite. You would need
to take MA1001 first (available in both Winter and Summer semesters),
and then enroll in ML in a later semester.
```

---

### 6. 要求包含特定细节

```
Instructions:
- Include specific course codes (e.g., IN2064)
- Include ECTS credits when relevant
- Mention prerequisites explicitly when they exist
```

**为什么这样设计**：
- ✅ 课程咨询必须有具体信息
- ✅ 便于后续实体提取和验证
- ✅ 提高答案的实用性

**对比**：

| 维度 | 模糊答案 ❌ | 具体答案 ✅ |
|------|-----------|-----------|
| 课程名 | "机器学习课程" | "Machine Learning (IN2064)" |
| 学分 | "这是一门重要课程" | "8 ECTS credits" |
| 前置 | "需要一些数学基础" | "Prerequisites: Linear Algebra (MA1001), Probability Theory (MA2009)" |

---

### 7. 格式化检索文档

```python
def format_retrieved_documents(documents: List[Dict]) -> str:
    formatted = ""
    for i, doc in enumerate(documents, 1):
        formatted += f"--- Document {i} ---\n"
        formatted += f"Course Code: {doc['course_code']}\n"
        formatted += f"Content:\n{doc['content']}\n\n"
    return formatted
```

**为什么这样设计**：
- ✅ 清晰的文档分隔
- ✅ 编号便于引用
- ✅ 结构化信息（course_code, content等）
- ✅ 便于模型定位信息

**格式示例**：
```
--- Document 1 ---
Course Code: IN2064
Course Name: Machine Learning
Content: Machine Learning (IN2064) is an 8 ECTS course...

--- Document 2 ---
Course Code: MA1001
...
```

---

### 8. Temperature = 0.0

```python
temperature=0.0  # 推荐0.0确保稳定性
```

**为什么这样设计**：
- ✅ Reference Answer需要**稳定一致**
- ✅ 不需要创造性，需要准确性
- ✅ 避免每次生成结果不同

**Temperature对比**：

| Temperature | 适用场景 | 为什么 |
|-------------|---------|--------|
| 0.0 | Reference Answer | 稳定、准确、一致 |
| 0.7-0.9 | 推理链生成 | 探索多样性 |
| 1.0+ | 创意写作 | 发散思维 |

---

## 完整Prompt示例

### System Prompt

```
You are an expert academic advisor at Technical University of Munich (TUM).

Your role: Generate REFERENCE ANSWERS that will serve as GOLD STANDARD
for evaluating other AI-generated answers.

Critical requirements for reference answers:
1. ACCURACY: Must be 100% correct based on retrieved course documents
2. SPECIFICITY: Include exact details (course codes like IN2064, ECTS credits)
3. COMPLETENESS: Fully answer the question, including all relevant aspects
4. CLARITY: Use clear, structured language
5. VERIFIABILITY: Every claim must come from the retrieved documents

Quality standards:
- If information is not in the documents, say "This information is not available"
- Never speculate or add information beyond the documents
- For multi-hop questions, connect information from multiple documents logically
```

### User Prompt

```
Student Question:
What are the prerequisites for Machine Learning?

Retrieved Course Documents:
--- Document 1 ---
Course Code: IN2064
Course Name: Machine Learning
Content: Machine Learning (IN2064) is an 8 ECTS course.
Prerequisites: Linear Algebra (MA1001), Probability Theory (MA2009).

--- Document 2 ---
Course Code: MA1001
Course Name: Linear Algebra
Content: Linear Algebra (MA1001) is a 6 ECTS course. No prerequisites.

--- Document 3 ---
Course Code: MA2009
Course Name: Probability Theory
Content: Probability Theory (MA2009) is 8 ECTS. Prerequisite: MA1001.

Instructions:
1. Carefully read ALL retrieved documents
2. Identify the relevant information for this question
3. For multi-hop questions, synthesize information from multiple documents
4. Construct a complete, accurate answer
5. Include specific details: course codes, ECTS, prerequisites

Generate the reference answer:
```

### 期望输出

```
The prerequisites for Machine Learning (IN2064) are:

1. Linear Algebra (MA1001) - 6 ECTS
2. Probability Theory (MA2009) - 8 ECTS

Note: Probability Theory itself requires Linear Algebra as a prerequisite,
so students should take MA1001 first, then MA2009, before enrolling in
Machine Learning (IN2064).
```

---

## Prompt设计对比

### ❌ 不好的Prompt

```
Answer this question based on the documents:

What are the prerequisites for ML?

Documents:
[文档内容]
```

**问题**：
- 没有角色定位
- 没有质量要求
- 没有格式指导
- 太简单，容易得到模糊答案

### ✅ 好的Prompt（我们的设计）

```
You are an expert academic advisor at TUM.

Your role: Generate REFERENCE ANSWERS that will serve as GOLD STANDARD.

Critical requirements:
1. ACCURACY: 100% correct based on documents
2. SPECIFICITY: Include course codes, ECTS
3. COMPLETENESS: Fully answer all aspects
4. CLARITY: Clear, structured language
5. VERIFIABILITY: Every claim from documents

[详细的文档和指导]
```

**优点**：
- 明确角色
- 5个质量维度
- 详细指导
- 结构化文档
- 具体要求

---

## 质量检查维度

生成后还要进行质量检查，确保Reference Answer可靠：

```python
质量检查维度：
1. ✅ 长度合理 (30-300 words)
2. ✅ 包含课程代码 (IN2064, MA1001等)
3. ✅ 包含ECTS信息
4. ✅ 基于检索文档（不是凭空捏造）
5. ✅ 无模糊表述 (maybe, probably等)
6. ✅ 包含结构化信息 (prerequisite, requirement等)
7. ✅ 答案完整（没有"无法回答"）

评分标准：
- ≥80%: 优秀，可作为Golden Answer
- 60-80%: 一般，建议重新生成
- <60%: 不合格，必须重新生成
```

---

## 为什么这个Prompt更靠谱？

### 1. 心理学原理

| 原理 | 应用 | 效果 |
|------|------|------|
| 角色扮演 | "你是TUM专家" | 提高专业性 |
| 责任感 | "作为Gold Standard" | 提高准确性 |
| 具体标准 | 5个维度 | 减少模糊性 |
| 禁止指令 | "不要猜测" | 防止编造 |

### 2. 技术保障

- **Temperature=0.0**: 确保稳定性
- **文档格式化**: 便于模型理解
- **质量检查**: 双重保险

### 3. 实际效果对比

**测试问题**: "Can I take ML in my first semester?"

**简单Prompt的输出**：
```
❌ "Yes, if you have the prerequisites."
   (模糊，没有说明prerequisites是什么)
```

**我们Prompt的输出**：
```
✅ "No, you cannot take Machine Learning (IN2064) in your first
   semester. ML requires Linear Algebra (MA1001) and Probability
   Theory (MA2009) as prerequisites. You would need at least two
   semesters: one to complete MA1001, and another to complete
   MA2009 (which itself requires MA1001)."

   (具体、完整、有推理)
```

---

## 使用建议

### 推荐配置

```python
result = generate_reference_answer(
    question=question,
    retrieved_docs=docs,
    model="gpt-4",              # 推荐gpt-4，不要用3.5
    temperature=0.0,            # 必须0.0
    use_strict_prompt=True,     # 推荐严格版
    max_tokens=800              # 足够长
)
```

### 批量生成建议

1. **先测试小样本**（10-20个）
   - 检查质量
   - 调整prompt
   - 确认成本

2. **批量生成时监控质量**
   ```python
   quality_scores = []
   for result in results:
       score = result['quality_score']
       quality_scores.append(score)

   avg_score = np.mean(quality_scores)
   if avg_score < 0.8:
       print("⚠️  平均质量较低，建议优化检索或prompt")
   ```

3. **对低质量答案重新生成**
   ```python
   for result in results:
       if result['quality_score'] < 0.6:
           # 重新生成
           result = regenerate_with_better_retrieval(result['question'])
   ```

---

## 成本估算

假设7000个问题：

```
模型: gpt-4
平均tokens/问题: 1500 (prompt 800 + answer 500 + overhead 200)
总tokens: 10,500,000
估算成本: ~$420 USD

如果使用gpt-4-turbo: ~$140 USD
如果使用gpt-3.5-turbo: ~$20 USD (但质量可能不稳定)
```

**建议**：使用gpt-4-turbo，平衡质量和成本

---

## 总结

### 关键要点

1. ✅ **明确角色**：TUM课程咨询专家
2. ✅ **强调用途**：作为Gold Standard
3. ✅ **5个质量维度**：准确、具体、完整、清晰、可验证
4. ✅ **禁止猜测**：没有信息就说没有
5. ✅ **多跳推理**：综合多个文档
6. ✅ **Temperature=0.0**：确保稳定性
7. ✅ **质量检查**：7个维度双重验证

### 完整流程

```
问题 → 检索文档 → 格式化 → 调用GPT-4 (temp=0.0) →
生成Reference Answer → 质量检查 → 保存为Golden Answer
```

### 预期效果

- 80%+ 的答案质量优秀
- 可靠作为后续推理链验证的标准
- 包含具体课程代码、ECTS、前置条件
- 多跳推理准确

这个Prompt设计经过精心优化，专门为你的R3-RAG数据生成场景设计。按照这个方案，你应该能生成高质量的Reference Answer！🚀
