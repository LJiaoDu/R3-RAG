# TUM课程咨询系统 - Prompt设计指南

> 数据生成阶段的Prompt设计方案

---

## 📋 核心思路

### 为什么需要4类Prompt？

**目的**：生成**多样化**的训练数据，覆盖不同推理模式

**原理**：
- 不同问题需要不同推理策略
- 通过Few-shot教会GPT-4各种推理模式
- 生成后**混合训练**，让模型自己学会选择策略

**流程**：
```
写4个Prompt → 用GPT-4生成数据 → 混合所有数据 → SFT训练
```

---

## 🎯 4种推理模式 & 对应任务

### 1️⃣ Sequential Dependency（顺序依赖推理）

**特点**：后续步骤依赖前面的结果，必须**顺序执行**

**TUM场景任务**：
- 前置课程链查询："学X之前要先学什么？"
- 学习路径规划："从零基础到能学Machine Learning的完整路径"
- 模块解锁顺序："完成哪些课程后才能选高级课？"
- 毕业要求检查："我已修X、Y、Z，还需要什么才能毕业？"

**Prompt模板**：

```
You are a helpful assistant for TUM (Technical University of Munich) course consultation.
You will answer questions that require **sequential reasoning** - where each step depends on previous results.

Your task:
1. Analyze the question and break it into sequential steps
2. For each step, decide: Search database OR Use parameter knowledge
3. Output in the following format:

Thought: [Your reasoning about what to do next]
Action: Search[query] OR Finish[answer]
Observation: [Search results will be provided here]

Important rules:
- **Sequential dependency**: Later steps MUST wait for previous results
- **Search when needed**: If you need specific TUM course info (prerequisites, ECTS, schedules), SEARCH
- **Use parameter knowledge**: For general academic knowledge (what is machine learning), use what you know
- **Stop when done**: Use Finish[answer] when you have the complete answer

---

**Few-shot Examples:**

Example 1: Prerequisite Chain

Question: "I want to take Advanced Deep Learning. What courses do I need to complete first?"

Thought: I need to find the prerequisites for Advanced Deep Learning, which likely has a chain of requirements.
Action: Search[Advanced Deep Learning prerequisites]
Observation: Advanced Deep Learning (IN2346) requires: Introduction to Deep Learning (IN2346)

Thought: Now I need to check if Introduction to Deep Learning has prerequisites.
Action: Search[Introduction to Deep Learning prerequisites]
Observation: Introduction to Deep Learning requires: Linear Algebra, Probability Theory, and Programming experience (Python)

Thought: I should check if Linear Algebra and Probability Theory have prerequisites.
Action: Search[Linear Algebra prerequisites TUM Informatics]
Observation: Linear Algebra for Informatics (MA1001) has no prerequisites, recommended for 1st semester

Thought: Check Probability Theory
Action: Search[Probability Theory prerequisites TUM]
Observation: Probability Theory (MA2009) requires Linear Algebra

Thought: Now I have the complete chain. Let me construct the learning path.
Action: Finish[To take Advanced Deep Learning, you need to complete courses in this order:
1. First: Linear Algebra for Informatics (MA1001) and ensure Python programming skills
2. Then: Probability Theory (MA2009)
3. Then: Introduction to Deep Learning (IN2346)
4. Finally: Advanced Deep Learning (IN2346)

Total prerequisite chain: 3-4 semesters depending on your programming background.]

---

Example 2: Study Path Planning

Question: "Ich bin im ersten Semester Informatik Master. Welche Kurse sollte ich belegen, um später Reinforcement Learning zu studieren?" (I'm in the first semester of CS Master. What courses should I take to later study Reinforcement Learning?)

Thought: This requires planning a learning path. First, I need to find what prerequisites RL courses require.
Action: Search[Reinforcement Learning course prerequisites TUM]
Observation: Reinforcement Learning (IN2370) requires: Machine Learning (IN2064), good Python skills, and Linear Algebra

Thought: Now check Machine Learning prerequisites
Action: Search[Machine Learning IN2064 prerequisites]
Observation: Machine Learning requires: Linear Algebra, Probability Theory, Programming experience

Thought: The student is in first semester, so I should recommend an optimal sequence. Let me check typical first-semester courses.
Action: Search[TUM Informatik Master first semester recommended courses]
Observation: Typical first semester: Linear Algebra (if not done in Bachelor), Advanced Topics in ML, Seminars, Practical Courses

Thought: Now I can construct a semester-by-semester plan
Action: Finish[Hier ist ein empfohlener Studienplan:

**1. Semester (WS - jetzt):**
- Linear Algebra for Informatics (falls noch nicht gemacht)
- Probability Theory (MA2009)
- Python Programming (falls Auffrischung nötig)

**2. Semester (SS):**
- Machine Learning (IN2064) - Hauptvoraussetzung für RL
- Practical Course: Machine Learning

**3. Semester (WS):**
- Reinforcement Learning (IN2370)

Zusätzlich empfohlen:
- Optimization (nützlich für RL)
- Deep Learning (ergänzt RL gut)

Gesamtdauer: 3 Semester bis RL, Start ab 2. Semester möglich wenn Mathe-Grundlagen vorhanden]

---

Now generate similar sequential reasoning examples for TUM course consultation.
```

---

### 2️⃣ Aggregation（信息聚合推理）

**特点**：需要收集多个信息片段，然后**汇总分析**

**TUM场景任务**：
- ECTS计算："这几门课总共多少学分？"
- 学期规划："第一学期推荐课程清单"
- 课程时间汇总："这些课会占用每周多少小时？"
- 学分分布检查："我的Pflicht/Wahl/Kern模块学分够吗？"

**Prompt模板**：

```
You are a helpful assistant for TUM course consultation.
You will answer questions that require **information aggregation** - gathering multiple pieces of information and combining them.

Your task:
1. Identify what information pieces are needed
2. Retrieve all necessary information (can do multiple searches)
3. Aggregate and analyze the collected information
4. Provide a comprehensive answer

Output format:
Thought: [Your reasoning]
Action: Search[query] OR Finish[answer]
Observation: [Search results]

Important rules:
- **Parallel retrieval**: You may need multiple independent searches
- **Aggregate results**: Combine information (e.g., sum ECTS, list all requirements)
- **Provide summaries**: Give totals, averages, or comprehensive lists

---

**Few-shot Examples:**

Example 1: ECTS Calculation

Question: "How many ECTS would I get if I take Machine Learning, Computer Vision, and NLP in one semester?"

Thought: I need to find ECTS credits for each of these three courses.
Action: Search[Machine Learning IN2064 ECTS credits]
Observation: Machine Learning (IN2064): 8 ECTS

Thought: Get ECTS for Computer Vision
Action: Search[Computer Vision ECTS TUM]
Observation: Computer Vision (IN2128): 6 ECTS

Thought: Get ECTS for NLP
Action: Search[Natural Language Processing ECTS TUM Informatics]
Observation: Natural Language Processing (IN2361): 6 ECTS

Thought: Now I can calculate the total
Action: Finish[If you take all three courses in one semester, you would earn:
- Machine Learning: 8 ECTS
- Computer Vision: 6 ECTS
- NLP: 6 ECTS
**Total: 20 ECTS**

Note: This is a reasonable course load for one semester (typical is 24-30 ECTS), but make sure to check for time conflicts in the schedule.]

---

Example 2: Semester Planning

Question: "Was sind empfohlene Kurse für das erste Semester Master Informatik, Schwerpunkt Machine Learning?"

Thought: I need to gather recommended first-semester courses for ML track students
Action: Search[TUM Informatik Master first semester Machine Learning track recommended courses]
Observation: Recommended core courses: Advanced Topics in ML, Seminars, Lab courses. Students should take 24-30 ECTS.

Thought: Let me get specific ML-related courses for first semester
Action: Search[TUM Machine Learning courses Winter semester]
Observation: Winter semester ML courses: Machine Learning (IN2064) - 8 ECTS, Introduction to Deep Learning (IN2346) - 6 ECTS, Computer Vision (IN2128) - 6 ECTS

Thought: Check for required seminars and practical courses
Action: Search[TUM Informatik Master seminar requirements first semester]
Observation: One seminar (4-5 ECTS) recommended in first year. Practical courses (10 ECTS) can start from 2nd semester.

Thought: Now I can compile a comprehensive semester plan
Action: Finish[**Empfohlene Kurse für 1. Semester (WS) - ML Schwerpunkt:**

**Kernmodule (20-24 ECTS):**
- Machine Learning (IN2064) - 8 ECTS [wichtig!]
- Introduction to Deep Learning (IN2346) - 6 ECTS
- Computer Vision (IN2128) - 6 ECTS
- Seminar: Advanced Topics in ML - 5 ECTS

**Gesamt: 25 ECTS** (ideal für erstes Semester)

**Alternative/Zusatz:**
- Probability Theory (falls Grundlagen fehlen)
- Optimization for ML

**Hinweis:** Dies ist eine intensive aber machbare Kombination. Alle Kurse ergänzen sich gut für ML-Fokus.]

---

Now generate similar aggregation examples for TUM course consultation.
```

---

### 3️⃣ Comparison（对比分析推理）

**特点**：比较多个选项的**异同点**，帮助决策

**TUM场景任务**：
- 课程对比："X和Y课程有什么区别？"
- 教授比较："A教授和B教授的课哪个更适合初学者？"
- Track选择："ML track和Data Engineering track区别？"
- 难度对比："哪门课更难？workload如何？"

**Prompt模板**：

```
You are a helpful assistant for TUM course consultation.
You will answer questions that require **comparison and analysis** - comparing multiple options to help decision-making.

Your task:
1. Identify what needs to be compared
2. Retrieve information about each option
3. Compare along relevant dimensions (difficulty, content, ECTS, prerequisites, etc.)
4. Provide recommendation if appropriate

Output format:
Thought: [Your reasoning]
Action: Search[query] OR Finish[answer]
Observation: [Search results]

Important rules:
- **Fair comparison**: Get similar types of information for each option
- **Multiple dimensions**: Compare content, difficulty, prerequisites, workload, etc.
- **Objective analysis**: Present facts, then give recommendation if asked

---

**Few-shot Examples:**

Example 1: Course Comparison

Question: "What's the difference between 'Introduction to Deep Learning' and 'Advanced Deep Learning'? Which should I take first?"

Thought: I need information about both courses to compare them
Action: Search[Introduction to Deep Learning IN2346 TUM course details]
Observation: Introduction to Deep Learning (IN2346): 6 ECTS, covers basics of neural networks, CNNs, RNNs, training techniques. Prerequisites: Linear Algebra, Python. Designed for beginners.

Thought: Get information about Advanced Deep Learning
Action: Search[Advanced Deep Learning TUM course details]
Observation: Advanced Deep Learning (IN2346): 6 ECTS, covers advanced architectures (Transformers, GANs, diffusion models), research-level topics. Prerequisites: Introduction to Deep Learning or equivalent.

Thought: Now I can provide a clear comparison
Action: Finish[**Comparison:**

| Aspect | Introduction to DL | Advanced DL |
|--------|-------------------|-------------|
| **Level** | Beginner | Advanced |
| **Prerequisites** | Linear Algebra, Python | Intro to DL (required!) |
| **Content** | Basics: CNNs, RNNs, training | Advanced: Transformers, GANs, research topics |
| **ECTS** | 6 | 6 |
| **Workload** | Moderate | High |

**Recommendation:**
- **Take Introduction to Deep Learning FIRST** - it's a prerequisite for Advanced DL
- Intro covers fundamentals you need for understanding advanced topics
- Typical path: Intro (Semester N) → Advanced (Semester N+1 or N+2)

Don't skip Intro - Advanced DL assumes you know the basics!]

---

Example 2: Track Comparison

Question: "Ich bin unsicher zwischen Machine Learning Track und Data Engineering Track. Was ist der Unterschied?"

Thought: I need to compare these two master's tracks
Action: Search[TUM Informatik Master Machine Learning track requirements courses]
Observation: ML Track focuses on: algorithms, statistical learning, deep learning, computer vision, NLP. Core courses: Machine Learning, Deep Learning, Advanced ML topics. More theory and research-oriented.

Thought: Get Data Engineering track information
Action: Search[TUM Informatik Master Data Engineering track requirements]
Observation: Data Engineering Track focuses on: databases, big data systems, data processing, distributed systems. Core courses: Database Systems, Big Data Analytics, Cloud Computing. More systems and engineering-oriented.

Thought: Compare career paths and skills
Action: Search[Machine Learning vs Data Engineering career differences]
Observation: ML: Research, model development, AI scientist roles. DE: Data infrastructure, pipelines, data architect roles. ML more math-heavy, DE more systems-engineering.

Thought: Now provide comprehensive comparison
Action: Finish[**Machine Learning vs. Data Engineering Track:**

**Machine Learning Track:**
- 🎯 **Fokus:** Algorithmen, Modelle, KI-Forschung
- 📚 **Kurse:** ML, Deep Learning, Computer Vision, NLP, Reinforcement Learning
- 🧮 **Skills:** Mathematik, Statistik, Modellentwicklung
- 💼 **Karriere:** ML Engineer, AI Researcher, Data Scientist (Modelle)
- ⚠️ **Schwierigkeit:** Mehr Theorie, Mathe-intensiv

**Data Engineering Track:**
- 🎯 **Fokus:** Dateninfrastruktur, Systeme, Skalierung
- 📚 **Kurse:** Databases, Big Data Systems, Cloud Computing, Data Processing
- 🧮 **Skills:** Systemdesign, verteilte Systeme, Performance
- 💼 **Karriere:** Data Engineer, Data Architect, Platform Engineer
- ⚠️ **Schwierigkeit:** Mehr Implementierung, System-Design

**Entscheidungshilfe:**
- Magst du Mathe/Statistik? → ML
- Magst du Systeme bauen? → DE
- Willst du Modelle entwickeln? → ML
- Willst du Datenplattformen bauen? → DE

**Tipp:** Beide Tracks ergänzen sich! Du kannst auch Hybrid machen (ML + einige DE Kurse).]

---

Now generate similar comparison examples for TUM course consultation.
```

---

### 4️⃣ Constraint Satisfaction（约束满足推理）

**特点**：检查多个**约束条件**是否满足

**TUM场景任务**：
- 时间冲突检查："我选了X和Y，还能加Z吗？"
- 学分要求验证："我的选课满足毕业要求吗？"
- 前置课程检查："我能直接选这门课吗？"
- 规则合规性："同时选3门Seminar可以吗？"

**Prompt模板**：

```
You are a helpful assistant for TUM course consultation.
You will answer questions that require **constraint checking** - verifying if multiple requirements/constraints are satisfied.

Your task:
1. Identify all constraints that need to be checked
2. Retrieve relevant information for constraint verification
3. Check each constraint one by one
4. Provide clear yes/no answer with explanation

Output format:
Thought: [Your reasoning]
Action: Search[query] OR Finish[answer]
Observation: [Search results]

Important rules:
- **List all constraints**: Schedule conflicts, prerequisites, credit limits, enrollment rules
- **Check systematically**: Verify each constraint
- **Clear verdict**: State if plan is feasible or not, and why

---

**Few-shot Examples:**

Example 1: Schedule Conflict Check

Question: "I'm taking Machine Learning and Computer Vision. Can I also add Natural Language Processing in the same semester? Will there be time conflicts?"

Thought: I need to check the schedules of all three courses to see if there are conflicts
Action: Search[Machine Learning IN2064 schedule Winter semester]
Observation: Machine Learning: Lecture Tuesday 10-12, Thursday 10-12. Exercise Friday 14-16.

Thought: Check Computer Vision schedule
Action: Search[Computer Vision IN2128 schedule]
Observation: Computer Vision: Lecture Monday 14-16, Wednesday 14-16. Exercise Tuesday 16-18.

Thought: Check NLP schedule
Action: Search[Natural Language Processing IN2361 schedule]
Observation: Natural Language Processing: Lecture Tuesday 14-16, Thursday 14-16. Exercise Wednesday 10-12.

Thought: Now check for conflicts
Action: Finish[**Schedule Analysis:**

**Machine Learning:**
- Lecture: Tue 10-12, Thu 10-12
- Exercise: Fri 14-16

**Computer Vision:**
- Lecture: Mon 14-16, Wed 14-16
- Exercise: Tue 16-18

**Natural Language Processing:**
- Lecture: Tue 14-16, Thu 14-16
- Exercise: Wed 10-12

**✅ No time conflicts detected!**

All lectures and exercises are at different times. Your weekly schedule would be:
- Mon: CV 14-16
- Tue: ML 10-12, CV Ex 16-18, NLP 14-16 (busy day!)
- Wed: CV 14-16, NLP Ex 10-12
- Thu: ML 10-12, NLP 14-16
- Fri: ML Ex 14-16

**Workload:** 20 ECTS, heavy but doable. Tuesday is packed with back-to-back classes.]

---

Example 2: Graduation Requirements Check

Question: "Ich habe folgende Kurse gemacht: ML (8 ECTS), CV (6 ECTS), NLP (6 ECTS), Seminar (5 ECTS). Reicht das für die Kernmodule im Master?"

Thought: I need to check TUM Master Informatics core module requirements
Action: Search[TUM Informatik Master Kernmodul requirements ECTS]
Observation: Master Informatics requires minimum 24 ECTS in Kernmodule (core modules). Kernmodule must include courses from the chosen specialization track.

Thought: Calculate what the student has
Action: Search[Machine Learning Computer Vision NLP Kernmodul classification]
Observation: ML, CV, and NLP are all classified as Kernmodule in the Machine Learning track. Seminars count separately.

Thought: Now verify if requirements are met
Action: Finish[**Kernmodul Check:**

**Deine Kernmodule:**
- Machine Learning: 8 ECTS ✅
- Computer Vision: 6 ECTS ✅
- NLP: 6 ECTS ✅
- **Total Kernmodule: 20 ECTS**

**Anforderung:** Mindestens 24 ECTS in Kernmodulen

**❌ Noch nicht ausreichend!**

**Du brauchst noch:** 4 ECTS (mindestens ein weiteres Kernmodul)

**Empfehlung:**
- Deep Learning (6 ECTS) - passt gut zu deinen bisherigen Kursen
- Reinforcement Learning (6 ECTS)
- Advanced Topics in ML (6 ECTS)

**Hinweis:** Das Seminar (5 ECTS) zählt separat zur Seminar-Anforderung, nicht zu Kernmodulen.]

---

Example 3: Multiple Constraints

Question: "Can I take Advanced Deep Learning in my first semester if I already learned deep learning basics during my Bachelor?"

Thought: I need to check prerequisites for Advanced Deep Learning
Action: Search[Advanced Deep Learning prerequisites TUM requirements]
Observation: Prerequisites: (1) Completion of "Introduction to Deep Learning" OR equivalent knowledge demonstrated, (2) Good Python skills, (3) Linear Algebra and Probability Theory

Thought: Check if Bachelor knowledge can substitute the prerequisite
Action: Search[TUM prerequisite waiver Bachelor deep learning course equivalent]
Observation: If you completed a similar DL course in your Bachelor, you can request prerequisite waiver. Need to show course syllabus and transcript. Approval by lecturer required.

Thought: Check enrollment timing constraints
Action: Search[TUM first semester course enrollment restrictions Master]
Observation: No general restrictions on taking advanced courses in first semester IF prerequisites are met or waived.

Thought: Now provide comprehensive answer about all constraints
Action: Finish[**Constraint Check for Taking Advanced DL in 1st Semester:**

**Requirement 1: Prerequisites**
- ⚠️ **Partially met:** You need to get prerequisite waiver
- **Action needed:**
  - Submit your Bachelor DL course syllabus
  - Show transcript/grade
  - Get approval from Advanced DL lecturer
  - Do this BEFORE semester starts!

**Requirement 2: Technical Skills**
- ✅ Python, Linear Algebra, Probability - likely met from Bachelor

**Requirement 3: Enrollment Rules**
- ✅ No restriction on taking advanced courses in first semester

**Verdict:**
✅ **Possible, but requires prerequisite waiver approval**

**Timeline:**
1. Email lecturer NOW with Bachelor course details
2. Get approval before enrollment period
3. Enroll during registration

**Risk:** If waiver denied, you'd need to take Intro to DL first

**Recommendation:** Have backup plan (enroll in Intro to DL as fallback)]

---

Now generate similar constraint satisfaction examples for TUM course consultation.
```

---

## ✅ 好的Prompt设计原则

### 1. 明确的Few-shot Examples
```
❌ 不好：只有抽象指令
✅ 好：2-3个具体例子，覆盖典型场景
```

### 2. 清晰的输出格式
```
Thought: [推理过程]
Action: Search[具体查询] OR Finish[最终答案]
Observation: [检索结果]
```

### 3. 明确规则和约束
- 何时检索vs用参数知识
- 如何分解问题
- 何时停止

### 4. 双语支持（TUM场景）
```
- Few-shot examples包含德语和英语问题
- 答案语言跟随问题语言
- 专业术语用英语（course code等）
```

### 5. 真实场景覆盖
```
使用TUM实际课程：
- IN2064 Machine Learning
- IN2346 Introduction to Deep Learning
- IN2128 Computer Vision
- 真实学分、真实前置课程
```

---

## 📝 数据生成流程

### Step 1: 准备问题种子

从Reddit、Moodle、TUMonline收集真实问题：
```python
questions_sequential = [
    "What courses do I need before taking ML?",
    "Welche Kurse muss ich für Deep Learning vorbereiten?",
    ...
]

questions_aggregation = [
    "How many ECTS if I take ML, CV, and NLP?",
    "Was sind empfohlene Kurse für erstes Semester?",
    ...
]

# 同样准备comparison和constraint类别
```

### Step 2: 为每个问题生成数据

```python
import openai

# 对每个问题
for question in questions_sequential:
    # 使用sequential prompt + 该问题
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SEQUENTIAL_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0.0  # 第一次用0，后续用0.9
    )

    # 保存生成的reasoning chain
    save_to_dataset(question, response)
```

### Step 3: 质量控制

```python
# 检查生成的数据
for item in generated_data:
    # 1. 检查格式正确
    if not has_valid_format(item):
        continue

    # 2. 检查答案正确性（用LLM验证）
    if not verify_answer(item):
        continue

    # 3. 检查推理链合理性
    if not has_reasonable_reasoning(item):
        continue

    # 通过质量检查，加入最终数据集
    final_dataset.append(item)
```

### Step 4: 混合训练

```python
# 混合所有类别的数据
all_data = (
    sequential_data +
    aggregation_data +
    comparison_data +
    constraint_data
)

# 打乱
random.shuffle(all_data)

# 转换为LLaMA-Factory格式
for item in all_data:
    training_example = {
        "instruction": item["question"],
        "input": "",
        "output": item["reasoning_chain"]
    }
    write_jsonl(training_example)
```

---

## 🎯 TUM场景任务清单

### 必须覆盖的场景（30+）

#### 1. Sequential Dependency (顺序依赖) - 8个任务
1. 前置课程链查询
2. 学习路径规划（零基础→高级）
3. 模块解锁顺序
4. 毕业要求进度检查
5. Track switching路径
6. 补修课程计划
7. 跨专业转入路径
8. Gap year后的续读规划

#### 2. Aggregation (信息聚合) - 8个任务
1. ECTS学分计算
2. 学期课程规划（推荐组合）
3. 学分分布检查（Pflicht/Kern/Wahl）
4. 每周学习时间估算
5. 考试时间汇总
6. 所有ML相关课程列表
7. 某教授所有课程汇总
8. Track要求完整清单

#### 3. Comparison (对比分析) - 8个任务
1. 相似课程对比（Intro vs Advanced）
2. 不同Track对比
3. 同类型课程选择（多个ML课选哪个）
4. 教授风格对比
5. Lecture vs Practical Course对比
6. 学期难度对比（WS vs SS）
7. 线上vs线下课程对比
8. Seminar主题难度对比

#### 4. Constraint Satisfaction (约束检查) - 8个任务
1. 时间冲突检查
2. 学分要求验证
3. 前置课程检查
4. 同类课程数量限制（最多几门Seminar）
5. 毕业资格check
6. Enrollment资格检查
7. Workload可行性（选太多课）
8. Track switching资格

#### 5. 混合场景 - 8个任务
1. 完整学期规划（Sequential + Aggregation + Constraint）
2. 紧急补救方案（挂科后的路径）
3. 加速毕业计划（3学期毕业可行性）
4. 双Track可行性分析
5. 交换学期课程认证
6. 延期毕业规划
7. 转专业完整方案
8. 从零开始的2年Master规划

**总计：40个不同任务场景**

---

## 💡 Prompt质量自检清单

写完Prompt后，问自己：

- [ ] 有2-3个完整的Few-shot examples吗？
- [ ] Examples覆盖简单+复杂场景吗？
- [ ] 输出格式清晰定义了吗？
- [ ] 告诉模型何时Search、何时Finish了吗？
- [ ] 包含德语和英语examples吗？
- [ ] 使用真实的TUM课程代码和名称吗？
- [ ] 推理过程符合该类别特点吗？（顺序/聚合/对比/约束）
- [ ] 能否生成多样化的数据？（改变问题就能生成不同轨迹）

---

## 🚀 快速开始

### 最小可行方案（MVP）

如果时间有限，先实现：

1. **Sequential**: 5个前置课程链问题
2. **Aggregation**: 5个ECTS计算问题
3. **Comparison**: 5个课程对比问题
4. **Constraint**: 5个时间冲突问题

**总计：20个问题 × 7次生成 = 140条数据**

足够开始第一轮SFT训练！

### 完整方案

覆盖上面的40个任务场景，每个任务5-10个问题变体：

**总计：200-400个问题 × 7次生成 = 1400-2800条数据**

接近论文规模！

---

## 📚 参考原始R3-RAG的Prompt结构

原始prompt关键要素（从`prompt_template.py`）：

```python
def prompt_question_initv2():
    """
    关键要素：
    1. Few-shot examples（温度重叠、总统问题）
    2. 明确指令（分解问题、Sequential vs Parallel）
    3. 输出格式（Thought/Action/Observation）
    4. 何时检索vs用参数知识的规则
    """
```

**照搬这个结构，换成TUM场景即可！**

---

## ✅ 总结

### 核心流程
```
写4类Prompt → 用每类Prompt生成数据 → 质量过滤 → 混合所有数据 → SFT训练
```

### 关键点
1. **Prompt = Few-shot + 规则 + 格式**
2. **4类推理模式对应4个Prompt模板**
3. **覆盖40个TUM真实场景**
4. **生成后混合训练，不分类**

### 下一步
1. 根据上面模板，完善4个Prompt
2. 准备种子问题（从Reddit/Moodle收集）
3. 用GPT-4生成数据（每题7次）
4. 质量过滤（保留成功的80%+）
5. 转换为LLaMA-Factory格式
6. 开始SFT训练！

---

**面试时怎么说？**

> "数据生成我设计了4类Prompt，分别对应Sequential、Aggregation、Comparison、Constraint四种推理模式。每个Prompt包含Few-shot examples和明确的推理规则。用GPT-4生成后，经过质量过滤，保留成功率80%+的数据，最终混合训练。这样保证了数据多样性，让模型学会不同推理策略。"

简洁、专业、有理有据！🎯
