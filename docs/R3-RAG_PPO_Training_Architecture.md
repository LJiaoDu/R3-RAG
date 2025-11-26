# R3-RAG PPO Training Architecture: Technical Deep Dive

## Executive Summary

R3-RAG (Reinforcement Learning-based Retrieval-Augmented Generation) implements a multi-step reasoning system using Proximal Policy Optimization (PPO). This document provides a comprehensive analysis of the training pipeline, addressing the critical question: **How does PPO update multi-hop questions with multiple reasoning steps?**

**Key Answer**: Each reasoning step is treated as an **independent training sample**. A 3-step multi-hop question generates 3 separate Experiences, each trained independently with token-level PPO updates.

## 1. Framework Overview

### 1.1 OpenRLHF Foundation

R3-RAG is built on the **OpenRLHF** framework (not LLaMA-Factory or MS-SWIFT), which provides:
- High-performance RLHF implementation using Ray + DeepSpeed + HuggingFace Transformers
- Core PPO training loop with Actor-Critic architecture
- Distributed training support across multiple GPUs

### 1.2 Custom Modifications

R3-RAG customizes OpenRLHF's Experience generation in:
- `/train/R3RAG_OpenRLHF/openrlhf/trainer/ppo_utils/experience_maker_prm_orm.py`

Key customizations:
- Multi-step generation loop with retrieval integration
- Dual reward system: PRM (Process Reward Model) + ORM (Outcome Reward Model)
- Step-by-step Sample creation with context accumulation

## 2. Multi-Step Reasoning Architecture

### 2.1 Generation Flow

For a single multi-hop question, the system iteratively generates:

```
Prompt: "Who was the director of the 2009 movie starring Daniel Radcliffe?"

Step 1: Thought + Query
→ Retrieval → Documents → PRM Reward (0.96)

Step 2: Thought + Query
→ Retrieval → Documents → PRM Reward (1.28)

Step 3: Thought + Answer
→ ORM Verification → Correct? → ORM Reward (2.0)
```

**Critical Insight**: Each step generates an **independent Sample**, not a single trajectory.

### 2.2 Code Reference: `generate_step_samples()`

Location: `experience_maker_prm_orm.py:308-655`

```python
for search_counter in range(num_search_one_attempt):
    # Generate one reasoning step
    output = actor.generate(SystemInput, temperature=0.6)
    mydict = split_response(output)

    if mydict.get('query'):
        # Retrieval step
        docs = GetRetrieval(mydict['query'])
        reward = calculate_doc_score(...)  # PRM: 0.0-1.0 scale
        reward_list.append(reward)

        # Create Sample for this step
        samples = Samples(
            sequences=sequences,      # Full sequence with context
            attention_mask=attention_mask,
            action_mask=action_mask,  # Marks trainable tokens
            num_actions=action_mask.size(1),
        )
        sample_list.append(samples)

    elif mydict.get('answer'):
        # Final answer step
        correctness = check_correctness(answer, golden_answers)  # ORM

        # Adjust all rewards based on final correctness
        if correctness:
            reward_list = [r * 1.6 for r in reward_list]
            reward_list.append(2.0)
        else:
            reward_list = [r * 0.1 for r in reward_list]
            reward_list.append(0.1)
```

**Output**:
- `sample_list`: [Sample_step1, Sample_step2, Sample_step3]
- `reward_list`: [0.96, 1.28, 2.0]

**Critical**: One-to-one mapping between Samples and scalar rewards.

## 3. Reward System

### 3.1 Process Reward Model (PRM)

**Purpose**: Evaluate document relevance for retrieval steps

**Implementation**: `calculate_doc_score()` at line 1451-1523

```python
def calculate_doc_score(problem, history_steps, current_analysis,
                       current_query, retrieved_docs):
    prompt = f"""Evaluate the current reasoning step (0.0-1.0 scale):
    [Problem]: {problem}
    [Historical Steps]: {history_steps}
    [Current Analysis]: {current_analysis}
    [Search Query]: {current_query}
    [Retrieved Documents]: {retrieved_docs}

    Score the quality and relevance of this step.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    score = extract_score(response.choices[0].message.content)
    return round(score, 3)
```

**Output**: Scalar reward ∈ [0.0, 1.0], e.g., 0.96

### 3.2 Outcome Reward Model (ORM)

**Purpose**: Verify final answer correctness

**Implementation**: `check_correctness()` at line 1276-1329

```python
def check_correctness(question, predicted_answer, golden_answers):
    # Direct match check
    if predicted_answer in golden_answers:
        return True, "Direct match"

    # LLM-based verification
    prompt = f"""Judge whether Answer 1 is correct:
    Question: {question}
    Given Answer: {predicted_answer}
    Standard Answers: {golden_answers}

    Output: True or False
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    result = "True" in response.choices[0].message.content
    return result, response.choices[0].message.content
```

**Reward Adjustment**:
- If ORM returns `True`: Multiply all PRM rewards by 1.6, add 2.0 for answer step
- If ORM returns `False`: Multiply all PRM rewards by 0.1, add 0.1 for answer step

### 3.3 Golden Answers Source

**Dataset**: `Yuan-Li-FNLP/R3-RAG-RLTrainingData` (HuggingFace)

**Data Flow**:
1. `blending_datasets()` loads from HuggingFace
2. `PromptDataset` (prompts_dataset.py:18-62) preprocesses
3. Each entry contains:
   ```python
   {
       "context_messages": "Question prompt...",
       "golden_answers": ["answer1", "answer2", ...]
   }
   ```
4. Stored in `experience.info["reward"]` for ORM evaluation

## 4. Sample vs Experience

### 4.1 Sample Structure

**Definition**: Raw generation output before advantage calculation

```python
class Samples:
    sequences: Tensor          # Token IDs [batch, seq_len]
    attention_mask: Tensor     # Attention mask [batch, seq_len]
    action_mask: Tensor        # Trainable tokens [batch, num_actions]
    num_actions: int           # Number of trainable tokens
```

**Creation**: One Sample per reasoning step in `generate_step_samples()`

**Example for Step 1**:
```python
sequences = [1, 1234, 5678, ..., 9012]  # 100 tokens
action_mask = [0, 0, 0, ..., 1, 1, 1]   # Last 20 tokens trainable
num_actions = 20
```

### 4.2 Experience Structure

**Definition**: Processed Sample with advantages and returns

```python
class Experience:
    sequences: Tensor             # Same as Sample
    action_log_probs: Tensor      # Log π(a|s) for each token
    values: Tensor                # V(s) for each token
    advantages: Tensor            # GAE advantages
    returns: Tensor               # GAE returns (targets for Critic)
    attention_mask: Tensor
    action_mask: Tensor
    info: dict                    # Contains reward, kl, etc.
    kl: Tensor                    # KL divergence per token
```

**Creation**: `make_experience()` at line 658-733

```python
def make_experience(self, samples: Samples, rewards) -> Experience:
    # rewards is SCALAR, e.g., 0.96
    r = torch.tensor([rewards], dtype=torch.bfloat16)

    # Calculate log_probs with current Actor
    action_log_probs = self.actor(
        sequences, num_actions, attention_mask
    )

    # Calculate values with Critic
    value = self.critic(
        sequences, num_actions, attention_mask
    )

    # Calculate KL divergence with frozen reference model
    base_action_log_probs = self.initial_model(...)
    kl = compute_approx_kl(
        action_log_probs,
        base_action_log_probs,
        action_mask
    )

    return Experience(
        sequences, action_log_probs, value,
        None, None,  # advantages/returns computed later
        attention_mask, action_mask,
        {"reward": r}, kl
    )
```

**Critical Conversion**: One Sample + one scalar reward → one Experience

## 5. Token-Level Reward Distribution

### 5.1 The Challenge

PPO operates at the **token level**, but we have **step-level scalar rewards**.

**Question**: How does scalar reward (0.96) become token-level rewards?

### 5.2 `compute_reward()` Function

Location: `/train/R3RAG_OpenRLHF/openrlhf/models/utils.py:37-74`

```python
def compute_reward(r, kl_coef, kl, action_mask, num_actions, ...):
    """
    Args:
        r: Scalar reward per sample, shape [batch]
        kl_coef: KL penalty coefficient (0.01)
        kl: Per-token KL divergence, shape [batch, num_actions]
        num_actions: Number of trainable tokens per sample

    Returns:
        Token-level rewards, shape [batch, num_actions]
    """
    reward = []
    for i, (kl_seg, action_len) in enumerate(zip(kl, num_actions)):
        # Every token gets KL penalty
        kl_reward = -kl_coef * kl_seg  # Shape: [num_actions]

        # ONLY last token gets the scalar reward
        kl_reward[action_len - 1] += r[i]

        reward.append(kl_reward)

    return torch.stack(reward)  # [batch, num_actions]
```

### 5.3 Concrete Example

**Input**:
- Scalar reward: `r = 0.96`
- KL coefficient: `kl_coef = 0.01`
- Number of trainable tokens: `num_actions = 20`
- KL divergence per token: `kl = [0.02, 0.03, 0.04, ..., 0.05]`

**Process**:
```python
# Step 1: Calculate KL penalty for each token
kl_reward = -0.01 * [0.02, 0.03, 0.04, ..., 0.05]
          = [-0.0002, -0.0003, -0.0004, ..., -0.0005]

# Step 2: Add scalar reward to LAST token
kl_reward[19] += 0.96
kl_reward[19] = -0.0005 + 0.96 = 0.9595

# Final token-level rewards
token_rewards = [-0.0002, -0.0003, -0.0004, ..., 0.9595]
```

**Output**: 20 token-level rewards, only last one is positive.

### 5.4 GAE Advantage Calculation

After reward distribution, calculate advantages using **Generalized Advantage Estimation (GAE)**:

**Parameters**:
- γ (gamma) = 0.99: Discount factor
- λ (lambda) = 0.95: GAE parameter

**Formula**:
```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
A_t = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}
```

**Example for 20 tokens**:
```python
# Token-level rewards (from compute_reward)
rewards = [-0.0002, -0.0003, ..., 0.9595]

# Critic predictions
values = [0.45, 0.46, 0.47, ..., 0.50]

# Calculate TD errors
deltas = [
    -0.0002 + 0.99*0.46 - 0.45,  # δ_0
    -0.0003 + 0.99*0.47 - 0.46,  # δ_1
    ...
    0.9595 + 0 - 0.50            # δ_19 (last token, no next state)
]

# Calculate GAE advantages (working backwards)
advantages = [0.12, 0.13, 0.15, ..., 1.36]  # Shape: [20]

# Calculate returns (targets for Critic)
returns = advantages + values
        = [0.57, 0.59, 0.62, ..., 1.86]
```

**Critical**: Each token gets its own advantage and return value.

## 6. Multi-Hop QA PPO Update Mechanism

### 6.1 The Core Question

**User's Question**: "对于一个多跳问题，一个prompt对应很多step，每个步都有个reward列表，ppo是如何更新的？"

Translation: "For a multi-hop question, one prompt corresponds to many steps, each step has a reward list, how does PPO update?"

### 6.2 The Answer

**Multi-hop QA in R3-RAG does NOT update as one logical chain. Instead, it processes EACH STEP independently.**

**Detailed Breakdown**:

1. **Generation Phase** (one multi-hop question):
   ```
   Prompt → Step 1 → Step 2 → Step 3
          ↓         ↓         ↓
       Sample_1  Sample_2  Sample_3
          ↓         ↓         ↓
       Reward: 0.96  1.28   2.0
   ```

2. **Experience Creation** (one-to-one mapping):
   ```python
   # Code: experience_maker_prm_orm.py:224-230
   for samples, myrewards in zip(samples_list, myrewards_list):
       # samples = Sample for one step
       # myrewards = scalar reward for that step
       experience = self.make_experience(samples, myrewards)
       experiences.append(experience)

   # Result: 3 independent Experiences
   experiences = [
       Experience_1 (100 tokens, reward=0.96),
       Experience_2 (120 tokens, reward=1.28),
       Experience_3 (8 tokens, reward=2.0)
   ]
   ```

3. **Replay Buffer** (shuffling):
   ```python
   # Code: ppo_trainer.py:189-262
   for rand_prompts in prompts_dataloader:
       experience_list = experience_maker.make_experience_list(rand_prompts)
       for experience in experience_list:
           self.replay_buffer.append(experience)

   # Replay buffer contains Experiences from:
   # - Different questions (Q1, Q2, Q3, ...)
   # - Different steps (Step 1, Step 2, Step 3)
   # - All SHUFFLED together
   ```

4. **PPO Training** (independent updates):
   ```python
   dataloader = DataLoader(
       self.replay_buffer,
       batch_size=64,
       shuffle=True  # Further shuffling
   )

   for experience in dataloader:
       # Could contain:
       # - Q1_Step1 (100 tokens)
       # - Q5_Step3 (8 tokens)
       # - Q2_Step2 (120 tokens)
       # - ...

       training_step(experience)  # Update Actor & Critic
   ```

### 6.3 Why This Works

**Question**: If steps are trained independently, how does the model maintain multi-step coherence?

**Answer**: Context accumulation in sequence construction.

**Example**:

**Step 1 Sequence**:
```
[System] + [Prompt] + [Step1_Thought] + [Step1_Query]
→ 100 tokens total
→ Last 20 tokens trainable (Step1_Query generation)
```

**Step 2 Sequence**:
```
[System] + [Prompt] + [Step1_Thought] + [Step1_Query] + [Docs1] +
[Step2_Thought] + [Step2_Query]
→ 120 tokens total
→ Last 30 tokens trainable (Step2_Query generation)
→ Step1 output included in context (frozen)
```

**Step 3 Sequence**:
```
[System] + [Prompt] + [Step1_Full] + [Docs1] + [Step2_Full] + [Docs2] +
[Step3_Answer]
→ 8 tokens total (just the answer)
→ All 8 tokens trainable
→ Step1 and Step2 outputs in context (frozen)
```

**Key Points**:
- Each step sees previous steps in its input context
- Previous step outputs are FROZEN (not trained again)
- Only current step tokens are trainable (marked by `action_mask`)
- Model learns conditional generation: P(Step_i | Prompt, Step_1, ..., Step_{i-1})

## 7. PPO Training Loop

### 7.1 Actor Update (Policy)

Location: `ppo_trainer.py:335-419`

```python
def training_step_actor(self, experience):
    """Update Actor model (policy network)"""

    # Recalculate log_probs with CURRENT Actor
    action_log_probs = self.actor(
        experience.sequences,
        experience.num_actions,
        experience.attention_mask
    )
    # Shape: [batch, num_actions]

    # PPO policy loss
    actor_loss = self.actor_loss_fn(
        action_log_probs,              # New policy π_θ
        experience.action_log_probs,   # Old policy π_θ_old
        experience.advantages,          # GAE advantages
        action_mask=experience.action_mask,
    )

    # Backpropagation
    self.strategy.backward(actor_loss, self.actor, self.actor_optim)

    # Update parameters
    self.strategy.optimizer_step(
        self.actor_optim,
        self.actor,
        self.actor_scheduler,
        name="actor"
    )
```

**Policy Loss Formula** (`loss.py:56-77`):
```python
def forward(self, log_probs, old_log_probs, advantages, action_mask):
    # Calculate importance ratio
    ratio = (log_probs - old_log_probs).exp()
    # ratio[t] = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

    # Two surrogate objectives
    surr1 = ratio * advantages
    surr2 = ratio.clamp(0.8, 1.2) * advantages  # clip_eps=0.2

    # Take minimum (pessimistic bound)
    loss = -torch.min(surr1, surr2)

    # Average over trainable tokens
    loss = masked_mean(loss, action_mask, dim=-1).mean()
    return loss
```

**Concrete Example** (100-token Experience):

```python
# Experience has 100 tokens, last 20 trainable
action_mask = [0]*80 + [1]*20

# Recalculate log_probs
log_probs = [-5.2, -4.8, ..., -3.1]      # 20 values (trainable tokens)
old_log_probs = [-5.0, -4.7, ..., -3.0]  # From experience
advantages = [0.12, 0.13, ..., 1.36]     # From GAE

# Token-by-token calculation
for t in range(20):
    ratio[t] = exp(log_probs[t] - old_log_probs[t])
    # e.g., exp(-5.2 - (-5.0)) = exp(-0.2) = 0.819

    surr1[t] = ratio[t] * advantages[t]
    # e.g., 0.819 * 0.12 = 0.098

    surr2[t] = clip(ratio[t], 0.8, 1.2) * advantages[t]
    # e.g., clip(0.819, 0.8, 1.2) * 0.12 = 0.8 * 0.12 = 0.096

    loss[t] = -min(surr1[t], surr2[t])
    # e.g., -min(0.098, 0.096) = -0.096

# Average over 20 tokens
actor_loss = mean(loss) = -0.15  # Example value

# Backprop updates θ to maximize objective (minimize negative loss)
```

**Learning Rate**: `5e-7` (configured in RLHF.sh:26)

### 7.2 Critic Update (Value Function)

Location: `ppo_trainer.py:421-472`

```python
def training_step_critic(self, experience):
    """Update Critic model (value network)"""

    # Recalculate values with CURRENT Critic
    values = self.critic(
        experience.sequences,
        experience.num_actions,
        experience.attention_mask
    )
    # Shape: [batch, num_actions]

    # Value loss
    critic_loss = self.critic_loss_fn(
        values,                  # New predictions V_φ
        experience.values,       # Old predictions V_φ_old
        experience.returns,      # GAE returns (targets)
        action_mask=experience.action_mask,
    )

    # Backpropagation
    self.strategy.backward(critic_loss, self.critic, self.critic_optim)

    # Update parameters
    self.strategy.optimizer_step(
        self.critic_optim,
        self.critic,
        self.critic_scheduler,
        name="critic"
    )
```

**Value Loss Formula** (`loss.py:80-105`):
```python
def forward(self, values, old_values, returns, action_mask):
    # Clip new values around old values
    values_clipped = old_values + (values - old_values).clamp(-0.2, 0.2)

    # Two squared error terms
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2

    # Take maximum (pessimistic bound)
    loss = torch.max(surr1, surr2)

    # Average over trainable tokens
    loss = masked_mean(loss, action_mask, dim=-1).mean()
    return 0.5 * loss
```

**Concrete Example** (same 100-token Experience):

```python
# Returns (targets from GAE)
returns = [0.57, 0.59, 0.62, ..., 1.86]  # 20 values

# Old values (from experience)
old_values = [0.45, 0.46, 0.47, ..., 0.50]

# Recalculate values
new_values = critic(sequences)
# → [0.48, 0.49, 0.50, ..., 0.55]  # 20 values

# Token-by-token calculation
for t in range(20):
    # Clipped prediction
    val_clipped[t] = old_values[t] + clip(
        new_values[t] - old_values[t],
        -0.2,
        0.2
    )
    # e.g., 0.45 + clip(0.48 - 0.45, -0.2, 0.2)
    #     = 0.45 + 0.03 = 0.48

    surr1[t] = (val_clipped[t] - returns[t]) ** 2
    # e.g., (0.48 - 0.57)^2 = 0.0081

    surr2[t] = (new_values[t] - returns[t]) ** 2
    # e.g., (0.48 - 0.57)^2 = 0.0081

    loss[t] = max(surr1[t], surr2[t])
    # e.g., max(0.0081, 0.0081) = 0.0081

# Average over 20 tokens
critic_loss = 0.5 * mean(loss) = 0.5 * 0.01 = 0.005

# Backprop updates φ to minimize squared error
```

**Learning Rate**: `9e-6` (configured in RLHF.sh:27)

**Note**: Critic learning rate is **18x larger** than Actor's (9e-6 vs 5e-7). This allows faster value function learning to provide better advantage estimates.

### 7.3 Combined Training Step

Location: `ppo_trainer.py:327-333`

```python
def training_step(self, experience, global_steps):
    """Update both Actor and Critic"""
    status = {}

    # Update Actor (if not frozen)
    if global_steps > self.freezing_actor_steps:
        status = self.training_step_actor(experience)

    # Update Critic (always)
    if self.critic is not None:
        status.update(self.training_step_critic(experience))

    return status
```

**Gradient Contribution**:
- Each token in `action_mask=1` contributes gradients
- 100-token Experience with 20 trainable tokens → 20 gradient contributions
- Loss is averaged over trainable tokens
- Backpropagation distributes gradients to all model parameters

## 8. Complete Pipeline Example

### 8.1 Setup

**Question**: "Who was the director of the 2009 movie starring Daniel Radcliffe?"

**Golden Answers**: ["David Yates"]

**Configuration** (from RLHF.sh):
- Batch size: 64 prompts
- Samples per prompt: 4
- Max epochs: 1
- Episodes: 4
- Actor LR: 5e-7
- Critic LR: 9e-6
- KL coefficient: 0.01
- GAE: γ=0.99, λ=0.95

### 8.2 Step-by-Step Execution

#### Phase 1: Multi-Step Generation

```python
# Input: 1 prompt from dataset
prompt = {
    "context_messages": "Who was the director...",
    "golden_answers": ["David Yates"]
}

# Generate 3 reasoning steps
sample_list, reward_list = generate_step_samples(prompt)

# Results:
# Step 1: Retrieve movie info
sample_list[0] = Sample(
    sequences=[SYS, PROMPT, "Let me search...", QUERY: "2009 Daniel Radcliffe movie"],
    num_actions=20  # QUERY tokens trainable
)
reward_list[0] = 0.96  # PRM score

# Step 2: Retrieve director info
sample_list[1] = Sample(
    sequences=[SYS, PROMPT, STEP1, DOCS1, "Found HP6...", QUERY: "director Harry Potter 6"],
    num_actions=30  # QUERY tokens trainable
)
reward_list[1] = 1.28  # PRM score

# Step 3: Provide answer
sample_list[2] = Sample(
    sequences=[SYS, PROMPT, STEP1, DOCS1, STEP2, DOCS2, ANSWER: "David Yates"],
    num_actions=8  # ANSWER tokens trainable
)

# ORM verification
correctness = check_correctness("David Yates", ["David Yates"])  # True

# Adjust rewards
reward_list = [0.96 * 1.6, 1.28 * 1.6, 2.0]  # [1.536, 2.048, 2.0]
```

#### Phase 2: Experience Creation

```python
# Convert each Sample to Experience
experiences = []
for sample, reward in zip(sample_list, reward_list):
    experience = make_experience(sample, reward)
    # Includes: log_probs, values, KL
    experiences.append(experience)

# Results: 3 independent Experiences
# Experience_1: 100 tokens, 20 trainable, reward=1.536
# Experience_2: 120 tokens, 30 trainable, reward=2.048
# Experience_3: 108 tokens, 8 trainable, reward=2.0
```

#### Phase 3: Token-Level Reward Distribution

```python
# For Experience_1 (20 trainable tokens)
r = 1.536
kl = [0.02, 0.03, ..., 0.05]  # 20 values

token_rewards = compute_reward(r, kl_coef=0.01, kl, ...)
# → [-0.0002, -0.0003, ..., 1.5355]  # Last token gets 1.536

# Calculate advantages (GAE)
values = critic(sequences)  # [0.45, 0.46, ..., 0.50]
advantages = compute_gae(token_rewards, values, γ=0.99, λ=0.95)
# → [0.12, 0.13, ..., 1.36]

# Calculate returns
returns = advantages + values
# → [0.57, 0.59, ..., 1.86]

# Store in Experience_1
experience.advantages = advantages
experience.returns = returns
```

#### Phase 4: Replay Buffer Accumulation

```python
# Process 64 prompts
for prompt in batch_of_64_prompts:
    experiences = make_experience_list(prompt)
    # Each prompt generates 1-3 Experiences (varies by num steps)
    for exp in experiences:
        replay_buffer.append(exp)

# Replay buffer now contains ~160 Experiences:
# - 64 prompts × average 2.5 steps per prompt
# - Mixed from different questions and steps
```

#### Phase 5: PPO Training

```python
# Normalize advantages across all Experiences
replay_buffer.normalize("advantages")

# Create dataloader with shuffling
dataloader = DataLoader(replay_buffer, batch_size=64, shuffle=True)

# Training loop
for epoch in range(1):  # max_epochs=1
    for batch in dataloader:
        # batch contains 64 Experiences (randomly sampled)

        # Update Actor
        actor_loss = training_step_actor(batch)
        # Gradients from all 64 Experiences
        # Each Experience contributes based on its trainable tokens

        # Update Critic
        critic_loss = training_step_critic(batch)
        # Gradients from all 64 Experiences
```

### 8.3 Key Observations

1. **Independence**: Step 1, Step 2, Step 3 of same question are treated as 3 independent training samples
2. **Shuffling**: These 3 Experiences are shuffled with Experiences from 63 other questions
3. **Batch Training**: A training batch might contain:
   - Q1_Step1, Q5_Step3, Q2_Step2, Q15_Step1, ...
4. **Token-Level Updates**: Each token contributes gradients proportional to its advantage
5. **Context Coherence**: Despite independent training, context accumulation maintains multi-step reasoning capability

## 9. Training Configuration

### 9.1 Hyperparameters (from RLHF.sh)

```bash
# Model
--pretrain "your_R3-RAG-CS_model_path"  # Base model checkpoint
--save_path "R3-RAG_save_path"          # Final model save location
--ckpt_path "checkpoint_save_path"       # Training checkpoints

# Reward Model
--remote_rm_url http://localhost:5000/get_reward  # PRM/ORM API endpoint

# Batch Sizes
--micro_train_batch_size 4       # Per-device training batch
--train_batch_size 64            # Total training batch (4 × 16 GPUs)
--micro_rollout_batch_size 1     # Per-device generation batch
--rollout_batch_size 64          # Total generation batch

# Sampling
--n_samples_per_prompt 4         # Generate 4 trajectories per prompt (best-of-4)

# Training Duration
--max_epochs 1                   # PPO epochs per batch
--num_episodes 4                 # Total training episodes
--save_steps 8                   # Save checkpoint every 8 steps

# Sequence Lengths
--prompt_max_len 4096            # Max input context length
--generate_max_len 512           # Max generation length per step

# Optimization
--actor_learning_rate 5e-7       # Policy network LR
--critic_learning_rate 9e-6      # Value network LR (18x Actor)
--init_kl_coef 0.01              # KL penalty coefficient
--adam_offload                   # Offload optimizer states to CPU

# PPO Settings
--advantage_estimator gae        # Use GAE for advantage calculation
--normalize_reward               # Normalize rewards across batch

# DeepSpeed
--zero_stage 2                   # ZeRO optimization stage 2
--bf16                           # Use bfloat16 precision

# Model Optimizations
--flash_attn                     # Flash Attention 2
--gradient_checkpointing         # Gradient checkpointing to save memory

# Data
--prompt_data "R3-RAG_RLTraingData"  # HuggingFace dataset
--input_key context_messages          # Key for input prompts

# Resumption
--load_checkpoint                # Resume from checkpoint if exists
```

### 9.2 Hardware Requirements

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
deepspeed --num_gpus=8 --module openrlhf.cli.train_ppo ...
```

- **GPUs**: 8 × high-memory GPUs (A100 recommended)
- **Memory**: Flash Attention + Gradient Checkpointing + ZeRO-2 for efficient training
- **Distributed**: DeepSpeed for multi-GPU training

## 10. Critical Insights Summary

### 10.1 Multi-Step Training Architecture

**Question**: How does PPO update multi-hop questions?

**Answer**:
- ✅ Each step is an **independent training sample**
- ❌ NOT trained as one connected trajectory
- ✅ Context accumulation provides coherence
- ✅ Experiences shuffled across questions and steps

### 10.2 Reward Structure

**Confusion**: "reward是不是列表形式？" (Is reward in list form?)

**Clarification**:
- **Step-level**: `reward_list = [0.96, 1.28, 2.0]` (one scalar per step)
- **Sample-level**: Each Sample paired with ONE scalar reward
- **Token-level**: `compute_reward()` distributes scalar to last token only
- **Gradient-level**: Each token contributes to loss based on its advantage

### 10.3 Token-Level Mechanism

**Question**: "转为token-level reward: 是什么意思？" (What does token-level reward mean?)

**Answer**:
```python
# Input: Step-level scalar
reward = 0.96

# Process: Distribute to tokens
token_rewards = [-0.0002, -0.0003, ..., 0.9595]
#                ↑                      ↑
#           KL penalty only      Scalar reward + KL penalty

# Output: Each token has a reward
# GAE then calculates advantages/returns per token
# PPO loss averaged over all trainable tokens
```

### 10.4 Actor-Critic Updates

Both models update for **every Experience**:

**Actor**:
- Learns to generate tokens that maximize expected advantages
- Clipped objective prevents large policy changes
- Learning rate: 5e-7 (conservative)

**Critic**:
- Learns to predict future returns for each token
- Provides advantage estimates for Actor training
- Learning rate: 9e-6 (18x Actor, needs faster convergence)

## 11. Code Reference Index

### Core Training Files

| File | Lines | Purpose |
|------|-------|---------|
| `experience_maker_prm_orm.py` | 308-655 | Multi-step generation loop |
| `experience_maker_prm_orm.py` | 658-733 | Experience creation |
| `experience_maker_prm_orm.py` | 1276-1329 | ORM evaluation |
| `experience_maker_prm_orm.py` | 1451-1523 | PRM evaluation |
| `ppo_trainer.py` | 189-262 | Main training loop |
| `ppo_trainer.py` | 263-323 | PPO update loop |
| `ppo_trainer.py` | 335-419 | Actor update |
| `ppo_trainer.py` | 421-472 | Critic update |
| `loss.py` | 56-77 | Policy loss (clipped) |
| `loss.py` | 80-105 | Value loss (clipped) |
| `utils.py` | 37-74 | Token-level reward distribution |
| `prompts_dataset.py` | 18-62 | Dataset with golden_answers |

### Training Script

| File | Lines | Purpose |
|------|-------|---------|
| `RLHF.sh` | 1-40 | Training configuration |

## 12. Comparison: Traditional PPO vs R3-RAG PPO

| Aspect | Traditional PPO | R3-RAG PPO |
|--------|----------------|------------|
| **Task** | Single episode (e.g., Atari game) | Multi-step reasoning |
| **Trajectory** | One trajectory per episode | Multiple Experiences per question |
| **Reward** | Cumulative episode reward | Step-level PRM/ORM rewards |
| **Update** | Update on full trajectory | Update on each step independently |
| **Context** | State observation | Accumulated conversation history |
| **Action Space** | Discrete actions (e.g., Up/Down) | Token generation (vocabulary size) |
| **Advantage** | Per-timestep in episode | Per-token in step |
| **Shuffling** | Shuffle trajectories | Shuffle steps from different questions |

**Key Difference**: Traditional PPO treats one game episode as one training sample. R3-RAG treats each reasoning step as one training sample, even though multiple steps form one logical question.

## 13. FAQ

### Q1: Why not train multi-step reasoning as one connected trajectory?

**A**: Practical and theoretical reasons:
- **Variable length**: Questions have 1-5 steps; batching is difficult
- **Credit assignment**: Step-level rewards clearly attribute credit
- **Sample efficiency**: More training samples (3 steps = 3 samples vs 1)
- **Convergence**: Independent updates converge faster than long trajectory updates

### Q2: Doesn't independent training break multi-step coherence?

**A**: No, because:
- **Context accumulation**: Each step sees previous steps in input
- **Pre-training**: Base model already has language coherence
- **Reward signal**: Correct final answer requires coherent reasoning chain

### Q3: Why is Critic LR 18x larger than Actor LR?

**A**:
- **Critic** needs to learn value function for new policy quickly
- **Actor** must change slowly to maintain stability (PPO principle)
- Fast Critic → Better advantage estimates → Better Actor updates

### Q4: What happens if a step generates wrong output?

**A**:
- Wrong retrieval query → Low PRM score (0.2-0.4)
- Wrong final answer → ORM False → All rewards ×0.1
- Low reward → Negative advantages → Policy learns to avoid
- Context contamination → Next steps also get low rewards

### Q5: How does best-of-4 sampling work?

**A** (`n_samples_per_prompt=4`):
- Generate 4 complete reasoning chains per prompt
- Each chain has 1-5 steps
- Each step becomes a Sample → Experience
- All Experiences from all 4 chains added to replay buffer
- Better chains (higher rewards) contribute more to learning via advantages

## 14. Conclusion

R3-RAG implements a **step-decomposed PPO training paradigm** where:

1. **Multi-step reasoning questions** are decomposed into **independent training samples**
2. Each step receives **step-level rewards** (PRM for retrieval, ORM for answers)
3. Scalar rewards are **distributed to tokens** (last token gets reward, others get KL penalty)
4. **Token-level PPO updates** train both Actor and Critic
5. **Context accumulation** maintains coherence across independently-trained steps

This design enables efficient training of complex reasoning behaviors while maintaining the stability and sample efficiency of PPO.

**Core Insight**: Multi-hop QA training in R3-RAG is **NOT** one logical chain update, but **step-by-step independent updates** with context-based coherence.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-25
**Author**: Technical Analysis of R3-RAG Training Pipeline
