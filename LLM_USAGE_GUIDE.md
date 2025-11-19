# LLM Response Generation Guide

这是Junyao Liu的任务指南：使用GPT-3.5生成对话响应并计算评估指标。

## 📋 任务清单

- [ ] 安装所需依赖
- [ ] 设置OpenAI API密钥
- [ ] 生成LLM响应（20条测试样本）
- [ ] 计算自动评估指标（BERTScore, BLEU, Distinct-n）
- [ ] 记录延迟和token成本
- [ ] 进行定性错误分析

---

## 🛠️ 安装依赖

```bash
# 安装基础依赖
pip install openai bert-score nltk numpy tqdm

# 或者使用requirements文件
pip install -r requirements_llm.txt
```

---

## 🔑 设置OpenAI API密钥

### 方法1：环境变量（推荐）
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 方法2：命令行参数
```bash
python generate_llm_responses.py --api-key 'your-api-key-here'
```

---

## 📝 生成LLM响应

### 基础使用（生成20条样本）

```bash
# 处理两个数据集，各20条样本
python generate_llm_responses.py --num-samples 20

# 只处理DailyDialog
python generate_llm_responses.py --dataset dailydialog --num-samples 20

# 只处理EmpatheticDialogues
python generate_llm_responses.py --dataset empathetic_dialogues --num-samples 20
```

### 自定义参数

```bash
python generate_llm_responses.py \
  --dataset both \
  --num-samples 20 \
  --model gpt-3.5-turbo \
  --temperature 0.7 \
  --max-tokens 150 \
  --output-dir data/llm_outputs
```

### 参数说明

- `--dataset`: 选择数据集 (`dailydialog`, `empathetic_dialogues`, `both`)
- `--num-samples`: 每个数据集处理的样本数量（默认20）
- `--model`: OpenAI模型名称（默认 `gpt-3.5-turbo`）
- `--temperature`: 采样温度（默认0.7）
- `--max-tokens`: 生成的最大token数（默认150）
- `--output-dir`: 输出目录（默认 `data/llm_outputs`）

---

## 📊 计算评估指标

生成响应后，计算自动评估指标：

```bash
# 计算DailyDialog的指标
python compute_llm_metrics.py \
  --input data/llm_outputs/dailydialog_gpt35_responses.jsonl \
  --output data/llm_outputs/dailydialog_metrics.json

# 计算EmpatheticDialogues的指标
python compute_llm_metrics.py \
  --input data/llm_outputs/empathetic_dialogues_gpt35_responses.jsonl \
  --output data/llm_outputs/empathetic_dialogues_metrics.json
```

### 使用自定义BERTScore模型

```bash
python compute_llm_metrics.py \
  --input data/llm_outputs/dailydialog_gpt35_responses.jsonl \
  --output data/llm_outputs/dailydialog_metrics.json \
  --bertscore-model microsoft/deberta-xlarge-mnli
```

---

## 📂 输出文件结构

运行脚本后，你将得到：

```
data/llm_outputs/
├── dailydialog_gpt35_responses.jsonl          # 生成的响应
├── dailydialog_gpt35_responses_summary.json   # 生成摘要
├── dailydialog_metrics.json                   # 评估指标
├── empathetic_dialogues_gpt35_responses.jsonl
├── empathetic_dialogues_gpt35_responses_summary.json
└── empathetic_dialogues_metrics.json
```

---

## 📄 输出文件格式

### 1. 响应文件 (`*_responses.jsonl`)

每一行是一个JSON对象：

```json
{
  "dataset": "dailydialog",
  "dialog_id": "dd_test_123",
  "turn_id": 5,
  "context": "Hello <eot> How are you?",
  "ground_truth_response": "I'm fine, thank you!",
  "generated_response": "I'm doing well, thanks for asking!",
  "emotion": 1,
  "model": "gpt-3.5-turbo",
  "temperature": 0.7,
  "max_tokens": 150,
  "latency_seconds": 1.234,
  "token_usage": {
    "prompt_tokens": 45,
    "completion_tokens": 12,
    "total_tokens": 57
  }
}
```

### 2. 摘要文件 (`*_summary.json`)

```json
{
  "dataset": "dailydialog",
  "model": "gpt-3.5-turbo",
  "num_samples": 20,
  "total_latency_seconds": 24.56,
  "total_tokens": 1140,
  "avg_latency_seconds": 1.228,
  "avg_tokens_per_sample": 57.0,
  "temperature": 0.7,
  "max_tokens": 150
}
```

### 3. 指标文件 (`*_metrics.json`)

```json
{
  "dataset": "dailydialog",
  "model": "gpt-3.5-turbo",
  "num_samples": 20,
  "bertscore_model": "microsoft/deberta-xlarge-mnli",
  "metrics": {
    "bleu-1": 0.2345,
    "bleu-2": 0.1567,
    "bleu-3": 0.0987,
    "bleu-4": 0.0543,
    "distinct-1": 0.6789,
    "distinct-2": 0.8234,
    "bertscore_precision": 0.8765,
    "bertscore_recall": 0.8543,
    "bertscore_f1": 0.8654,
    "avg_response_length": 15.3,
    "total_latency_seconds": 24.56,
    "avg_latency_seconds": 1.228,
    "total_tokens": 1140
  }
}
```

---

## 🔍 定性错误分析

查看生成的响应文件，分析：

1. **语义准确性**：响应是否与上下文相关？
2. **流畅性**：语法是否正确，表达是否自然？
3. **情感一致性**：是否匹配对话的情感？
4. **多样性**：响应是否过于重复或模板化？

### 示例分析代码

```python
import json

# 加载响应
with open('data/llm_outputs/dailydialog_gpt35_responses.jsonl', 'r') as f:
    responses = [json.loads(line) for line in f]

# 查看前几个样本
for i, r in enumerate(responses[:5]):
    print(f"\n=== Sample {i+1} ===")
    print(f"Context: {r['context']}")
    print(f"Ground Truth: {r['ground_truth_response']}")
    print(f"Generated: {r['generated_response']}")
    print(f"Latency: {r['latency_seconds']:.3f}s")
    print(f"Tokens: {r['token_usage']['total_tokens']}")
```

---

## ⚠️ 注意事项

1. **API成本**：GPT-3.5-turbo约 $0.0015/1K tokens (prompt) + $0.002/1K tokens (completion)
   - 20条样本预计花费约 $0.05-0.10

2. **Rate Limits**：脚本已添加0.1秒延迟以避免速率限制

3. **环境变量**：确保设置了 `OPENAI_API_KEY`

4. **依赖版本**：
   - openai >= 0.27.0
   - bert-score >= 0.3.13
   - nltk >= 3.8

---

## 🚀 完整工作流程

```bash
# Step 1: 设置API密钥
export OPENAI_API_KEY='your-api-key-here'

# Step 2: 生成响应（20条样本）
python generate_llm_responses.py --num-samples 20

# Step 3: 计算DailyDialog指标
python compute_llm_metrics.py \
  --input data/llm_outputs/dailydialog_gpt35_responses.jsonl \
  --output data/llm_outputs/dailydialog_metrics.json

# Step 4: 计算EmpatheticDialogues指标
python compute_llm_metrics.py \
  --input data/llm_outputs/empathetic_dialogues_gpt35_responses.jsonl \
  --output data/llm_outputs/empathetic_dialogues_metrics.json

# Step 5: 查看结果
cat data/llm_outputs/dailydialog_metrics.json
cat data/llm_outputs/empathetic_dialogues_metrics.json
```

---

## 📞 问题排查

### 问题1：API密钥错误
```
Error: Incorrect API key provided
```
**解决**：检查 `OPENAI_API_KEY` 环境变量是否设置正确

### 问题2：BERTScore安装失败
```
Warning: bert_score not installed
```
**解决**：
```bash
pip install bert-score
```

### 问题3：NLTK数据缺失
```
LookupError: Resource punkt not found
```
**解决**：
```python
import nltk
nltk.download('punkt')
```

---

## 📈 下一步

完成生成和评估后：
1. ✅ 将结果文件保存到项目目录
2. ✅ 记录关键指标（BLEU, BERTScore, latency, tokens）
3. ✅ 进行定性分析，找出优缺点
4. ✅ 与Hanxiao Wang的小模型结果进行对比
5. ✅ 准备discussion和presentation材料

Good luck! 🎉

