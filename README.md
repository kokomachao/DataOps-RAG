# Data Platform Knowledge Base RAG：问答与排障 SOP 助手

> 目标：基于 Kaggle StackSample（StackOverflow Posts 数据集）按 Spark / Flink / Kafka / Hadoop / Hive 标签构建知识库，
> 通过 **Dense(Embedding)+Sparse(BM25) 混合召回 + RRF 融合** 做组件级检索，并输出标准化排障 SOP。

本项目参考了 Datawhale 的 All-in-RAG 教程中的整体思路：数据准备→分块→索引→检索优化（混合检索/RRF）→生成集成与格式化输出。  
- All-in-RAG 总览：https://datawhalechina.github.io/all-in-rag/  
- 混合检索/RRF 介绍（倒数排序融合）：https://github.com/datawhalechina/all-in-rag/blob/main/docs/chapter4/11_hybrid_search.md

---

## ✨ 功能特性

- **数据构建**：解析 StackSample `Posts.xml`，抽取 Q&A、Tags、Score、AcceptedAnswer 等元数据
- **组件/标签知识库**：按 Spark/Flink/Kafka/Hadoop/Hive 聚合，支持 tag / component 过滤
- **向量检索**：BGE Embedding + FAISS（本地持久化）或 Milvus（服务化）
- **稀疏检索**：BM25（持久化到磁盘）
- **混合召回与排序**：Dense + BM25 并行召回，使用 **RRF** 融合排序（可调常数 c）
- **工程化**：索引落盘、结果缓存（diskcache）、Docker 化部署（可选 Milvus）
- **输出**：答案 + 标准化排障 SOP（结构化 JSON）

---

## 🧱 目录结构

```
data-platform-rag-assistant/
├── app/                    # FastAPI 服务
├── rag/                    # 核心：数据/索引/检索/链路
├── scripts/                # CLI 脚本入口
├── storage/                # 默认索引落盘目录（运行后生成）
├── docker/                 # Dockerfile & compose
└── requirements.txt
```

---

## 0) 环境准备


## 0) 前置准备（Conda 环境推荐）

### 0.1 安装 Miniconda / Mambaforge（任选其一）
- 推荐 `mamba`（更快），没有也可以用 `conda`。

### 0.2 创建并激活环境

```bash
# 进入项目根目录
conda env create -f environment.yml
conda activate stackrag-dp
```

> 如果你更倾向 “conda + pip” 混装，也可以：
> ```bash
> conda create -n stackrag-dp python=3.11 -y
> conda activate stackrag-dp
> conda install -c conda-forge faiss-cpu numpy pandas lxml beautifulsoup4 tqdm fastapi uvicorn pydantic python-dotenv diskcache orjson typer -y
> pip install -r requirements.pip.txt
> ```

### 0.3 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY（或你使用的 OpenAI-compatible 网关）
```

### 0.4 硬件/磁盘建议（你这个 CSV 数据很大）
- 建议预留：**至少 20GB 磁盘空间**（数据解压 + 中间 JSONL + 索引）
- 内存：建议 **16GB+**（如果机器小，构建数据时用 `--max-questions` 先跑通）


### 0.5 纯 pip/venv 方式（可选）

### 0.1 Python 方式（推荐先跑通）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cp .env.example .env
```

### 0.2 数据集准备（Kaggle StackSample）

1) 进入 Kaggle 数据集页面下载（搜索 “StackSample”）。  
2) 解压后将 `Posts.xml` 放到：`data/raw/Posts.xml`

> 本仓库不包含数据文件（需自行下载并遵守 Kaggle/StackOverflow 数据许可）。

---

## 1) 一键构建：数据 → 索引

### 1.1 生成知识库数据（JSONL）

```bash
python -m scripts.cli build-dataset \
  --posts data/raw/Posts.xml \
  --out data/processed/stack_qa.jsonl \
  --components spark flink kafka hadoop hive \
  --max-questions 200000
```

### 1.2 构建索引（FAISS + BM25）

```bash
python -m scripts.cli build-index \
  --data data/processed/stack_qa.jsonl \
  --backend faiss \
  --storage storage \
  --chunk-size 900 --chunk-overlap 150
```

### 1.3 启动服务

```bash
python -m scripts.cli serve --host 0.0.0.0 --port 8000
# 打开：http://localhost:8000/docs
```

---

## 2) Milvus 方式（可选）

启动 Milvus：

```bash
docker compose -f docker/compose.milvus.yml up -d
```

构建索引：

```bash
python -m scripts.cli build-index \
  --data data/processed/stack_qa.jsonl \
  --backend milvus \
  --storage storage
```

启动服务：

```bash
VECTOR_BACKEND=milvus python -m scripts.cli serve --host 0.0.0.0 --port 8000
```

---

## 3) API 使用

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "question":"Spark job stuck at stage 2, executor memory keeps OOM, how to troubleshoot?",
    "components":["spark"],
    "tags":["apache-spark","spark-streaming"],
    "top_k":6
  }'
```

---

## License

代码：MIT（示例工程）。数据集许可请遵循 Kaggle/StackOverflow 原始许可。


---

## ✅ 适配 Kaggle StackSample（CSV 三表）的数据构建方式（你截图这种）


### 1.0 数据文件注意事项（CSV / Excel）
- `Questions.csv/Answers.csv/Tags.csv` 通常可用 Excel 打开，但**建议保持为 UTF-8 编码 CSV**（避免解析乱码）。
- 文件很大时，脚本会采用 **分块读取（chunksize）**（安装了 pandas 会更快）。


Kaggle 版 StackSample 通常是三个 CSV：
- `Questions.csv`：问题（Id, Title, Body, CreationDate, ClosedDate, Score, OwnerUserId）
- `Answers.csv`：回答（ParentId 指向 Questions.Id）
- `Tags.csv`：标签（Id=Questions.Id, Tag=标签名，一题多行）

本仓库已新增 CSV 版本构建脚本：`build-dataset-csv`

### 1) 生成知识库数据（JSONL）

```bash
python -m scripts.cli build-dataset-csv \
  --questions data/raw/Questions.csv \
  --answers data/raw/Answers.csv \
  --tags data/raw/Tags.csv \
  --out data/processed/stack_qa.jsonl \
  --components spark flink kafka hadoop hive
```

> 注意：Kaggle CSV 不包含 AcceptedAnswerId，因此本实现会为每个问题选择 **score 最高**的回答作为答案。
