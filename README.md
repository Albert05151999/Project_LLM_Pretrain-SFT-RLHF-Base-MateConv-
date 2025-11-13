# 🧠 MateConv 小模型预训练 - SFT - RLHF 项目

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Framework](https://img.shields.io/badge/DeepSpeed-Enabled-orange.svg)]()
[![WandB](https://img.shields.io/badge/Tracking-W%26B-yellow.svg)]()

---

## 📘 项目简介

本项目旨在研究 **小规模语言模型（≈0.1B 参数）** 在使用 **Mixture of Experts (MoE)** 架构下的表现，
并与传统 **Feed-Forward Network (FFN)** 架构进行对比，验证 MoE 架构在生成任务上的潜在优势。

研究目标包括：

- 跑通 **Pretrain → SFT → DPO** 的完整训练流程；  
- 观察 MoE 与 FFN 在生成质量与负载均衡上的差异；  
- 探讨小模型在 DPO 下的 **灾难性遗忘 (Catastrophic Forgetting)** 现象。

> 模型基础：`MateConv-0.02B`（LLaMA 架构、RoPE 位置编码、RMSNorm、KV-Cache 推理优化）  
> 改造后启用 **DeepSeek 风格 MoE 并行 FFN 层**，整体规模约为 **0.1B 参数**。

---

## 🧩 数据与模型配置

| 阶段 | 使用数据集 | 来源 |
|------|-------------|------|
| Tokenizer | HuggingFace 开源小词表（6400词） | HuggingFace |
| 预训练 (Pretrain) | 中文通用语料（31GB） | [序列猴子开源数据集](https://github.com/mobvoi/seq-monkey-data/blob/main/docs/pretrain_open_corpus.md) |
| SFT 微调 | 匠数科技 DeepCtrl SFT 数据集 | [ModelScope: deepctrl-sft-data](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/files) |
| RLHF / DPO | 由 GPT-4o 自动生成偏好对 (正/反样本) | 通过 `7.gen_prefs_openai.py` 自动生成 |

---

## ⚙️ 环境配置

### 1️⃣ 基础环境
```bash
pip install -r requirements.txt
```

需包含：
- `torch`, `transformers`, `deepspeed`, `wandb`, `tqdm`, `datasets`
- Linux GPU 环境（推荐 2 张显卡）

### 2️⃣ WandB 环境变量
```bash
export WANDB_API_KEY="你的_API_KEY"
export WANDB_PROJECT="MateConv_MoE"
```

---

## 🗂️ 数据下载

### ✅ 小数据集（百度网盘）
```bash
./BaiduPCS-Go login -bduss=你的bduss
./BaiduPCS-Go cd /path/to/data
nohup ./BaiduPCS-Go d "/path/to/data" -saveto /root/dataset > /root/bpcs.log 2>&1 &
tail -f /root/bpcs.log
```

### ✅ 大数据集（推荐方式）
使用 `hfd + aria2c + git-lfs` 下载。

---

## 🔧 数据预处理（Notebook）

1️⃣ `1.Tokenizer_Training.ipynb` — tokenizer 训练  
2️⃣ `2.Prepare_Train_Data.ipynb` — 数据清洗与格式化  
3️⃣ `3.pretrain.py` — 预训练数据加载与构建  

后台运行：
```bash
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root > jlab.log 2>&1 &
tail -f jlab.log
```

---

## 🚀 训练流程

### 🧱 (1) 预训练 Pretrain
```bash
mkdir -p logs
nohup deepspeed --master_port 29500 --num_gpus=2 4.pretrain.py --epochs 15   > logs/train.log 2>&1 & echo $! > train.pid
tail -f logs/train.log
```

停止训练：
```bash
kill -2 $(ps -o pgid= -p $(cat train.pid) | tr -d ' ')
```

---

### 🧩 (2) SFT 全量微调
```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 nohup deepspeed --num_gpus=2 5.full_sft.py --out_dir out --epochs 5   > logs/sft.log 2>&1 & echo $! > sft.pid
tail -f logs/sft.log
```

---

## 📊 WandB 监控 MoE 负载均衡

### 预训练阶段
```bash
export WANDB_PROJECT="MateConv_MoE-pretrain"
nohup deepspeed --master_port 29500 --num_gpus=2 4.pretrain.py   --epochs 15 --use_wandb --wandb_project "$WANDB_PROJECT"   > logs/pretrain.log 2>&1 & echo $! > pretrain.pid
```

### SFT 阶段
```bash
export WANDB_PROJECT="MateConv_MoE-sft"
nohup deepspeed --master_port 29500 --num_gpus=2 5.full_sft.py   --out_dir out --epochs 5 --use_wandb --wandb_project "$WANDB_PROJECT"   > logs/sft.log 2>&1 & echo $! > sft.pid
```

---

## 🧹 WandB 缓存自动清理（每半小时）

```bash
mkdir -p /root/autodl-tmp
nohup bash -c '
while true; do
  date
  find "/root/.cache/wandb/artifacts" -type f -mmin +30 -delete || true
  find "/root/.cache/wandb/artifacts" -type d -empty -delete || true
  sleep 3600
done
' >> /root/autodl-tmp/wandb_clean.log 2>&1 &
```

查看：
```bash
tail -f /root/autodl-tmp/wandb_clean.log
```

---

## 🧮 推理与评估

在完成 SFT 后，可使用：
```bash
6.inference&evaluate.ipynb
```
评估不同架构（MoE vs FFN）在生成质量上的差异。

---

## 🧩 DPO 训练流程

### 参数设定
| 项目 | 值 |
|------|----|
| 生成模型 | GPT-4o, temperature=0.7 |
| 评审模型 | GPT-4o-mini, temperature=0.0 |
| N_PROMPTS | 3000 |
| K_SAMPLES | 4 |
| β | 0.2 |
| KL 系数 | 0.05 |

### 数据生成
```bash
export OPENAI_API_KEY="你的_API_KEY"
nohup python 7.gen_prefs_openai.py > logs/gen_prefs.log 2>&1 & echo $! > gen_prefs.pid
tail -f logs/gen_prefs.log
```

### DPO 训练
```bash
export WANDB_PROJECT="MateConv_DPO"
export WANDB_API_KEY="你的_API_KEY"
nohup deepspeed --master_port 29500 --num_gpus=2 9.train_dpo.py   --out_dir out_dpo   --pairs_path ./dataset/dpo_pairs.jsonl   --epochs 20 --max_steps 1800 --batch_size 16 --learning_rate 1e-4   --beta 0.2 --kl_coef 0.05 --warmup_ratio 0.06   --accumulation_steps 2 --grad_clip 1.0 --use_wandb   > logs/dpo_train.log 2>&1 & echo $! > dpo_train.pid
tail -f logs/dpo_train.log
```

---

## 📈 实验总结

- MoE 架构在小模型中表现出轻微但稳定的生成提升；
- DPO 阶段在 0.1B 规模下存在灾难性遗忘风险；
- WandB 可有效观察各个专家（Experts）的负载均衡情况；
- 整体流程成功跑通 Pretrain → SFT → DPO 的端到端实验链路。

---

## 📜 许可证

本项目基于 **MIT License** 开源协议发布。  
如需引用或二次开发，请注明原始仓库链接：

> 🔗 [Project_LLM_Pretrain-SFT-RLHF-Base-MateConv-](https://github.com/Albert05151999/Project_LLM_Pretrain-SFT-RLHF-Base-MateConv-.git)

---

## 🧭 作者信息

**Author:** Albert Lee  
**Institution:** National University of Singapore  
**Reference:** 赋范空间 MateConv开源0.02B中文模型 https://kq4b3vgg5b.feishu.cn/docx/R6aJdgo0mo2Tb1xBy05cEAcen9Y