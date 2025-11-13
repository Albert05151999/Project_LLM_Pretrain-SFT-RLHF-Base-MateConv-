import csv
import re
import jsonlines
import psutil
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------- ✅ 配置区 ----------
CONTAIN_HISTORY = False  # 是否包含历史记录
CHUNK_SIZE = 50000  # 每批写入行数

# Tokenizer 和数据路径配置
TOKENIZER_PATH = './model/mateconv_tokenizer'

# 指定目标数量（基于估算，一条样本约 2~3KB）
CHINESE_TARGET_COUNT = 6_000_000  # 大约1/2
ENGLISH_TARGET_COUNT = 950_000  # 大约1/3

CHINESE_JSONL_PATH = './deepctrl-sft-data/sft_data_zh.jsonl'
ENGLISH_JSONL_PATH = './deepctrl-sft-data/sft_data_en.jsonl'

# 输出路径配置
OUTPUT_DIR = './data'
OUTPUT_FILENAME = 'sft_data_mixed.csv' if CONTAIN_HISTORY else 'sft_data_mixed_single.csv'
OUTPUT_PATH = f'{OUTPUT_DIR}/{OUTPUT_FILENAME}'

# ---------- 🔁 加载分词器 ----------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
print('✅ tokenizer词表大小：', len(tokenizer))


def log_memory_usage():
    """打印内存使用情况。"""
    mem = psutil.virtual_memory()
    print(f"[MEMORY] Used: {mem.used / 1e9:.2f} GB / {mem.total / 1e9:.2f} GB")


def chinese_ratio(text):
    """计算文本中中文字符的比例。"""
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(chinese_chars) / len(text) if text else 0


def process_and_write_data(data, file_path):
    """处理数据并写入 CSV 文件。"""
    q_lst, a_lst, history_lst = [], [], []

    for per in data:
        history = per.get('history', '')
        q = (per.get('q') or '').strip()
        a = (per.get('a') or '').strip()

        # 过滤无效数据
        if (CONTAIN_HISTORY and (not history or len(history) == 0)) or not q or not a:
            continue
        if len(q) < 10 or len(a) < 5:
            continue
        if len(q) > 256 or len(a) > 256:
            continue
        if not (chinese_ratio(q) > 0.9 and chinese_ratio(a) > 0.9) and 'zh' in file_path:
            continue
        if not (chinese_ratio(q) < 0.1 and chinese_ratio(a) < 0.1) and 'en' in file_path:
            continue

        q_lst.append(q)
        a_lst.append(a)
        history_lst.append(history if CONTAIN_HISTORY else [])

    # 创建并写入 DataFrame
    df = pd.DataFrame({'history': history_lst, 'q': q_lst, 'a': a_lst})
    df.to_csv(file_path, mode='a', header=False, index=False,
              lineterminator='\r\n', escapechar='\\', quoting=csv.QUOTE_MINIMAL)


def sft_process():
    """按比例抽样中文和英文 JSONL 文件，清洗并写入合并 CSV。"""
    data_sources = [
        {"path": CHINESE_JSONL_PATH, "target_count": CHINESE_TARGET_COUNT},
        {"path": ENGLISH_JSONL_PATH, "target_count": ENGLISH_TARGET_COUNT}
    ]

    # 创建 CSV 并写入表头
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('history,q,a\n')

    for src in data_sources:

        with open(src['path'], 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        selected = 0
        valid_buffer = []

        print(f"📂 开始处理文件：{src['path']}，目标抽取 {src['target_count']} 条")
        with jsonlines.open(src['path']) as reader:
            for obj in tqdm(reader, desc=f"Sampling from {src['path']}",total=total_lines):
                q = (obj.get('input') or '') + (obj.get('q') or '')
                a = (obj.get('output') or '') + (obj.get('a') or '')
                history = obj.get('history', '')

                # 清洗逻辑 - 只要满足下列任何条件，就不保留这条样本
                if (CONTAIN_HISTORY and (not history or len(history) == 0)) or not q or not a:
                    continue
                if len(q) < 10 or len(a) < 5 or len(q) > 512 or len(a) > 512:
                    continue
                if not (chinese_ratio(q) > 0.7 and chinese_ratio(a) > 0.7) and 'zh' in src['path']:
                    continue
                if not (chinese_ratio(q) < 0.1 and chinese_ratio(a) < 0.1) and 'en' in src['path']:
                    continue

                valid_buffer.append({'history': history if CONTAIN_HISTORY else [], 'q': q, 'a': a})
                selected += 1

                if len(valid_buffer) >= CHUNK_SIZE:
                    process_and_write_data(valid_buffer, OUTPUT_PATH)
                    valid_buffer = []

                if selected >= src['target_count']:
                    break

            if valid_buffer:
                process_and_write_data(valid_buffer, OUTPUT_PATH)

        print(f"✅ 完成：{src['path']} 实际采样：{selected} 条")

    log_memory_usage()
    print("🎉 数据抽样与写入完成！")

if __name__ == "__main__":
    sft_process()