'''
读取 ./dataset/sft_data_mixed_single.csv，随机采样 N_PROMPTS 个 q；
用 OPENAI_MODEL 生成 K 个候选；
用 JUDGE_MODEL 评审并打分（1–10，或直接给出排序）；
产出 成对数据：(q, pos, neg, history) 到 ./dataset/dpo_pairs.jsonl
'''
# 1) 全局 Session 复用 + 适度并发
import os, json, time, random, re, threading
import pandas as pd
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===== 设置 OPENAI_API_KEY =====
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

if not OPENAI_API_KEY:
    raise RuntimeError("ENV OPENAI_API_KEY not set")

OPENAI_MODEL = os.getenv("OPENAI_GEN_MODEL", "gpt-4o")
JUDGE_MODEL  = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")
N_PROMPTS    = int(os.getenv("N_PROMPTS", "20000"))
K_SAMPLES    = int(os.getenv("K_SAMPLES", "4"))
OUT_PATH     = Path("./dataset/dpo_pairs.jsonl")

MAX_WORKERS  = int(os.getenv("MAX_WORKERS", "4"))      # 并发度
OPENAI_RPS   = float(os.getenv("OPENAI_RPS", "2"))     # 全局限速：请求/秒
OPENAI_BURST = float(os.getenv("OPENAI_BURST", "2"))   # 突发桶容量
MAX_TRIES    = int(os.getenv("MAX_TRIES", "8"))        # 单请求最大重试
RPS_SLEEP    = float(os.getenv("RPS_SLEEP", "0.0"))    # 每请求后额外小睡(可选)

API_URL = "https://api.openai.com/v1/chat/completions"
assert "OPENAI_API_KEY" in os.environ, "请先设置 OPENAI_API_KEY"

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# ----- 全局 HTTP 会话 -----
SESSION = requests.Session()
retry = Retry(total=0)  # 我们自己做 429/5xx 退避，这里关掉 urllib3 的自动重试
adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=retry)
SESSION.mount("https://", adapter)

# ===== 令牌桶限速器 =====
class RateLimiter:
    def __init__(self, rate_per_sec=2.0, burst=2.0):
        self.rate = float(rate_per_sec)
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.lock = threading.Lock()
        self.last = time.monotonic()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last
            # 回填令牌
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens < 1.0:
                need = 1.0 - self.tokens
                sleep_s = need / self.rate
                time.sleep(sleep_s)
                now = time.monotonic()
                # 消耗一个请求的令牌
                self.tokens = max(0.0, self.tokens + (now - self.last) * self.rate - 1.0)
                self.last = now
            else:
                self.tokens -= 1.0
                self.last = now

GLOBAL_RL = RateLimiter(rate_per_sec=OPENAI_RPS, burst=OPENAI_BURST)

def _explain_rate_limit(resp):
    """打印并返回限流原因（RPM/TPM/未知），便于诊断。"""
    h = resp.headers
    rr  = h.get("x-ratelimit-remaining-requests")
    rt  = h.get("x-ratelimit-remaining-tokens")
    rrr = h.get("x-ratelimit-reset-requests")
    rrt = h.get("x-ratelimit-reset-tokens")
    ra  = h.get("retry-after")

    print("[rate] remain_req=", rr, "remain_tok=", rt,
          "reset_req=", rrr, "reset_tok=", rrt, "retry_after=", ra)

    reason = None
    if resp.status_code == 429:
        # 哪个先到 0/提示重置就更可能是哪条
        if rr == "0" or rrr is not None:
            reason = "RPM 限流（请求数/分钟）"
        if rt == "0" or rrt is not None:
            reason = "TPM 限流（token/分钟）"
        if reason is None:
            reason = "未知（可能是项目/账单/模型配额）"
        print("[rate] reason:", reason)
    return reason

def _post_with_backoff(url, headers, payload, timeout=60, max_tries=8):
    """统一的请求封装：限速 + 429/5xx 退避（含 Retry-After）。"""
    for attempt in range(max_tries):
        try:
            GLOBAL_RL.acquire()
            resp = SESSION.post(url, headers=headers, json=payload, timeout=timeout)

            # 遇到 429 先读 Retry-After，否则做指数退避
            if resp.status_code == 429:
                _explain_rate_limit(resp)
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        wait = float(ra)
                    except Exception:
                        wait = 2 ** attempt + random.uniform(0, 1)
                else:
                    wait = 2 ** attempt + random.uniform(0, 1)
                time.sleep(min(wait, 60.0))
                continue

            # 临时性后端错误也退避
            if resp.status_code in (500, 502, 503, 504):
                wait = min(2 ** attempt + random.uniform(0, 1), 30.0)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            if RPS_SLEEP > 0:
                time.sleep(RPS_SLEEP)
            return resp

        except requests.RequestException as e:
            if attempt == max_tries - 1:
                raise
            wait = min(2 ** attempt + random.uniform(0, 1), 30.0)
            print(f"[WARN] request error: {e}; retry in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError("Exceeded maximum retries")

def chat_once(model, messages, **kwargs):
    payload = {"model": model, "messages": messages}
    payload.update(kwargs)
    resp = _post_with_backoff(API_URL, HEADERS, payload, timeout=60, max_tries=MAX_TRIES)
    return resp.json()["choices"][0]["message"]["content"]

# 2) 一次生成 K 个候选（通用版：让模型在一条回复里输出K段，以分隔符切分）
SEP = "\n\n----\n\n"

def gen_candidates(q, history, k=K_SAMPLES):
    msgs = []
    for turn in (history or []):
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            msgs.append({"role": "user", "content": str(turn[0])})
            msgs.append({"role": "assistant", "content": str(turn[1])})
    msgs.append({"role": "user", "content": (
        f"请为以下问题给出 {k} 个**相互不同**的候选回答。"
        f"用如下分隔符分开：{SEP}\n"
        f"问题：{q}\n请只输出候选内容本身，不要编号。"
    )})

    txt = chat_once(
        OPENAI_MODEL, msgs,
        temperature=0.9, top_p=0.95, max_tokens=800,
        presence_penalty=0.5, frequency_penalty=0.2
    )
    # 切分并清洗
    parts = [p.strip() for p in txt.split(SEP)]
    parts = [p for p in parts if p]
    if len(parts) < k:
        # 备用：按编号切
        more = re.split(r"\n\s*\d+\.\s*", txt)
        more = [m.strip() for m in more if m.strip()]
        parts = parts if len(parts) >= k else more
    return parts[:k]

# 3) 一次评审，直接排序 K 个候选
def judge_rank(q, history, cands):
    cand_block = "\n\n".join([f"[{i+1}]\n{c}" for i, c in enumerate(cands)])
    msgs = [
        {"role": "system", "content": "你是严格的中文评审。请只输出候选编号的排序，例如: 1,3,2,4"},
        {"role": "user", "content": (
            f"问题：{q}\n历史：{history}\n候选如下（编号见方括号）：\n{cand_block}\n"
            f"请给出从好到坏的编号排序，仅输出数字和逗号。"
        )}
    ]
    txt = chat_once(JUDGE_MODEL, msgs, temperature=0.0, max_tokens=64)
    nums = re.findall(r"\d+", txt)
    order = [int(x) for x in nums if 1 <= int(x) <= len(cands)]
    # 兜底：若解析不完整，补足缺的索引
    remain = [i for i in range(1, len(cands)+1) if i not in order]
    order += remain
    return order

def build_pairs_from_order(cands, order, tail_neg=2):
    res = []
    if not order:
        return res
    pos = cands[order[0]-1]
    for idx in order[-tail_neg:]:
        neg = cands[idx-1]
        if neg.strip() and neg.strip() != pos.strip():
            res.append((pos, neg))
    if not res and len(cands) >= 2:
        res.append((cands[0], cands[1]))
    return res

# 4) 单个样本处理（供并发池调用）
def process_one(row):
    q = str(row.get("q", "")).strip()
    hist_raw = row.get("history", "[]")
    try:
        history = eval(hist_raw) if isinstance(hist_raw, str) else (hist_raw or [])
    except Exception:
        history = []

    cands = gen_candidates(q, history, k=K_SAMPLES)
    if len(cands) < 2:
        return []  # 无法成对
    order = judge_rank(q, history, cands)
    pairs = build_pairs_from_order(cands, order, tail_neg=2)

    out = []
    for pos, neg in pairs:
        out.append({"q": q, "history": history, "pos": pos, "neg": neg})
    return out

# 5) 主流程：并发 + 断点追加写入
def main():
    df = pd.read_csv("./dataset/sft_data_mixed_single.csv")
    pool = df.sample(n=min(N_PROMPTS, len(df)), random_state=2025).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with OUT_PATH.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one, row) for _, row in pool.iterrows()]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                recs = fut.result()
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_written += 1
            except Exception as e:
                print(f"[WARN] item#{i} failed: {e}")
            if i % 50 == 0:
                print(f"[{i}/{len(pool)}] 已完成 {i} 个问题，已写入 pairs: {n_written}")

    print(f"完成。生成对数: {n_written} -> {OUT_PATH}")

if __name__ == "__main__":
    main()