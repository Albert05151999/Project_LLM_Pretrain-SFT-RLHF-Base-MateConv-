'''
把 jsonl 的 (q, pos, neg, history) 转成 三路 token（prompt / pos / neg），
并返回 (X, Y_pos, Y_neg, mask_pos, mask_neg)，与 SFT 的 Loss Mask 设计保持一致（只对回答段计损失）
'''
import json, re, numpy as np, torch
from torch.utils.data import Dataset
from pathlib import Path

class DpoDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        super().__init__()
        self.path = Path(jsonl_path)
        self.items = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.tok = tokenizer
        self.max_length = max_length
        self.pad_id = 0
        self.bos_id_seq = self.tok("<s>assistant").data["input_ids"]  # 与 SFTDataset 相同用法

    def __len__(self):
        return len(self.items)

    def _build_ids_and_mask(self, q, history, a_text):
        # 复用 chat 模版（与训练/评测一致）
        msgs = []
        for h in (history or []):
            if isinstance(h, (list, tuple)) and len(h) >= 2:
                msgs.append({"role": "user", "content": str(h[0])})
                msgs.append({"role": "assistant", "content": str(h[1])})
        msgs.append({"role": "user", "content": str(q)})
        msgs.append({"role": "assistant", "content": str(a_text)})

        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt).data["input_ids"][:self.max_length]

        # 找到 assistant 开始（回答起点）
        def find_sub(main, sub):
            last = -1
            for i in range(0, len(main) - len(sub) + 1):
                if main[i:i+len(sub)] == sub:
                    last = i
            return last

        a_start = find_sub(ids, self.bos_id_seq) + len(self.bos_id_seq)
        pad_len = self.max_length - len(ids)
        ids = ids + [self.pad_id] * pad_len

        # loss mask：仅回答部分参与（去掉最后 token 的对齐）
        mask_len = len(ids) - a_start - pad_len
        loss_mask = [0] * a_start + [1] * mask_len + [0] * pad_len

        arr = np.array(ids, dtype=np.int64)
        X  = arr[:-1]
        Y  = arr[1:]
        LM = np.array(loss_mask[1:], dtype=np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(LM)

    def __getitem__(self, idx):
        it = self.items[idx]
        q, history, pos, neg = it["q"], it.get("history", []), it["pos"], it["neg"]

        X_pos, Y_pos, LM_pos = self._build_ids_and_mask(q, history, pos)
        X_neg, Y_neg, LM_neg = self._build_ids_and_mask(q, history, neg)

        # 注意：X_pos == X_neg（同一 prompt），但为了简单起见各自返回
        return X_pos, Y_pos, LM_pos, X_neg, Y_neg, LM_neg
