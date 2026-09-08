# -*- coding: utf-8 -*-
"""灰鸦渡 NPC QLoRA 微调训练脚本（独立微调路线 步骤 4/6）

基座：Qwen2.5-3B-Instruct (D:\\Model)
方案：4-bit NF4 量化 + LoRA(r=16, alpha=32, all-linear) + completion-only loss
数据：finetune/data/train.jsonl (5085 条 messages 格式)
产物：finetune/output/lora_adapter/  (LoRA 权重，步骤 5 再合并为 Safetensors)

关键实现点（针对当前环境 transformers 5.15 / peft 0.20 / RTX 3050 Ti 4GB）：
  1. completion-only loss：prompt(system+user) 段 labels 置 -100，只学 NPC 的回答，
     避免模型去拟合玩家提问与系统人设，显著改善角色扮演质量。
  2. 【显存关键】尾部 logits 优化：Qwen2ForCausalLM.forward 默认对全部 L 个位置算 logits，
     内置 loss_function 会 .float() 上采样 + .contiguous() 拷贝，词表 151936 下
     单条 564-token 样本仅 logits 链路峰值就约 1.3GB，直接把 4GB 卡挤到共享内存（实测
     峰值 6.15GB，65s/步）。改为传 logits_to_keep=k，只对尾部 completion 段（平均 34 token）
     算 logits，省下约 1.2GB，并自行计算 CE 以保证与内置 loss 数值等价。
  3. 启动时自动做 loss 等价校验，防止掩码错位造成"loss 正常下降但学错位置"的静默错误。
  4. transformers 5.x 已移除 TrainingArguments.warmup_ratio，按总优化步数换算成 warmup_steps。
  5. transformers 5.x 的 apply_chat_template(tokenize=True) 返回 BatchEncoding 而非 id 列表。
  6. expandable_segments 分配器：消除长尾样本造成的显存碎片过预留。
  7. 训练进度实时写入 finetune/logs/progress.json，供外部定时监视（每 3 小时查一次）。

用法：
  python finetune/scripts/train.py --smoke 5   # 冒烟测试：只跑 5 个优化步，验证全链路与速度
  python finetune/scripts/train.py             # 全量训练
  python finetune/scripts/train.py --resume    # 从最近 checkpoint 断点续训
"""
import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# 必须在导入 torch/transformers 之前设置：规避 OpenMP 双运行时冲突与 transformers 5.x 联网挂起
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 消除碎片过预留；若与 bitsandbytes 冲突可外部置空覆盖
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

CONFIG_PATH = os.path.join(ROOT, "finetune", "config", "lora_config.json")
IGNORE_INDEX = -100
DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def abs_path(p):
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(ROOT, p))


# ============================================================
# 日志：同时输出到控制台与 finetune/logs/train.log
# ============================================================
class Tee:
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.fh = open(log_path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# ============================================================
# 数据编码：chat template + completion-only 掩码
# ============================================================
def _to_id_list(out):
    """transformers 5.x：apply_chat_template(tokenize=True) 返回 BatchEncoding"""
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if len(out) and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


def encode_messages(tokenizer, messages, max_length):
    """编码单条对话，prompt 段掩码为 -100，只保留 assistant 段计算 loss

    返回 None 表示该样本应丢弃（无 completion / 截断后 completion 全丢）。
    """
    if messages[-1]["role"] != "assistant":
        return None

    ids_full = _to_id_list(tokenizer.apply_chat_template(messages, tokenize=True))
    ids_prompt = _to_id_list(
        tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    )
    plen = len(ids_prompt)
    if plen >= len(ids_full):
        return None
    # 校验 prompt 确为完整序列前缀，否则掩码位置不可信
    if ids_full[:plen] != ids_prompt:
        return None

    input_ids = ids_full[:max_length]
    cut = min(plen, len(input_ids))
    labels = [IGNORE_INDEX] * cut + input_ids[cut:]
    if all(l == IGNORE_INDEX for l in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def load_encoded_dataset(tokenizer, cfg, log, max_length):
    train_file = abs_path(cfg["data"]["train_file"])
    log(f"[数据] 读取并编码: {train_file}")
    t0 = time.time()

    encoded, skipped, comp_tokens, seq_lens = [], 0, 0, []
    with open(train_file, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                messages = json.loads(line)["messages"]
            except Exception as ex:
                skipped += 1
                log(f"[数据] 第 {ln} 行 JSON 解析失败，跳过: {ex}")
                continue
            rec = encode_messages(tokenizer, messages, max_length)
            if rec is None:
                skipped += 1
                continue
            encoded.append(rec)
            comp_tokens += sum(1 for l in rec["labels"] if l != IGNORE_INDEX)
            seq_lens.append(len(rec["input_ids"]))
            if ln % 1000 == 0:
                log(f"[数据] 已编码 {ln} 行...")

    log(f"[数据] 有效 {len(encoded)} 条，跳过 {skipped} 条，耗时 {time.time()-t0:.0f}s")
    if seq_lens:
        log(f"[数据] 序列长度 min={min(seq_lens)} max={max(seq_lens)} "
            f"mean={sum(seq_lens)/len(seq_lens):.0f}")
    log(f"[数据] 参与 loss 的 completion token 总数: {comp_tokens} "
        f"(平均 {comp_tokens/max(len(encoded),1):.1f} token/条)")
    return encoded


def split_train_eval(encoded, eval_ratio, seed=42):
    """切出小规模验证集用于监控过拟合（eval_ratio<=0 时不切分）"""
    if eval_ratio <= 0:
        return encoded, None
    n_eval = max(1, int(len(encoded) * eval_ratio))
    idx = list(range(len(encoded)))
    random.Random(seed).shuffle(idx)
    eval_idx = set(idx[:n_eval])
    train = [x for i, x in enumerate(encoded) if i not in eval_idx]
    ev = [x for i, x in enumerate(encoded) if i in eval_idx]
    return train, ev


class ChatDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


@dataclass
class PadCollator:
    """右补齐；padding 位 labels 置 -100，不参与 loss"""
    pad_token_id: int

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ============================================================
# 尾部 logits loss：只对 completion 段算 logits，省下 [L x 151936] 的 fp32 开销
# ============================================================
def tail_logits_loss(model, input_ids, attention_mask, labels, num_items_in_batch=None):
    """与模型内置 CE 数值等价，但 lm_head/交叉熵只在尾部 k 个位置上计算

    对齐关系：logits_to_keep=k → 尾部覆盖输入位置 [L-k, L-1]；
    logits[:, :-1] 对应位置 [L-k, L-2]，分别预测 token 位置 [L-k+1, L-1]，
    与 labels[:, -k:][:, 1:] 一一对应。k 取 (L - 批内最早 completion 起点 + 1)，
    保证第一个 completion token 的预测位（其前一位）也被包含。
    """
    seq_len = input_ids.shape[1]
    valid = labels != IGNORE_INDEX
    if not bool(valid.any()):
        # 整个 batch 无 completion（正常数据下不可达）：返回挂在计算图上的 0，避免 Trainer 收到 None
        zero = input_ids.sum().float() * 0.0
        return zero.requires_grad_(True), None

    first_valid = valid.float().argmax(dim=1).min().item()
    k = max(2, min(seq_len - int(first_valid) + 1, seq_len))

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        use_cache=False,
        logits_to_keep=k,
    )
    logits = outputs.logits                       # [B, k, V]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, -k:][:, 1:].contiguous()

    reduction = "sum" if num_items_in_batch else "mean"
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction=reduction,
    )
    if num_items_in_batch:
        loss = loss / num_items_in_batch
    return loss, outputs


class TailLogitsTrainer(Trainer):
    """用尾部 logits loss 替换 Trainer 默认的全量 logits loss"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = tail_logits_loss(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["labels"],
            kwargs.get("num_items_in_batch"),
        )
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """验证只算 loss：尾部 logits 形状随 k 变化，不能进 Trainer 的 gather 逻辑"""
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, _ = tail_logits_loss(
                model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
            )
        return (loss.detach(), None, None)


def model_device(model):
    """取模型参数所在设备（4-bit 量化 + LoRA 包装后仍应落在 cuda:0）"""
    for p in model.parameters():
        return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def verify_loss_equivalence(model, records, collator, log, n=3):
    """校验尾部 logits loss 与模型内置全量 loss 数值一致

    这是防止"掩码错位但 loss 照样下降"这类静默错误的必要闸门。
    必须在 eval 模式下比对：lora_dropout=0.05 会让两次前向结果不可比。
    注意：本函数绕过 Trainer，collator 产出的是 CPU 张量，必须自行搬到模型所在设备，
    否则 embedding 查表会报 "index is on cpu, different from other tensors on cuda:0"。
    """
    log(f"[校验] 比对 尾部logits loss vs 内置全量 loss（{min(n, len(records))} 条样本）")
    device = model_device(model)
    log(f"[校验] 模型设备: {device}")
    was_training = model.training
    model.eval()
    max_diff = 0.0
    try:
        for rec in records[:n]:
            batch = to_device(collator([rec]), device)
            with torch.no_grad():
                ref = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                ).loss.item()
                got, _ = tail_logits_loss(
                    model, batch["input_ids"], batch["attention_mask"], batch["labels"]
                )
                got = got.item()
            diff = abs(ref - got)
            max_diff = max(max_diff, diff)
            L = batch["input_ids"].shape[1]
            ncomp = int((batch["labels"] != IGNORE_INDEX).sum().item())
            log(f"[校验] L={L} completion={ncomp} 内置={ref:.6f} 尾部={got:.6f} 差={diff:.2e}")
    finally:
        if was_training:
            model.train()

    ok = max_diff < 2e-3
    log(f"[校验] 最大偏差 {max_diff:.2e} → {'一致，掩码正确' if ok else '不一致！掩码可能错位'}")
    return ok


# ============================================================
# 模型：4-bit 量化基座 + LoRA
# ============================================================
def build_model(cfg, log):
    qcfg, lcfg, tcfg = cfg["quantization"], cfg["lora"], cfg["training"]
    compute_dtype = DTYPE_MAP[qcfg["bnb_4bit_compute_dtype"]]

    log(f"[模型] 加载 4-bit 量化基座: {cfg['base_model']}")
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        trust_remote_code=True,
        quantization_config=bnb,
        dtype=compute_dtype,
        attn_implementation=tcfg.get("attn_implementation", "eager"),
        device_map={"": 0},
    )
    model.config.use_cache = False  # 与梯度检查点不兼容
    log(f"[模型] 基座加载完成，耗时 {time.time()-t0:.0f}s")

    log("[模型] prepare_model_for_kbit_training (含梯度检查点 + input require grads)")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=lcfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora)
    log(f"[模型] LoRA r={lcfg['r']} alpha={lcfg['lora_alpha']} "
        f"dropout={lcfg['lora_dropout']} target={lcfg['target_modules']}")
    model.print_trainable_parameters()
    return model


# ============================================================
# 进度回调：写 progress.json 供外部定时监视
# ============================================================
class ProgressCallback(TrainerCallback):
    def __init__(self, log, progress_path, t0):
        self.log = log
        self.path = progress_path
        self.t0 = t0
        self.history = []

    def _flush(self, rec):
        rec["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs:
            return
        elapsed = time.time() - self.t0
        done = state.global_step
        total = state.max_steps or 1
        speed = elapsed / max(done, 1)
        eta = speed * max(total - done, 0)
        vram = torch.cuda.max_memory_reserved() / 1024 ** 3 if torch.cuda.is_available() else 0.0
        self._flush({
            "status": "running",
            "step": done,
            "total_steps": total,
            "epoch": round(state.epoch or 0.0, 4),
            "loss": round(logs["loss"], 4),
            "learning_rate": logs.get("learning_rate"),
            "elapsed_sec": round(elapsed, 1),
            "sec_per_step": round(speed, 3),
            "eta_sec": round(eta, 1),
            "eta_hours": round(eta / 3600, 2),
            "peak_vram_gb": round(vram, 2),
            "eval_history": self.history,
        })
        flag = "  ⚠ 超出物理显存" if vram > 4.0 else ""
        self.log(f"[进度] step {done}/{total} ({done/total*100:.1f}%) "
                 f"loss={logs['loss']:.4f} lr={logs.get('learning_rate', 0):.2e} "
                 f"{speed:.2f}s/step 已用 {elapsed/3600:.2f}h 预计剩余 {eta/3600:.2f}h "
                 f"显存峰值 {vram:.2f}GB{flag}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        if "eval_loss" in metrics:
            self.history.append({
                "epoch": round(state.epoch or 0.0, 4),
                "step": state.global_step,
                "eval_loss": round(metrics["eval_loss"], 4),
            })
            self.log(f"[验证] epoch {state.epoch:.2f} eval_loss={metrics['eval_loss']:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        self._flush({
            "status": "finished",
            "step": state.global_step,
            "total_steps": state.max_steps,
            "epoch": round(state.epoch or 0.0, 4),
            "elapsed_sec": round(time.time() - self.t0, 1),
            "eval_history": self.history,
        })


# ============================================================
# TrainingArguments 构建（含 5.x warmup_ratio → warmup_steps 换算）
# ============================================================
def build_args(cfg, log, ckpt_dir, n_train, epochs, max_steps, smoke, has_eval,
               accum_override, save_steps=0):
    tcfg = cfg["training"]
    bs = tcfg["per_device_train_batch_size"]
    accum = accum_override or tcfg["gradient_accumulation_steps"]

    # transformers 5.x 无 warmup_ratio，按总优化步数换算
    if max_steps > 0:
        total_steps = max_steps
        per_epoch = max_steps
    else:
        per_epoch = math.ceil(n_train / (bs * accum))
        total_steps = per_epoch * epochs
    warmup_steps = math.ceil(tcfg.get("warmup_ratio", 0.0) * total_steps)
    log(f"[参数] batch={bs} x accum={accum} (等效 {bs*accum}) | 总优化步数 {total_steps} "
        f"(每 epoch {per_epoch} 步) | warmup_steps={warmup_steps} "
        f"(由 warmup_ratio={tcfg.get('warmup_ratio', 0.0)} 换算)")

    known = {f.name for f in dataclasses.fields(TrainingArguments)}

    # 存档策略：冒烟不存档；显式 --save-steps 优先于配置的 epoch 存档
    if smoke:
        save_strategy, save_steps_val, save_limit = "no", None, None
    elif save_steps and save_steps > 0:
        save_strategy, save_steps_val, save_limit = "steps", save_steps, None
        n_ckpt = max(1, total_steps // save_steps)
        log(f"[参数] 按步存档：每 {save_steps} 步一次，全程约 {n_ckpt} 个回滚点，"
            f"全部保留（单个约 130MB，共约 {n_ckpt * 130 / 1024:.1f}GB 磁盘）")
    else:
        save_strategy, save_steps_val, save_limit = tcfg.get("save_strategy", "epoch"), None, 3

    kwargs = dict(
        output_dir=ckpt_dir,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=accum,
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=tcfg["optim"],
        logging_steps=1 if smoke else tcfg["logging_steps"],
        logging_first_step=True,
        save_strategy=save_strategy,
        save_steps=save_steps_val,
        save_total_limit=save_limit,
        eval_strategy="epoch" if (has_eval and not smoke) else "no",
        prediction_loss_only=True,
        bf16=tcfg.get("bf16", True),
        fp16=False,
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,   # Windows 下避免多进程 spawn 问题
        seed=tcfg.get("seed", 42),
    )
    if smoke:
        kwargs["max_steps"] = max_steps
        kwargs["num_train_epochs"] = 1
    else:
        kwargs["num_train_epochs"] = epochs

    dropped = [k for k in list(kwargs) if k not in known]
    for k in dropped:
        log(f"[参数] 当前 transformers 不支持 {k}，已忽略")
        kwargs.pop(k)
    return TrainingArguments(**kwargs), total_steps


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="灰鸦渡 NPC QLoRA 微调训练")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="冒烟测试：只跑 N 个优化步，产物写入 output/_smoke，不污染正式目录")
    ap.add_argument("--resume", action="store_true", help="从最近 checkpoint 断点续训")
    ap.add_argument("--epochs", type=float, default=None, help="覆盖配置中的 num_train_epochs")
    ap.add_argument("--max-length", type=int, default=None, help="覆盖配置中的 max_length")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="覆盖 gradient_accumulation_steps（调小可减少每步耗时，便于评估速度）")
    ap.add_argument("--save-steps", type=int, default=0, metavar="N",
                    help="按步数存档（覆盖配置里的 save_strategy=epoch）。长跑建议 100，"
                         "约每 2 小时一个回滚点，避免崩溃后损失数小时进度")
    ap.add_argument("--eval-ratio", type=float, default=None,
                    help="验证集比例；默认取配置 data.eval_ratio，配置缺省则为 0（不开验证）")
    ap.add_argument("--skip-verify", action="store_true", help="跳过 loss 等价校验（不建议）")
    cli = ap.parse_args()

    smoke = cli.smoke > 0
    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
    out = cfg["output"]
    log_dir = os.path.dirname(abs_path(out.get("log_file", "finetune/logs/train.log")))
    log_path = os.path.join(log_dir, "train_smoke.log" if smoke else "train.log")
    log = Tee(log_path)

    log("=" * 66)
    log(f"灰鸦渡 QLoRA 训练启动  |  模式: {'冒烟测试 ' + str(cli.smoke) + ' 步' if smoke else '全量'}")
    log("=" * 66)
    log(f"[环境] torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log(f"[环境] GPU={p.name} 物理显存={p.total_memory/1024**3:.2f}GB")
    else:
        log("[环境] 未检测到 CUDA，无法进行 4-bit QLoRA 训练，已中止")
        return 1

    epochs = cli.epochs if cli.epochs is not None else cfg["training"]["num_train_epochs"]
    max_length = cli.max_length or cfg["training"]["max_length"]
    # 用户已从配置移除 eval_ratio，故缺省为 0（不开验证），可用 CLI 显式开启
    eval_ratio = cli.eval_ratio if cli.eval_ratio is not None else cfg["data"].get("eval_ratio", 0.0)

    adapter_dir = abs_path("finetune/output/_smoke/adapter" if smoke else out["lora_adapter"])
    ckpt_dir = abs_path(
        "finetune/output/_smoke/ckpt" if smoke
        else out.get("checkpoint_dir", os.path.join(os.path.dirname(out["lora_adapter"]), "ckpt"))
    )
    progress_path = os.path.join(log_dir, "progress_smoke.json" if smoke else "progress.json")
    os.makedirs(ckpt_dir, exist_ok=True)

    log(f"[路径] checkpoint={ckpt_dir}")
    log(f"[路径] 最终 adapter={adapter_dir}")
    log(f"[参数] epochs={epochs} max_length={max_length} eval_ratio={eval_ratio}")

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tokenizer.padding_side = "right"          # 训练必须右补齐
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"[分词] pad={tokenizer.pad_token!r} eos={tokenizer.eos_token!r} "
        f"padding_side={tokenizer.padding_side}")

    # --- 数据 ---
    encoded = load_encoded_dataset(tokenizer, cfg, log, max_length)
    if not encoded:
        log("[错误] 无有效训练样本，已中止")
        return 1
    train_recs, eval_recs = split_train_eval(encoded, 0.0 if smoke else eval_ratio)
    log(f"[数据] 训练 {len(train_recs)} 条" + (f"，验证 {len(eval_recs)} 条" if eval_recs else "，无验证集"))

    if smoke:
        accum = cli.grad_accum or cfg["training"]["gradient_accumulation_steps"]
        need = cli.smoke * cfg["training"]["per_device_train_batch_size"] * accum
        train_recs = train_recs[: max(need, 1)]
        log(f"[冒烟] 截取 {len(train_recs)} 条样本用于 {cli.smoke} 个优化步")

    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)
    train_ds = ChatDataset(train_recs)
    eval_ds = ChatDataset(eval_recs) if eval_recs else None

    # --- 模型 ---
    model = build_model(cfg, log)

    # --- loss 等价校验（掩码正确性闸门）---
    if not cli.skip_verify:
        longest = sorted(encoded, key=lambda r: -len(r["input_ids"]))[:2]
        if not verify_loss_equivalence(model, longest + train_recs[:1], collator, log, n=3):
            log("[错误] loss 校验未通过，已中止（掩码或对齐有误，继续训练会学到错误位置）")
            return 3
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- 训练参数 ---
    args, total_steps = build_args(
        cfg, log, ckpt_dir, len(train_ds), epochs,
        cli.smoke if smoke else 0, smoke, eval_ds is not None, cli.grad_accum,
        cli.save_steps,
    )
    log(f"[预估] 按当前配置需完成 {total_steps} 个优化步；"
        f"冒烟模式下可直接读出 s/step 以推算总时长")

    # --- 续训 ---
    resume_from = None
    if cli.resume and not smoke:
        resume_from = get_last_checkpoint(ckpt_dir) if os.path.isdir(ckpt_dir) else None
        log(f"[续训] {'从 ' + resume_from + ' 恢复' if resume_from else '未找到 checkpoint，从头开始'}")

    t0 = time.time()
    trainer = TailLogitsTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[ProgressCallback(log, progress_path, t0)],
    )

    log("[训练] 开始 ...")
    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except torch.cuda.OutOfMemoryError:
        log("[错误] 显存溢出 (OOM)。可依次尝试：--grad-accum 8、--max-length 640、"
            "关闭其他占显存进程（如 rag_chain 推理）后重试。")
        return 2
    except KeyboardInterrupt:
        log(f"[中断] 用户中止，已训练 {time.time()-t0:.0f}s；"
            f"checkpoint 保留在 {ckpt_dir}，可用 --resume 续训")
        return 130

    total_sec = time.time() - t0
    log(f"[训练] 完成，总耗时 {total_sec/3600:.2f} 小时")

    # --- 保存最终 adapter ---
    log(f"[产物] 保存 LoRA adapter 到 {adapter_dir}")
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    tr = trainer.state.log_history
    losses = [x["loss"] for x in tr if "loss" in x]
    evals = [x for x in tr if "eval_loss" in x]
    log("-" * 66)
    if losses:
        log(f"[结果] 首个 loss={losses[0]:.4f} → 末个 loss={losses[-1]:.4f}")
    for e in evals:
        log(f"[结果] eval_loss={e['eval_loss']:.4f} @ epoch {e.get('epoch', 0):.2f}")
    log(f"[结果] 显存峰值 {torch.cuda.max_memory_reserved()/1024**3:.2f}GB")
    log(f"[结果] adapter: {adapter_dir}")
    log("[下一步] 运行 python finetune/scripts/merge_lora.py 合并为 Safetensors（步骤 5/6）")
    log.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
