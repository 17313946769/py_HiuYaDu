# -*- coding: utf-8 -*-
"""QLoRA 单样本速度基准：定位 4GB 卡上的真实瓶颈

背景：冒烟测试实测 71.9s/优化步（约 4.5s/样本），显存 4.11GB。
尾部 logits 优化省下 2GB 却没省时间，说明瓶颈在 transformer 主体前向/反向，
而非 lm_head/CE。本脚本对比 attention 实现 × 梯度检查点的组合，
用实测数据决定全量训练该用哪套配置，避免 19 小时的无效长跑。

用法：
  python finetune/scripts/bench_speed.py                  # 跑全部组合
  python finetune/scripts/bench_speed.py --samples 5      # 每个组合多测几次
  python finetune/scripts/bench_speed.py --no-expandable  # 关掉 expandable_segments 对照
  python finetune/scripts/bench_speed.py --only sdpa_ckpt # 只跑指定组合
"""
import argparse
import gc
import json
import os
import sys
import time

# 分配器策略必须在 import torch 之前定，故先手工解析这一个开关
_NO_EXPANDABLE = "--no-expandable" in sys.argv
if _NO_EXPANDABLE:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# 复用 train.py 的真实代码路径，保证基准结果与训练一致
from train import (
    CONFIG_PATH,
    DTYPE_MAP,
    IGNORE_INDEX,
    PadCollator,
    encode_messages,
    model_device,
    tail_logits_loss,
    to_device,
)

PHYSICAL_VRAM_GB = 4.0


def log(msg=""):
    print(msg, flush=True)


def build_variant(cfg, attn_impl, grad_ckpt):
    """按指定 attention 实现与梯度检查点开关构建模型"""
    qcfg, lcfg = cfg["quantization"], cfg["lora"]
    compute_dtype = DTYPE_MAP[qcfg["bnb_4bit_compute_dtype"]]
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
        attn_implementation=attn_impl,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=grad_ckpt,
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
    return get_peft_model(model, lora)


def pick_probe_records(tokenizer, cfg, max_length):
    """取最长与中位长度各一条作为探针，覆盖最坏与典型情况"""
    recs = []
    with open(os.path.join(ROOT, cfg["data"]["train_file"]), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = encode_messages(tokenizer, json.loads(line)["messages"], max_length)
            if r:
                recs.append(r)
    recs.sort(key=lambda r: len(r["input_ids"]))
    longest = recs[-1]
    median = recs[len(recs) // 2]
    return [("长样本", longest), ("中位样本", median)]


def bench_variant(name, cfg, attn_impl, grad_ckpt, probes, tokenizer, samples, log):
    log("-" * 70)
    log(f"[{name}] attn={attn_impl} grad_ckpt={grad_ckpt}")
    try:
        model = build_variant(cfg, attn_impl, grad_ckpt)
    except Exception as ex:
        log(f"[{name}] 模型构建失败: {ex}")
        return None

    device = model_device(model)
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    results = {}
    try:
        for tag, rec in probes:
            batch = to_device(collator([rec]), device)
            L = batch["input_ids"].shape[1]
            ncomp = int((batch["labels"] != IGNORE_INDEX).sum().item())

            # 预热一次，触发 CUDA 上下文与 kernel 缓存
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = tail_logits_loss(
                    model, batch["input_ids"], batch["attention_mask"], batch["labels"])
                loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            times = []
            for _ in range(samples):
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = tail_logits_loss(
                        model, batch["input_ids"], batch["attention_mask"], batch["labels"])
                    loss.backward()
                torch.cuda.synchronize()
                times.append(time.time() - t0)
                model.zero_grad(set_to_none=True)

            peak = torch.cuda.max_memory_reserved() / 1024 ** 3
            avg = sum(times) / len(times)
            results[tag] = {"L": L, "completion": ncomp, "sec": avg, "vram_gb": peak,
                            "loss": float(loss.item())}
            log(f"    {tag}: L={L} completion={ncomp} → {avg:.3f}s/样本 "
                f"显存峰值={peak:.2f}GB loss={loss.item():.4f}")
    except torch.cuda.OutOfMemoryError:
        log(f"[{name}] OOM，该组合在 4GB 卡上不可用")
        results = None
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser(description="QLoRA 速度基准")
    ap.add_argument("--samples", type=int, default=3, help="每个组合每种样本的计时次数")
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--only", default="", help="只跑名字包含该子串的组合")
    cli = ap.parse_args()

    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
    max_length = cli.max_length or cfg["training"]["max_length"]

    log("=" * 70)
    log("QLoRA 速度基准  |  目标：找出 4GB 卡上最快的可行组合")
    log("=" * 70)
    log(f"[环境] torch={torch.__version__} alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r}")
    p = torch.cuda.get_device_properties(0)
    log(f"[环境] GPU={p.name} 物理显存={p.total_memory/1024**3:.2f}GB")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    probes = pick_probe_records(tokenizer, cfg, max_length)
    for tag, r in probes:
        log(f"[探针] {tag} L={len(r['input_ids'])}")

    variants = [
        ("eager_ckpt", "eager", True),      # 当前配置（基线）
        ("sdpa_ckpt", "sdpa", True),        # 主要候选
        ("sdpa_nockpt", "sdpa", False),
        ("eager_nockpt", "eager", False),
    ]

    summary = {}
    for name, attn, ckpt in variants:
        if cli.only and cli.only not in name:
            continue
        res = bench_variant(name, cfg, attn, ckpt, probes, tokenizer, cli.samples, log)
        if res:
            summary[name] = res

    log("=" * 70)
    log("汇总（s/样本，越低越好；显存须 ≤ 4.00GB 才不溢出到共享内存）")
    log("=" * 70)
    log(f"{'组合':<16}{'长样本 s':>10}{'中位 s':>10}{'长样本GB':>11}{'中位GB':>9}  可行性")
    for name, res in summary.items():
        lo = res.get("长样本")
        me = res.get("中位样本")
        if not lo or not me:
            continue
        peak = max(lo["vram_gb"], me["vram_gb"])
        ok = "显存溢出" if peak > PHYSICAL_VRAM_GB else "OK"
        log(f"{name:<16}{lo['sec']:>10.3f}{me['sec']:>10.3f}"
            f"{lo['vram_gb']:>11.2f}{me['vram_gb']:>9.2f}  {ok}")

    # 用中位样本推算全量训练时长
    if summary:
        n_train = 5085
        accum = cfg["training"]["gradient_accumulation_steps"]
        epochs = cfg["training"]["num_train_epochs"]
        steps = -(-n_train // accum) * epochs
        log("-" * 70)
        log(f"全量时长预估（{steps} 个优化步，按各自中位样本速度 × accum={accum}）：")
        for name, res in sorted(summary.items(), key=lambda kv: kv[1].get("中位样本", {}).get("sec", 9e9)):
            me = res.get("中位样本")
            if not me:
                continue
            hours = me["sec"] * accum * steps / 3600
            peak = max(res.get("长样本", me)["vram_gb"], me["vram_gb"])
            flag = "  ⚠显存溢出" if peak > PHYSICAL_VRAM_GB else ""
            log(f"  {name:<16} ≈ {hours:5.2f} 小时{flag}")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())