# -*- coding: utf-8 -*-
"""灰鸦渡 LoRA 合并脚本（独立微调路线 步骤 5/6）

把训练产出的 LoRA 权重合并回 Qwen2.5-3B 基座，输出完整 bf16 Safetensors：
  finetune/output/lora_adapter/  ->  finetune/output/merged_model/

为什么全程用 CPU 而不碰 GPU：
  合并需要以 bf16 完整加载 3.09B 参数（约 6.2GB），4GB 显存根本放不下；
  而合并本身只是 W += (B @ A) * scaling 的权重加法，CPU 上数十秒即可完成，
  完全不需要 GPU。这样也不会与训练/推理抢占显存。

前置条件（重要，脚本会强制预检）：
  必须在训练进程结束后再运行。实测本机 15.7GB 总内存，训练期间仅剩 5.2GB
  可用，不足以容纳 6.2GB 的 bf16 基座，强行执行会因内存不足失败或剧烈换页。

关于精度：
  默认按 bf16 合并（与训练时的 compute_dtype 一致）。虽然 bf16 尾数只有 8 位，
  但下游推理还要再做 4-bit 量化，量化带来的精度损失远大于合并本身，
  因此没有必要为了合并去占 12.4GB 内存加载 fp32 基座。

用法：
  python finetune/scripts/merge_lora.py
  python finetune/scripts/merge_lora.py --adapter finetune/output/ckpt/checkpoint-500
  python finetune/scripts/merge_lora.py --dtype float16
  python finetune/scripts/merge_lora.py --force        # 覆盖已存在的输出目录
"""
import argparse
import ctypes
import gc
import json
import os
import shutil
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# 必须在导入 torch 之前设置：规避 OpenMP 双运行时冲突与 transformers 5.x 联网挂起
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 合并只用 CPU，明确屏蔽 CUDA，避免误占显存或触发驱动初始化开销
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

CONFIG_PATH = os.path.join(ROOT, "finetune", "config", "lora_config.json")
DTYPE_MAP = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}
BYTES_PER_PARAM = {"bfloat16": 2, "float16": 2, "float32": 4}


def abs_path(p):
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(ROOT, p))


class Tee:
    """同时输出到控制台与日志文件"""

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
# 环境预检
# ============================================================
def free_ram_gb():
    """可用物理内存（GB）。Windows 走 GlobalMemoryStatusEx，其他平台退化为 psutil/估算"""
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullAvailPhys / 1024 ** 3, st.ullTotalPhys / 1024 ** 3
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.available / 1024 ** 3, vm.total / 1024 ** 3
    except Exception:
        return float("inf"), float("inf")


def base_model_size_gb(base_model):
    """从 safetensors 索引读取基座权重的真实字节数，读不到则退回经验值"""
    idx = os.path.join(base_model, "model.safetensors.index.json")
    if os.path.isfile(idx):
        try:
            total = json.load(open(idx, encoding="utf-8"))["metadata"]["total_size"]
            return total / 1024 ** 3
        except Exception:
            pass
    single = os.path.join(base_model, "model.safetensors")
    if os.path.isfile(single):
        return os.path.getsize(single) / 1024 ** 3
    return 6.2  # Qwen2.5-3B bf16 经验值


def preflight(base_model, dtype_name, min_free_gb, log):
    """内存与输入预检：宁可提前拒绝，也不要在加载到一半时 OOM"""
    need_gb = base_model_size_gb(base_model)
    scale = BYTES_PER_PARAM[dtype_name] / 2.0
    need_gb *= scale
    # 序列化与 PEFT 包装还需要额外余量
    require_gb = need_gb * 1.35

    free, total = free_ram_gb()
    log(f"[预检] 基座权重约 {need_gb:.2f}GB ({dtype_name})，含余量需约 {require_gb:.2f}GB 可用内存")
    log(f"[预检] 当前内存: 可用 {free:.2f}GB / 总量 {total:.2f}GB")

    if min_free_gb is not None:
        require_gb = min_free_gb
        log(f"[预检] 使用 --min-free-gb 覆盖阈值: {require_gb:.2f}GB")

    if free < require_gb:
        log(f"[预检] 内存不足（需 {require_gb:.2f}GB，仅 {free:.2f}GB）。")
        log("[预检] 最可能的原因是训练进程仍在运行并占用内存。")
        log("[预检] 请等训练结束后重试；确认无训练进程可查：Get-Process python")
        return False

    disk_free = shutil.disk_usage(ROOT).free / 1024 ** 3
    if disk_free < need_gb * 1.5:
        log(f"[预检] 磁盘空间不足：需约 {need_gb*1.5:.1f}GB，仅剩 {disk_free:.1f}GB")
        return False
    log(f"[预检] 磁盘可用 {disk_free:.1f}GB，通过")
    return True


# ============================================================
# 合并生效校验：防止 adapter 名字不匹配导致的静默空合并
# ============================================================
def probe_lora_weight(model):
    """取第一个带 base_layer 的 LoRA 模块，返回 (模块名, 权重和, 权重范数)"""
    for name, mod in model.named_modules():
        base = getattr(mod, "base_layer", None)
        if base is not None and hasattr(base, "weight"):
            w = base.weight.detach().float()
            return name, w.sum().item(), w.norm().item()
    return None, None, None


def probe_plain_weight(model, name):
    """合并后按同名路径取权重（LoRA 包装已卸除，模块直接持有 weight）"""
    for mod_name, mod in model.named_modules():
        if mod_name == name and hasattr(mod, "weight"):
            w = mod.weight.detach().float()
            return w.sum().item(), w.norm().item()
    return None, None


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="灰鸦渡 LoRA 合并 → Safetensors")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter 目录（默认取配置 output.lora_adapter）")
    ap.add_argument("--out", default=None,
                    help="合并输出目录（默认取配置 output.merged_model）")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP),
                    help="合并与保存精度，默认 bfloat16")
    ap.add_argument("--min-free-gb", type=float, default=None,
                    help="覆盖内存预检阈值（GB）")
    ap.add_argument("--force", action="store_true", help="输出目录已存在时先删除")
    cli = ap.parse_args()

    cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
    out_cfg = cfg["output"]
    base_model = cfg["base_model"]
    adapter_dir = abs_path(cli.adapter or out_cfg["lora_adapter"])
    out_dir = abs_path(cli.out or out_cfg["merged_model"])
    dtype_name = DTYPE_MAP[cli.dtype]

    log_dir = os.path.dirname(abs_path(out_cfg.get("log_file", "finetune/logs/train.log")))
    log = Tee(os.path.join(log_dir, "merge.log"))

    log("=" * 66)
    log("灰鸦渡 LoRA 合并（步骤 5/6）")
    log("=" * 66)
    log(f"[路径] 基座   : {base_model}")
    log(f"[路径] adapter: {adapter_dir}")
    log(f"[路径] 输出   : {out_dir}")
    log(f"[参数] dtype={dtype_name}  device=CPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")

    # --- adapter 完整性检查 ---
    if not os.path.isdir(adapter_dir):
        log(f"[错误] adapter 目录不存在: {adapter_dir}")
        log("[提示] 训练是否已完成？产物应在 output.lora_adapter；"
            "也可用 --adapter 指定 finetune/output/ckpt/checkpoint-XXX")
        return 1
    for required in ("adapter_config.json",):
        if not os.path.isfile(os.path.join(adapter_dir, required)):
            log(f"[错误] adapter 目录缺少 {required}，不是有效的 PEFT 输出: {adapter_dir}")
            return 1
    has_weights = any(
        os.path.isfile(os.path.join(adapter_dir, f))
        for f in ("adapter_model.safetensors", "adapter_model.bin")
    )
    if not has_weights:
        log(f"[错误] adapter 目录缺少权重文件 (adapter_model.safetensors/.bin)")
        return 1

    acfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8"))
    log(f"[adapter] r={acfg.get('r')} alpha={acfg.get('lora_alpha')} "
        f"target={acfg.get('target_modules')} base={acfg.get('base_model_name_or_path')}")
    if acfg.get("base_model_name_or_path") and \
            os.path.normpath(acfg["base_model_name_or_path"]) != os.path.normpath(base_model):
        log(f"[警告] adapter 记录的基座路径与配置不一致，仍按配置 {base_model} 合并")

    # --- 输出目录处理 ---
    if os.path.isdir(out_dir):
        if not cli.force:
            log(f"[错误] 输出目录已存在: {out_dir}")
            log("[提示] 加 --force 覆盖，或用 --out 指定其他目录")
            return 1
        log(f"[输出] --force：删除已存在的 {out_dir}")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- 预检 ---
    if not preflight(base_model, dtype_name, cli.min_free_gb, log):
        return 2

    # --- 延迟导入：预检失败时不必付出加载 torch 的代价 ---
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch_dtype = getattr(torch, dtype_name)
    log(f"[加载] 以 {dtype_name} 在 CPU 上加载基座（low_cpu_mem_usage 避免双份内存）")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    base.eval()
    log(f"[加载] 基座就绪，耗时 {time.time()-t0:.0f}s")

    free_after, _ = free_ram_gb()
    log(f"[加载] 加载后剩余内存 {free_after:.2f}GB")

    log(f"[合并] 挂载 adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir, dtype=torch_dtype)

    name, sum_before, norm_before = probe_lora_weight(model)
    if name is None:
        log("[错误] 未在模型中找到任何 LoRA 层，adapter 可能未正确挂载")
        return 3
    log(f"[合并] 探针层 {name}: sum={sum_before:.6f} norm={norm_before:.6f}")

    t1 = time.time()
    with torch.no_grad():
        merged = model.merge_and_unload()
    log(f"[合并] merge_and_unload 完成，耗时 {time.time()-t1:.0f}s")

    # --- 校验合并确实生效 ---
    # probe_lora_weight 给的是 PeftModel 包装下的全路径（base_model.model.model.layers.…），
    # 而 merge_and_unload 卸掉包装后模块名少了前两层，拿原名去查必然落空 —— 实测表现是
    # 一句「合并后找不到探针层，无法校验」的假警告，空合并检测形同虚设。
    plain_name = name
    for prefix in ("base_model.model.", "base_model."):
        if plain_name.startswith(prefix):
            plain_name = plain_name[len(prefix):]
            break
    sum_after, norm_after = probe_plain_weight(merged, plain_name)
    if sum_after is None:
        log(f"[警告] 合并后找不到探针层 {plain_name}（包装期原名 {name}），无法校验；"
            "请人工确认输出")
    else:
        delta = abs(norm_after - norm_before)
        rel = delta / max(norm_before, 1e-9)
        log(f"[校验] 探针层 {plain_name}: sum={sum_after:.6f} norm={norm_after:.6f}")
        log(f"[校验] 权重范数变化 绝对={delta:.6e} 相对={rel:.3%}")
        if rel < 1e-6:
            log("[错误] 合并前后权重几乎无变化 —— LoRA 增量未写入基座（空合并）。")
            log("[提示] 常见原因：adapter 与基座不匹配、target_modules 未命中、"
                "或 adapter 权重全零（训练未真正更新）。已中止，未写出产物。")
            return 4
        log("[校验] 合并生效，权重已改变")

    # --- 清理可能残留的量化配置，否则下游 4-bit 加载会冲突 ---
    if getattr(merged.config, "quantization_config", None) is not None:
        log("[清理] 移除 config 中残留的 quantization_config（合并产物应为全精度）")
        delattr(merged.config, "quantization_config")

    log(f"[保存] 写出 Safetensors 到 {out_dir}")
    t2 = time.time()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    log(f"[保存] 完成，耗时 {time.time()-t2:.0f}s")

    del merged, model, base
    gc.collect()

    # --- 产物盘点 ---
    total_bytes = 0
    log("-" * 66)
    for fn in sorted(os.listdir(out_dir)):
        fp = os.path.join(out_dir, fn)
        if os.path.isfile(fp):
            mb = os.path.getsize(fp) / 1024 ** 2
            total_bytes += os.path.getsize(fp)
            log(f"[产物] {fn:<42} {mb:9.1f} MB")
    log("-" * 66)
    log(f"[产物] 合计 {total_bytes/1024**3:.2f} GB，目录: {out_dir}")

    saved_cfg = os.path.join(out_dir, "config.json")
    if os.path.isfile(saved_cfg):
        sc = json.load(open(saved_cfg, encoding="utf-8"))
        has_q = "quantization_config" in sc
        log(f"[产物] config.json: arch={sc.get('architectures')} "
            f"dtype={sc.get('dtype') or sc.get('torch_dtype')} "
            f"quantization_config={'有(异常!)' if has_q else '无(正确)'}")
        if has_q:
            log("[警告] 输出 config.json 仍带 quantization_config，下游加载可能出错")

    log("[下一步] 运行 python finetune/scripts/test_finetune.py 做回归测试（步骤 6/6）")
    log.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
