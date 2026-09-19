# coding=utf-8
"""``transformers`` 本地后端：论文主路径。

对应论文的两处需求：

* §3.1 数据增强：本地加载 Qwen 7B/13B，按 Prompt 生成 JSON 增强样本；
* §3.3 LLM 对齐：LoRA 微调 + 导出任务向量交给 TIES-Merging 合并。

设计要点
--------
1. **重依赖全部延迟导入**：``torch`` / ``transformers`` / ``peft`` 只在真正构造
   后端时才 import，这样本模块可以被静态检查、也可以在没有 GPU 的机器上被导入；
2. **任务向量以 LoRA 参数为粒度**：微调只更新 LoRA 的 ``lora_A`` / ``lora_B``，
   因此 ``τ = θ_ft - θ_base`` 在 LoRA 上等价于``{A: A_ft - A_init, B: B_ft - B_init}``。
   合并只作用在这两个矩阵上，基座权重始终冻结——这正是 TIES-Merging 在 LoRA
   场景下的标准做法，也避免了 13B 全量任务向量的显存灾难；
3. **批内温度抖动**：同一批样本用略微不同的温度采样，服务论文"多样性提升"目标。
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .base import GenerationResult, LLMBackend, TaskVectorLike
from .prompts import PromptSpec

__all__ = ["HFBackend"]

# Qwen 系列 LoRA 的默认目标模块
_QWEN_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class HFBackend(LLMBackend):
    """基于 HuggingFace ``transformers`` 的本地 LLM 后端。

    Args:
        model_name: 模型名或本地路径。
        local_dir: 本地权重目录；非空时优先于 ``model_name``。
        torch_dtype: ``bfloat16`` / ``float16`` / ``float32``。
        device_map: 传给 ``from_pretrained`` 的设备映射，默认 ``auto``。
        load_in_8bit: 是否 8bit 量化加载（13B 单卡 24GB 场景需要）。
        max_new_tokens: 单次生成的最大新 token 数。
        generation: 生成参数（temperature / top_p / top_k / repetition_penalty / do_sample / seed）。
        lora: LoRA 配置字典（r / alpha / dropout / target_modules / lr / epochs / ...）。
        trust_remote_code: 是否信任远端代码（部分模型需要）。
    """

    name = "transformers"
    supports_finetuning = True
    supports_task_vector = True

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        local_dir: str = "",
        torch_dtype: str = "bfloat16",
        device_map: Any = "auto",
        load_in_8bit: bool = False,
        max_new_tokens: int = 1024,
        generation: Optional[Mapping[str, Any]] = None,
        lora: Optional[Mapping[str, Any]] = None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.model_path = local_dir or model_name
        self.torch_dtype_name = torch_dtype
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.max_new_tokens = max_new_tokens
        self.generation_config = dict(generation or {})
        self.lora_config = dict(lora or {})
        self.trust_remote_code = trust_remote_code

        # 运行时状态（延迟初始化）
        self._model = None
        self._tokenizer = None
        self._peft_model = None
        self._base_lora_state: Optional[Dict[str, Any]] = None
        # 加载模型必须串行：Augmentor 的线程池可能并发进入 generate()，
        # 若无锁保护，多个线程会同时 from_pretrained 同一份 7B/13B 权重（显存爆炸）。
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 加载
    # ------------------------------------------------------------------ #
    def _resolve_dtype(self):
        import torch

        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if self.torch_dtype_name not in mapping:
            raise ValueError(
                f"不支持的 torch_dtype={self.torch_dtype_name!r}；"
                f"可选 {sorted(mapping)}"
            )
        return mapping[self.torch_dtype_name]

    def _ensure_loaded(self) -> None:
        """首次调用时加载模型与分词器（线程安全）。

        双重检查：先在无锁路径快速返回（已加载是最常见的情况），
        未加载时再进锁，进锁后重新判断一次，避免两个线程都执行加载。
        """
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return
            self._load_locked()

    def _load_locked(self) -> None:
        """真正的加载逻辑，必须持有 :attr:`_load_lock` 才能调用。"""
        import torch  # noqa: F401  （确保依赖存在，报错信息更清晰）
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            padding_side="left",  # 因果 LM 批量生成必须左填充
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": self.device_map,
        }
        if self.load_in_8bit:
            # 8bit 时必须用 fp16 计算，且不能同时指定 dtype
            load_kwargs["load_in_8bit"] = True
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = self._resolve_dtype()

        model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        model.eval()

        # 先本地变量构造完成，再一次性发布，避免其他线程看到"半初始化"状态
        self._tokenizer = tokenizer
        self._model = model

    # ------------------------------------------------------------------ #
    # 生成
    # ------------------------------------------------------------------ #
    def _build_inputs(self, prompts: Sequence[PromptSpec]):
        """把 PromptSpec 批量编码成模型输入。"""
        texts = []
        for spec in prompts:
            messages = spec.as_messages()
            if hasattr(self._tokenizer, "apply_chat_template"):
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:  # pragma: no cover - 兼容没有 chat 模板的分词器
                text = spec.as_prompt_text()
            texts.append(text)

        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.generation_config.get("max_input_length", 3072)),
        )
        device = getattr(self._model, "device", None)
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        return encoded, texts

    def generate(
        self,
        prompts: Sequence[PromptSpec],
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> List[GenerationResult]:
        """批量生成增强结果。

        **真正的批处理**：按温度把 Prompt 分成至多 3 组，每组一次
        ``model.generate`` 调用，而不是逐条生成。这样 ``llm.augmentation.batch_size``
        才有意义（否则它只能当线程数用，而本地后端又是串行的）。

        温度与多样性的关系（论文"多样性提升"目标的落点）：

        * ``temperature_jitter = 0``（默认即 0.05）时所有样本同温，
          多样性完全来自采样本身；
        * ``temperature_jitter > 0`` 时按 ``i % 3`` 把样本分到
          ``base - jitter / base / base + jitter`` 三档，让同一批数据的改写风格有差异；
        * **同一档内温度相同**，因此可以安全地合成一个 batch——
          逐条生成与分批生成在同一档内是无差别的。

        可复现性：每个 batch 生成前调用一次 ``torch.manual_seed(seed + group_index)``，
        同一份输入以**相同的 batch_size 与相同的顺序**重复运行会得到完全一致的结果。

        Note:
            温度分档是按"在本次调用内部的下标"决定的（``index % 3``），
            因此改变 ``llm.augmentation.batch_size``（它决定 Augmentor 每次
            提交多少条 Prompt）会改变某条样本落在哪一档、以及它所在组的随机种子，
            生成结果随之变化。这与"逐条生成"的旧行为不同：
            旧实现每条样本单独调用，下标恒为 0，所有样本都落在同一档
            （实际上是 ``base - jitter``），既没有分档多样性，也无法批处理。
            **要复现同一批增强结果，请固定 batch_size、seed 与输入顺序。**
        """
        if not prompts:
            return []

        try:
            self._ensure_loaded()
        except Exception as exc:  # pragma: no cover - 环境相关
            return [
                GenerationResult(uid=spec.meta.get("uid", ""), prompt_hash=spec.prompt_hash,
                                 error=f"模型加载失败：{exc}")
                for spec in prompts
            ]

        import torch

        base_temperature = (
            temperature
            if temperature is not None
            else float(self.generation_config.get("temperature", 0.9))
        )
        jitter = float(self.generation_config.get("temperature_jitter", 0.0))
        top_p = float(self.generation_config.get("top_p", 0.9))
        top_k = int(self.generation_config.get("top_k", 50))
        repetition_penalty = float(self.generation_config.get("repetition_penalty", 1.05))
        do_sample = bool(self.generation_config.get("do_sample", True))
        seed = self.generation_config.get("seed")

        # 按温度分档：jitter=0 时只有一档（全部同温）
        groups: Dict[int, List[int]] = {}
        for index in range(len(prompts)):
            group_index = (index % 3) if jitter else 0
            groups.setdefault(group_index, []).append(index)

        results: List[Optional[GenerationResult]] = [None] * len(prompts)

        with torch.no_grad():
            for group_index in sorted(groups):
                indices = groups[group_index]
                sample_temperature = max(
                    0.01, base_temperature + jitter * (group_index - 1)
                ) if jitter else max(0.01, base_temperature)

                # 只编码本组的 Prompt
                group_specs = [prompts[index] for index in indices]
                encoded, _ = self._build_inputs(group_specs)
                if seed is not None:
                    torch.manual_seed(int(seed) + group_index)

                try:
                    output_ids = self._model.generate(
                        **encoded,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=do_sample,
                        temperature=sample_temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self._tokenizer.pad_token_id,
                        eos_token_id=self._tokenizer.eos_token_id,
                        **kwargs,
                    )
                    prompt_length = encoded["input_ids"].shape[1]
                    for row, original_index in enumerate(indices):
                        completion = output_ids[row][prompt_length:]
                        text = self._tokenizer.decode(completion, skip_special_tokens=True)
                        results[original_index] = GenerationResult(
                            text=text,
                            uid=str(prompts[original_index].meta.get("uid", "")),
                            prompt_hash=prompts[original_index].prompt_hash,
                            meta={
                                "temperature": sample_temperature,
                                "temperature_group": group_index,
                                "backend": self.name,
                                "model_name": self.model_name,
                            },
                        )
                except Exception as exc:
                    # 整组失败：逐条标记错误，但不中断其它组
                    for original_index in indices:
                        results[original_index] = GenerationResult(
                            uid=str(prompts[original_index].meta.get("uid", "")),
                            prompt_hash=prompts[original_index].prompt_hash,
                            error=f"生成失败（温度档 {group_index}）：{exc}",
                        )

        return [
            result if result is not None
            else GenerationResult(
                uid=str(prompts[index].meta.get("uid", "")),
                prompt_hash=prompts[index].prompt_hash,
                error="生成结果缺失",
            )
            for index, result in enumerate(results)
        ]

    # ------------------------------------------------------------------ #
    # LoRA 微调
    # ------------------------------------------------------------------ #
    def _ensure_peft(self):
        """注入 LoRA 适配器（幂等且线程安全），并记录初始化状态以计算任务向量。

        走 :meth:`_ensure_loaded` 的同一把锁：LoRA 注入会改模型结构，
        并发进入会让两个线程各自包一层适配器。
        """
        if self._peft_model is not None:
            return self._peft_model

        with self._load_lock:
            if self._peft_model is not None:
                return self._peft_model

            # 注意：这里调用 _load_locked 而不是 _ensure_loaded——
            # 后者会再次获取同一把非重入锁，直接死锁。
            if self._model is None:
                self._load_locked()
            from peft import LoraConfig, get_peft_model

            config = self.lora_config or {}
            target_modules = config.get("target_modules") or list(_QWEN_TARGET_MODULES)
            lora_config = LoraConfig(
                r=int(config.get("r", 8)),
                lora_alpha=int(config.get("alpha", 16)),
                lora_dropout=float(config.get("dropout", 0.05)),
                bias=str(config.get("bias", "none")),
                task_type="CAUSAL_LM",
                target_modules=list(target_modules),
            )
            peft_model = get_peft_model(self._model, lora_config)
            # 冻结状态快照：TIES-Merging 需要 θ_base（这里即 LoRA 的初始值）
            base_state = {
                name: param.detach().clone()
                for name, param in peft_model.named_parameters()
                if param.requires_grad
            }
            self._peft_model = peft_model
            self._base_lora_state = base_state
            return self._peft_model

    def _lora_parameter_names(self) -> List[str]:
        model = self._ensure_peft()
        return [name for name, param in model.named_parameters() if param.requires_grad]

    def finetune(
        self,
        records: Sequence[Mapping[str, Any]],
        output_dir: str,
        **kwargs: Any,
    ) -> Optional[str]:
        """用增强数据做自举微调（论文 §3.3）。

        Args:
            records: 微调样本；每条须含 ``prompt``（喂给模型的文本）与
                ``completion``（期望输出），或直接给 ``text``。
            output_dir: LoRA 适配器保存目录。

        Returns:
            适配器目录路径。

        Note:
            训练超参全部来自 ``llm.lora`` 配置节，不与 CL 训练共享优化器——
            论文把 LLM 微调描述为"周期性"发生的独立步骤。
        """
        import torch
        from torch.utils.data import DataLoader, Dataset

        if not records:
            raise ValueError("微调样本为空")

        model = self._ensure_peft()
        tokenizer = self._tokenizer
        config = self.lora_config or {}
        max_length = int(config.get("max_seq_length", 1024))
        batch_size = int(config.get("batch_size", 1))
        accumulate = max(1, int(config.get("gradient_accumulation_steps", 8)))
        epochs = int(config.get("epochs", 1))
        learning_rate = float(config.get("learning_rate", 1e-4))
        weight_decay = float(config.get("weight_decay", 0.0))
        warmup_ratio = float(config.get("warmup_ratio", 0.03))

        class _TextDataset(Dataset):
            def __init__(self, items: Sequence[Mapping[str, Any]]):
                self.items = list(items)

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, index: int) -> Dict[str, Any]:
                item = self.items[index]
                if "text" in item:
                    text = str(item["text"])
                else:
                    prompt = str(item.get("prompt", ""))
                    completion = str(item.get("completion", ""))
                    text = f"{prompt}{completion}"
                encoded = tokenizer(
                    text + (tokenizer.eos_token or ""),
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"][0]
                attention_mask = encoded["attention_mask"][0]
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

        def _collate(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
            return {
                key: torch.stack([item[key] for item in batch], dim=0)
                for key in ("input_ids", "attention_mask", "labels")
            }

        loader = DataLoader(
            _TextDataset(records), batch_size=batch_size, shuffle=True, collate_fn=_collate
        )
        device = next(model.parameters()).device
        model.train()

        trainable = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
        total_steps = max(1, (len(loader) // accumulate) * epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=max(0.0, min(0.9, warmup_ratio)),
        )

        step = 0
        for _epoch in range(epochs):
            for batch_index, batch in enumerate(loader):
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / accumulate
                loss.backward()
                if (batch_index + 1) % accumulate == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
        # 收尾：处理最后不足 accumulate 的梯度
        if len(loader) % accumulate != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        model.eval()
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
        return output_dir

    # ------------------------------------------------------------------ #
    # 任务向量
    # ------------------------------------------------------------------ #
    def export_task_vector(self) -> Optional[TaskVectorLike]:
        """导出 ``τ = θ_ft - θ_base``（LoRA 参数粒度）。

        只包含可训练参数（``lora_A`` / ``lora_B``），基座权重不参与，
        因此 13B 模型的任务向量也只有几十 MB。
        """
        if self._peft_model is None or self._base_lora_state is None:
            return None
        vector: Dict[str, Any] = {}
        for name, param in self._peft_model.named_parameters():
            if not param.requires_grad:
                continue
            base = self._base_lora_state.get(name)
            if base is None:
                continue
            vector[name] = (param.detach().float().cpu() - base.float().cpu())
        return vector

    def apply_task_vector(self, task_vector: TaskVectorLike, scaling: float = 1.0) -> None:
        """把合并后的任务向量写回 LoRA 参数：``θ ← θ_base + scaling · τ``。"""
        if self._peft_model is None or self._base_lora_state is None:
            raise RuntimeError("必须先调用 finetune() 建立 LoRA 适配器，再写回任务向量")

        import torch

        with torch.no_grad():
            for name, param in self._peft_model.named_parameters():
                if not param.requires_grad:
                    continue
                base = self._base_lora_state.get(name)
                if base is None:
                    continue
                delta = task_vector.get(name)
                if delta is None:
                    # 合并结果里没有该参数：回到基座值
                    param.copy_(base.to(param.device, dtype=param.dtype))
                    continue
                moved = delta.to(param.device, dtype=torch.float32) * float(scaling)
                param.copy_((base.float() + moved).to(param.dtype))

    def reset_to_base(self) -> None:
        """把 LoRA 参数恢复到微调前的初始值（论文 Algorithm 2 每次都从 θ_0 出发）。"""
        if self._peft_model is None or self._base_lora_state is None:
            return
        for name, param in self._peft_model.named_parameters():
            base = self._base_lora_state.get(name)
            if base is not None:
                param.data.copy_(base.to(param.device, dtype=param.dtype))

    def close(self) -> None:
        """释放显存。"""
        self._peft_model = None
        self._base_lora_state = None
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
