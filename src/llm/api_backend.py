# coding=utf-8
"""OpenAI 兼容接口后端（vLLM / Ollama / DashScope / OpenAI 均可）。

用途：在没有本地 GPU 或需要快速跑通流水线时替代本地权重。
代价：**不支持梯度微调**，因此论文 Algorithm 2 的"微调 + TIES 合并"阶段不可用。
上层（:class:`src.training.joint_trainer.JointAlignmentTrainer`）会检测
``supports_finetuning=False`` 并自动跳过微调，只做数据增强
——这正好覆盖论文 Proposed-1/2/3（M=0）的实验设置。

请求实现细节
------------
* 用标准库 ``urllib`` 而不是 ``requests``：少一个依赖，且在受限环境里更稳；
* 只支持 ``/v1/chat/completions``；若服务端不支持 ``response_format``，
  由 Prompt 约束 + :mod:`src.llm.parser` 的括号配对抽取兜底；
* 并发由 :class:`src.llm.augmentor.Augmentor` 统一控制，这里保持线程安全
  （每次调用独立连接，无共享可变状态）。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .base import GenerationResult, LLMBackend
from .prompts import PromptSpec

__all__ = ["APIBackend"]


class APIBackend(LLMBackend):
    """OpenAI 兼容的 chat completions 后端。

    Args:
        base_url: 例如 ``https://api.openai.com/v1`` 或本地 vLLM 的 ``http://localhost:8000/v1``。
        api_key_env: 存放 API key 的环境变量名；未设置时按无鉴权服务处理。
        model: 服务端模型名。
        timeout: 单次请求超时秒数。
        max_retries: 单条请求的最大重试次数（指数退避）。
        extra_body: 附加请求体字段（例如 ``{"chat_template_kwargs": {...}}``）。
    """

    name = "api"
    supports_finetuning = False
    supports_task_vector = False

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "qwen2.5-7b-instruct",
        timeout: float = 120.0,
        max_retries: int = 3,
        generation: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.model_name = model
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.generation_config = dict(generation or {})
        self.extra_body = dict(extra_body or {})

    # ------------------------------------------------------------------ #
    def _api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _post(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """发一次请求，返回解析后的响应体。"""
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = self._api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(
            self._endpoint(), data=data, headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)

    def _generate_one(self, spec: PromptSpec, temperature: float) -> GenerationResult:
        """带重试的单条生成。"""
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": spec.as_messages(),
            "temperature": temperature,
            "top_p": float(self.generation_config.get("top_p", 0.9)),
            "max_tokens": int(self.generation_config.get("max_new_tokens", 1024)),
        }
        # 要求服务端返回 JSON（部分服务端不支持，失败后由解析器兜底）
        if self.generation_config.get("response_format_json", False):
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.extra_body)

        last_error: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                body = self._post(payload)
                choices = body.get("choices") or []
                if not choices:
                    last_error = f"响应中没有 choices：{str(body)[:200]}"
                else:
                    message = choices[0].get("message") or {}
                    text = message.get("content") or ""
                    if not text.strip():
                        last_error = "响应内容为空"
                    else:
                        return GenerationResult(
                            text=text,
                            uid=str(spec.meta.get("uid", "")),
                            prompt_hash=spec.prompt_hash,
                            meta={
                                "temperature": temperature,
                                "backend": self.name,
                                "model_name": self.model_name,
                                "usage": body.get("usage"),
                            },
                        )
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:  # pragma: no cover
                    pass
                last_error = f"HTTP {exc.code}：{detail or exc.reason}"
                if exc.code in (400, 401, 403, 404):
                    break  # 参数或鉴权错误，重试无意义
            except Exception as exc:
                last_error = f"{type(exc).__name__}：{exc}"

            if attempt < self.max_retries - 1:
                time.sleep(min(2 ** attempt, 8))

        return GenerationResult(
            uid=str(spec.meta.get("uid", "")),
            prompt_hash=spec.prompt_hash,
            error=last_error or "未知错误",
        )

    def generate(
        self,
        prompts: Sequence[PromptSpec],
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> List[GenerationResult]:
        """逐条顺序生成。

        Note:
            并发在 :class:`src.llm.augmentor.Augmentor` 层通过线程池实现；
            本方法保持顺序执行，是为了让"重试 + 退避"的行为可预期。
        """
        base_temperature = (
            temperature
            if temperature is not None
            else float(self.generation_config.get("temperature", 0.9))
        )
        jitter = float(self.generation_config.get("temperature_jitter", 0.0))

        if not self._api_key() and self.generation_config.get("require_api_key", False):
            return [
                GenerationResult(
                    uid=str(spec.meta.get("uid", "")),
                    prompt_hash=spec.prompt_hash,
                    error=f"环境变量 {self.api_key_env} 未设置",
                )
                for spec in prompts
            ]

        results: List[GenerationResult] = []
        for index, spec in enumerate(prompts):
            sample_temperature = max(0.01, base_temperature + jitter * ((index % 3) - 1))
            results.append(self._generate_one(spec, sample_temperature))
        return results

    def describe(self) -> Dict[str, Any]:
        payload = super().describe()
        payload["base_url"] = self.base_url
        payload["api_key_env"] = self.api_key_env
        return payload
