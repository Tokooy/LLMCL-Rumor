# coding=utf-8
"""随机种子与确定性。

复现类项目必须锁死随机性，否则 Table 3–8 的数字无法对齐。
"""

from __future__ import annotations

import os
import random
from typing import Optional

__all__ = ["set_seed"]


def set_seed(seed: int, deterministic: bool = False, cuda_devices: Optional[str] = None):
    """统一设置 random / numpy / torch 的随机种子。

    Args:
        seed: 随机种子。
        deterministic: 是否强制 cuDNN 确定性算法。开启后速度会下降，
            但跨机器复现更稳；默认关闭以贴近论文的实验配置。
        cuda_devices: 形如 ``"0 1"`` 的可见 GPU 列表，用于写 ``CUDA_VISIBLE_DEVICES``。
            必须在第一次创建 CUDA 上下文之前调用才生效。

    Note:
        torch 是延迟导入的：``src.utils`` 被设计为不强制安装 torch 也能导入，
        这样纯数据预处理脚本可以在无 GPU 环境下运行。
    """
    if cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices).replace(",", " ")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - 数据脚本以外都会装 numpy
        np = None
    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover - 无 torch 时静默跳过
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 允许在需要时通过环境变量打开更严格的确定性（可能报错，故不默认开启）
        if os.environ.get("LLMCL_STRICT_DETERMINISM"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
