# coding=utf-8
"""单元测试公开接口。

测试分两层：

* **纯 CPU 逻辑测试**（不需要 torch/transformers）：解析、校验、配置、数据模型、
  式(7)(8)(9) 的标量运算等；
* **依赖 torch 的测试**：张量算子（Algorithm 1 的 Trim/Elect Sign/Disjoint Merge、
  InfoNCE 损失、投影头）。

后者统一用 ``pytest.importorskip("torch")``，因此在只装了 CPU 依赖的机器上
也能跑完整测试集（相关用例会显示为 skipped，而不是 failed）。
所有测试都**不加载模型权重、不访问网络**。
"""

__all__ = []
