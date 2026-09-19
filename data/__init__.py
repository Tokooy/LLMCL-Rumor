# coding=utf-8
"""数据层：原始数据解析、编码器输入构造、PyTorch Dataset。

依赖方向
--------
``data.processors`` 不依赖 torch，可以在无 GPU 环境的机器上运行预处理；
``data.dataset`` 需要 torch，只在训练阶段被导入。
"""

__all__ = ["processors", "dataset"]
