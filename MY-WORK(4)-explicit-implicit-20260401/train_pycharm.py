"""
PyCharm训练脚本 - 放在项目根目录下（和frnet/、configs/、tools/同级）

使用方法：
    在PyCharm中直接右键 Run 即可，不需要命令行参数。
    修改下面的 CONFIG_FILE 和其他参数来控制训练。
"""

import os
import os.path as osp

# ============================================================
# 在这里修改你的配置
# ============================================================
# CONFIG_FILE = r'/home/user/YHY/MY-WORK(3)/configs/frnet/frnet-semantickitti_seg.py'  # 改成你的config路径
CONFIG_FILE = r'/home/user/YHY/MY-WORK(4)-explicit-implicit-20260401/configs/frnet/frnet-semantickitti_seg.py'  # 改成你的config路径
WORK_DIR = './work_dirs/frnet_explicit'              # 输出目录
RESUME_FROM = None   # 断点续训的checkpoint路径，None表示从头训
AMP = False          # 是否开启混合精度
GPU_ID = 0           # 使用哪张GPU
# ============================================================

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

from mmengine.config import Config
from mmengine.runner import Runner

# 注册自定义模块（关键！不然找不到你新增的模块）
import frnet.models  # noqa: F401


def main():
    cfg = Config.fromfile(CONFIG_FILE)
    cfg.launcher = 'none'
    cfg.work_dir = WORK_DIR

    # 断点续训
    if RESUME_FROM is not None:
        cfg.resume = True
        cfg.load_from = RESUME_FROM

    # 混合精度
    if AMP:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()