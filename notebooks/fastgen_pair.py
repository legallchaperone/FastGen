import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FastGen × molab 配对 notebook

    这个 notebook 用于在 [molab](https://molab.marimo.io) 上运行
    [FastGen](https://github.com/NVlabs/FastGen) 的 fork，并让 Claude Code 通过
    `marimo-pair` skill（见仓库 `.claude/skills/marimo-pair`）连接到同一个内核配对。

    内容：

    1. 环境信息：Python、PyTorch 版本，CUDA 是否可用，GPU 名称和显存
    2. 用一个小张量在 GPU 上做一次运算（没有 GPU 时只给提示）
    3. 按钮：克隆 fork 并执行 `pip install -e .`
    4. 按钮：下载 CIFAR-10 并启动 DMD2 快速训练

    **打开 notebook 不会自动下载数据、安装或训练**，第 3、4 步都要手动点按钮。
    """)
    return


@app.cell
def _():
    import marimo as mo
    import os
    import platform
    import subprocess
    import sys
    from pathlib import Path
    return Path, mo, os, platform, subprocess, sys


@app.cell
def _(platform, sys):
    try:
        import torch
    except ImportError:
        torch = None

    def _gpu_info():
        info = {
            "Python": sys.version.split()[0],
            "平台": platform.platform(),
            "PyTorch": torch.__version__ if torch is not None else "未安装",
        }
        if torch is None:
            info["CUDA 可用"] = False
            return info
        cuda_ok = torch.cuda.is_available()
        info["CUDA 可用"] = cuda_ok
        info["CUDA 版本 (torch 编译)"] = torch.version.cuda or "无"
        if cuda_ok:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info[f"GPU {i}"] = props.name
                info[f"GPU {i} 显存"] = f"{props.total_memory / 1024**3:.1f} GiB"
        return info

    env_info = _gpu_info()
    for _k, _v in env_info.items():
        print(f"{_k}: {_v}")
    return env_info, torch


@app.cell
def _(env_info, mo, torch):
    if torch is None:
        gpu_check = mo.callout(
            mo.md("没有安装 PyTorch，跳过 GPU 测试。先运行下面的安装步骤。"),
            kind="warn",
        )
    elif not env_info["CUDA 可用"]:
        gpu_check = mo.callout(
            mo.md(
                "**没有检测到 CUDA GPU。** 在 molab 里请切换到 GPU 运行时；"
                "本地没有 NVIDIA GPU 时这是正常的。FastGen 训练需要 GPU。"
            ),
            kind="warn",
        )
    else:
        _a = torch.randn(1024, 1024, device="cuda")
        _b = torch.randn(1024, 1024, device="cuda")
        _c = _a @ _b
        torch.cuda.synchronize()
        gpu_check = mo.callout(
            mo.md(
                f"GPU 运算成功：`(1024×1024) @ (1024×1024)` 在 "
                f"`{_c.device}` 上完成，结果均值 {_c.mean().item():.4f}。"
            ),
            kind="success",
        )
    gpu_check
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 克隆 fork 并安装

    如果 notebook 已经在 FastGen 仓库里运行（能找到 `train.py`），直接使用当前仓库；
    否则把 fork 克隆到 `~/FastGen`。然后执行 `pip install -e .`。
    """)
    return


@app.cell
def _(Path, mo):
    FORK_URL = "https://github.com/legallchaperone/FastGen.git"

    def _find_repo():
        try:
            here = Path(__file__).resolve().parent.parent
        except NameError:
            here = Path.cwd()
        for cand in (here, Path.cwd(), Path.cwd().parent):
            if (cand / "train.py").exists() and (cand / "fastgen").is_dir():
                return cand
        return Path.home() / "FastGen"

    REPO_DIR = _find_repo()
    install_button = mo.ui.run_button(label="克隆 fork 并 pip install -e .")
    mo.vstack([mo.md(f"仓库目录：`{REPO_DIR}`"), install_button])
    return FORK_URL, REPO_DIR, install_button


@app.cell
def _(mo, os, subprocess):
    def run_streaming(cmd, cwd=None):
        """运行命令并把输出实时显示在 cell 下方；返回退出码。"""
        mo.output.append(mo.md(f"`$ {' '.join(cmd)}`"))
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        for line in proc.stdout:
            print(line, end="")
        return proc.wait()
    return (run_streaming,)


@app.cell
def _(FORK_URL, REPO_DIR, install_button, mo, run_streaming, sys):
    mo.stop(not install_button.value, mo.md("点击上面的按钮开始克隆和安装。"))

    if not (REPO_DIR / "train.py").exists():
        _rc = run_streaming(["git", "clone", FORK_URL, str(REPO_DIR)])
        mo.stop(_rc != 0, mo.callout(mo.md("git clone 失败"), kind="danger"))
    _rc = run_streaming([sys.executable, "-m", "pip", "install", "-e", "."], cwd=REPO_DIR)
    mo.callout(
        mo.md("安装完成" if _rc == 0 else f"pip install 失败（退出码 {_rc}）"),
        kind="success" if _rc == 0 else "danger",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 下载数据并训练（手动触发）

    依次运行：

    ```bash
    python scripts/download_data.py --dataset cifar10
    python train.py --config=fastgen/configs/experiments/EDM/config_dmd2_fast.py
    ```

    需要 GPU，并会下载数据集和 EDM 教师权重。
    """)
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="下载 CIFAR-10 并开始训练", kind="danger")
    train_button
    return (train_button,)


@app.cell
def _(REPO_DIR, env_info, mo, run_streaming, sys, train_button):
    mo.stop(not train_button.value, mo.md("未运行。点击按钮后才会下载数据和训练。"))
    mo.stop(
        not (REPO_DIR / "train.py").exists(),
        mo.callout(mo.md(f"`{REPO_DIR}` 下没有 FastGen，先运行安装步骤。"), kind="warn"),
    )
    mo.stop(
        not env_info["CUDA 可用"],
        mo.callout(mo.md("没有可用的 CUDA GPU，不启动训练。"), kind="warn"),
    )

    _rc = run_streaming(
        [sys.executable, "scripts/download_data.py", "--dataset", "cifar10"], cwd=REPO_DIR
    )
    mo.stop(_rc != 0, mo.callout(mo.md(f"数据下载失败（退出码 {_rc}）"), kind="danger"))
    _rc = run_streaming(
        [
            sys.executable,
            "train.py",
            "--config=fastgen/configs/experiments/EDM/config_dmd2_fast.py",
        ],
        cwd=REPO_DIR,
    )
    mo.callout(
        mo.md("训练结束" if _rc == 0 else f"训练失败（退出码 {_rc}）"),
        kind="success" if _rc == 0 else "danger",
    )
    return


if __name__ == "__main__":
    app.run()
