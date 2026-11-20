# RUA4demo · 羽毛球真机抓取

本项目在天机双臂平台上使用右侧 Sharpa 灵巧手抓取桌面羽毛球。右腕 D405 和头部 D455 提供 RGB-D，SAM3 分割球头，指腹触觉驱动闭合、补抓和收紧。当前单 cell 的目标是夹住球头、提起约 25 cm 并验证离桌高度。最近的工位试验验证了接近和闭合试探，完整提起仍待真机验证。

## Python cell 怎样执行

单 cell 是一段预先写好的 Python 程序。服务端为本轮创建运行目录，把源码保存为 `grasp_shuttlecock.py`，绑定连接真机的 `robo` 对象，再执行这段程序。动作之间由关节回读、视觉结果和触觉判定决定下一步。读代码可从 [cell 定义和真机执行入口](exploration/live_cell.py) 开始。

```python
def run(robo):
    robo.capture_tactile_baseline()
    robo.locate_target()
    robo.prepare()
    robo.approach()
    if (robo.close_hand() or robo.retry_grasp()) and robo.tighten_grasp():
        robo.lift()

result = run(robo)
```

| 调用 | 真机上的行为 |
| --- | --- |
| `capture_tactile_baseline()` | 在任何抓取动作前采集拇指、食指各 8 帧 F6 数据。每根手指以力向量的中位数作为基线，整轮补抓共用。 |
| `locate_target()` | 优先使用与关节状态对齐的腕部 RGB-D，必要时回退到主相机。分割羽毛球和球头，从有效深度拟合球头球面，转换到机器人安装坐标系。 |
| `prepare()` | 在平行于桌面的球头截面上选取一对相对接触点，联合求解腕姿和两指手形。规划预抓时每侧约留 1 mm，闭合终点每侧向内 4 mm。逐点求逆运动学并检查几何路径，候选失败后换腕姿或夹持方向。 |
| `approach()` | 先驱动手指到预抓手形并记录实际开口。预成形中卡滞的关节保持实测角度，其余关节继续，必要时按实测手形复核路径。腕部先抬到比起点和目标都高至少 6 cm 的位置，在高处横移转腕，再以抓取姿态下降。 |
| `close_hand()` | 双指沿规划点闭合。每指连续 3 个新鲜触觉帧相对基线的力向量变化达到固定阈值 `0.00001`，即判为该指轻触。先达标的一指保持位置，另一指继续。 |
| `retry_grasp()` | 首次闭合未确认双指接触时执行。根据无接触、单指稳定、接触不稳定或关节卡滞选择原位试探、卸力或重新定位。局部候选和重新定位机会耗尽会结束本轮。 |
| `tighten_grasp()` | 用关节回读估算两块指腹的位置，向其中点小步收紧，每指目标额外行程最多 4 mm。每根手指的收紧阈值取 `0.02` 和该指基线噪声 5 倍中的较大值。两指均持续达标才进入提起。上述数值均为 F6 原生单位。 |
| `lift()` | 沿桌面法向提起，运动中持续检查夹持。随后用新的视觉观测核对羽毛球离桌高度是否为 20–30 cm。正常完成后，外层运行器负责放回和释放。 |

抓取点和指腹轨迹在 [cork_pinch.py](exploration/cork_pinch.py)，腕部插值在 [cartesian.py](exploration/cartesian.py)，手指驱动、接触判定和补抓分别在 [live_pinch.py](exploration/live_pinch.py)、[pinch_contact.py](exploration/pinch_contact.py) 和 [pinch_retry.py](exploration/pinch_retry.py)。

## 在现有机器人部署上运行

需要已配置的机器人部署目录，默认是仓库旁的 `../real_experiment/deployment`，也可设置 `DEXHUMANID_RUA_DEPLOYMENT`。运行器使用其中的 Python 环境、天机和 Sharpa SDK、URDF 及运动学库。移植到其他工位时需匹配相机序列号、机器人配置、外参和 `config/station_geometry.xml` 中的 mesh 文件路径。[启动脚本](run.sh) · [右手工位配置](config/right_shuttlecock_station.json)

首次准备源码依赖时，在仓库根目录执行：

```bash
git clone https://github.com/DAGroup-PKU/PyRUA-Lean.git upstream
git -C upstream checkout e754cf43c822b91f4728777a78071fa4e012c371
git -C upstream apply ../patches/sam3-loopback.patch
git clone https://github.com/DAGroup-PKU/PyRUA-Lean.git upstream-rsijudge
git -C upstream-rsijudge checkout 62eb1ce1a9878512356a41631fba6c340f0a8db6
```

在一个终端启动 SAM3。脚本支持通过 `PYRUA_SAM3_PYTHON` 和 `SAM3_CHECKPOINT_PATH` 指定推理环境和已下载的权重。[服务脚本](perception/start_sam3.sh)

```bash
HF_HUB_OFFLINE=1 ./perception/start_sam3.sh
```

在另一终端检查设备占用和定位服务，再启动一次真机试抓。cell 从当前实测位姿规划，工位归位是单独的操作。

```bash
./run.sh check
curl --noproxy '*' -fsS http://127.0.0.1:8790/health
OPENBLAS_NUM_THREADS=1 ./run.sh explore --real --single-cell \
  --target shuttlecock --arm right --profile config/right_shuttlecock_station.json
```

## 查看本轮结果

启动时打印的 `run` 字段指向本机运行目录。先看 `result.json` 的 `phase`、`reason`、`success` 和 `lift`。定位或规划失败可能在首次运动前发生。排查时可按阶段查看以下文件：

- `tactile_baseline.json` 保存动作前采集的基线帧。
- `locate/` 保存相机观测，`grasp_plans` 在结果中记录抓取点、腕姿、预抓手形及开口回读。
- `motion/arm_trajectory.jsonl` 保存臂轨迹目标和实际关节位置。
- `pinch_commands.jsonl` 保存闭合、卸力和补抓指令，以及相应的关节回读和触觉值。

核心逻辑的离线回归用例见 [tests](tests)，其中包括预抓手形、三段接近、单指保持、接触不稳定时补抓、收紧和提起判定。

## 可重复执行的工位操作

在机器人主机的仓库根目录执行。以下步骤独立于抓取 cell。

1. 运行 `./run.sh observe --live --calibration config/d455_mount_from_color.json`。输出中的 `run` 是这次观测目录，打开其中的 `head.png` 查看工位。该模式同时保存归位需要的关节回读。只看画面时可以加 `--camera-only`。
2. 运行 `./run.sh reset-shuttle <本次观测目录>` 离线检查归位路径。在命令末尾加 `--execute` 执行真机归位。脚本从工位配置读取右臂起始角，从本机的 `.local/station/reset_right_hand.json` 读取右手目标手形（22 个弧度值的 JSON 数组，也可用 `--hand-reference` 指定）。首次部署可从已验证起始姿态观测的 `observation.json` 中提取 `joints.hand_rad.right`。脚本随后核对当前关节状态，归位后记录角度误差及手的去使能回读。已在起始位的右臂会跳过臂运动。
3. 运行 `./perception/ensure_sam3.sh` 检查本机 SAM3。健康时直接返回，掉线时后台启动服务并等待健康响应。默认端口为 8790，启动日志在 `.local/sam3/8790.log`。此命令可在每次试抓前重复调用。

## Fast-SAM3D 单图三维重建

推理代码位于 [perception/Fast-SAM3D](perception/Fast-SAM3D)，基于 [Fast-SAM3D 上游版本 c76188f](https://github.com/wlfeng0509/Fast-SAM3D/tree/c76188fda755b761fe9255eabe87aaad39df829a)。输入为一张 RGB 图和一个同分辨率的物体掩膜。启动脚本默认启用 Fast-SAM3D 加速，使用 25 个采样步，关闭蒸馏。

### 环境和权重

使用已有的 Fast-SAM3D 环境，或按[上游安装说明](perception/Fast-SAM3D/README.md#installation)准备 CUDA 和 Python 依赖。本机验证环境为 RTX 4090、Python 3.11、PyTorch 2.5.1+cu121。请保留仓库中的加速配置 YAML，将 SAM 3D Objects 的 `*.ckpt` 放在 `perception/Fast-SAM3D/checkpoints/hf/`，MoGe 权重放在 `checkpoints/moge-vitl/model.pt`，DINOv2 源码放在 `checkpoints/dinov2/`（本机验证提交 `7764ea0f`）。[官方权重](https://huggingface.co/facebook/sam-3d-objects) · [ModelScope 权重](https://modelscope.cn/models/facebook/sam-3d-objects) · [SHA-256 校验清单](perception/Fast-SAM3D/checkpoints/verified-weights.json)

在已有部署的工位上，可从 RUA4demo 根目录复用环境和权重。下面的链接命令适用于首次配置的检出目录。

```bash
cd perception/Fast-SAM3D
station_model_dir="$HOME/lianjh/Fast-SAM3D"
export FAST_SAM3D_PYTHON="$station_model_dir/.venv/bin/python"
ln -s "$station_model_dir"/checkpoints/hf/*.ckpt checkpoints/hf/
ln -s "$station_model_dir/checkpoints/moge-vitl" checkpoints/
ln -s "$station_model_dir/checkpoints/dinov2" checkpoints/
```

### 体素模式和蒸馏开关

在 `perception/Fast-SAM3D` 目录运行以下命令即可使用随附样例。处理自己的图片时，追加 `--image_path /绝对路径/image.png --mask_index 0`，并将目标掩膜保存到同目录的 `0.png`。灰度掩膜中非零值为前景，RGBA 掩膜使用 alpha 通道。分别运行两种模式时使用不同输出目录。

```bash
# 25 步体素，关闭蒸馏
./infer.sh --voxel-only --steps 25 --output_dir Look/voxel25-off

# 25 步体素，开启第一阶段蒸馏推理
./infer.sh --voxel-only --distill-stage1 --steps 25 --output_dir Look/voxel25-on
```

| 参数 | 行为 |
| --- | --- |
| `--voxel-only` | 第一阶段结束即返回，导出 `voxels-<mask_index>.npz`。省略此参数时继续第二阶段，导出 GLB 网格和 PLY 点云。 |
| `--distill-stage1` | 将 `use_stage1_distillation` 设为 `True`，使用现有权重的第一阶段 shortcut 推理。省略时关闭。 |
| `--steps N` | 指定采样步数。体素模式只运行第一阶段。 |

NPZ 中的 `native_coords` 是自适应降采样前的物体局部占用网格索引，形状为 `(N, 3)`，每个轴取值 0–63。比较蒸馏前后的体素 IoU 时应使用此字段。`coords` 保存降采样后的索引，`voxel` 保存模型返回的归一化坐标，`downsample_factor` 保存降采样因子。转换到机器人基座系还需要度量尺度、物体位姿和相机外参。

主视角 14 件物体的工位实测中，模型常驻、同帧深度共享、均使用 25 步时，整桌体素推理关闭蒸馏为 **21.17 秒**，开启为 **13.13 秒**。两项均排除模型加载、分割和写盘。分别执行上述 shell 命令会启动新进程并重新加载模型；同一帧多掩膜复用模型和深度的调用见[测速脚本](perception/Fast-SAM3D/notebook/benchmark_voxel_frame.py)。
