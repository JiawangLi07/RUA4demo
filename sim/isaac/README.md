# 天机右手的 Isaac Sim 抓取

头部 D455 从当前仿真帧定位物体，按几何选择手形和预抓位。接近途中手指同步预形。到位后，腕部 D405 以实测手眼位姿重新曝光，按新深度修正接触区。手指按接触反馈闭合，先试提 2 cm，再抬升并保持 2 s。一次回合只初始化一次刚体。失抓后，手把仍在手里的物体放回桌面，松手、退开、等待静稳，再选择新的手形。

| 场景名 | 选取的接触区 | 持稳时测量的接触 |
| --- | --- | --- |
| `screwdriver` | 从横截面中选取连续的粗手柄，沿柄长分配指位 | 同一物理步有拇指和至少两根其他手指同时承力。独立评分还逐步核对接触点位于手柄两侧。 |
| `bottle`、`tube` | 直立瓶身或筒壁 | 分开记录中央掌体、掌骨、指腹、近节和中节。瓶子的掌面载荷控制会调节拇指根部。 |
| `shuttle` | 横放羽毛球较窄的球托端 | 拇指和对握指的指腹接触。羽毛裙按刚体建模。 |
| `box` | 扁盒的侧边 | 第一种手形未能维持对握时，重新观察并换手形。 |

## 输入和运行

实测环境为 Isaac Sim 5.1、Isaac Lab 0.53.1、NVIDIA CUDA，物理频率 240 Hz，录像 15 fps。使用带有 Isaac Lab、NumPy、SciPy 和 Pillow 的运行时 Python。编码 MP4 需要系统命令 `ffmpeg`。以下命令从 RUA4demo 仓库根目录运行。

场景、机械手 CAD、手形库和输入网格由工位部署提供。仿真使用当前帧的原生 RGB-D 和实例掩膜，将米制 SAM3D 先验配准到观测。相机内参和手眼关系从载入的 USD 场景读取。外部输入目录结构如下。

```text
$RUA_SIM_ASSETS/
  real2sim_0417/source/
    scene.usdc  robot_cad.urdf  scenario.json  initial_pose.json
    hand_motor_config.json  tube_texture.png
  sam3d_glb/
    driver.npz  bottle.npz  tube.npz  box.npz  shuttle.npz  shuttle_side_prior.npz
  hand_postures.json
```

`hand_postures.json` 含 `posture` 和 `pinch_postures` 两项，分别对应初始张手及参考位姿、按 Sharpa 手形名索引的 22 维关节角（弧度）。现有重放包的 `replay/cases/*.json` 已包含这两项。可以直接将 `RUA_SIM_ASSETS` 指向该 `replay` 目录，先将其中的 `real2sim_0417.tgz` 展开到同一目录。内置物体的质量、摩擦、初始摆位、网格 SHA-256 及控制参数见[物体配置](config/objects.json)和[工位配置](config/station.json)。

```bash
export RUA_SIM_ASSETS=/path/to/station-assets
export ISAAC_PYTHON=/path/to/isaac-lab/bin/python
"$ISAAC_PYTHON" -m sim.isaac --case screwdriver --check
"$ISAAC_PYTHON" -m sim.isaac --case screwdriver --headless --device cuda:0 \
  --out runs/isaac-screwdriver-01 --video
python -m sim.isaac.score runs/isaac-screwdriver-01
```

输出目录应使用新的路径，这样能保留此前的录像和原始记录。使用其他物体时可替换 `--case`，自定义工位可改传 `--config /path/to/case.json`。`--check` 能在 CPU 上核对输入路径、网格结构和散列。原生运行和评分以退出码 0 表示通过，以 1 表示抓取未通过。

## 用 36 类手形生成候选姿态

可选模块 `sim.isaac.catalog` 使用工位的 DexBridge v3 手形库。它按米制网格的截面、三轴尺寸和桌面朝向查询 33 类编号手形及 3 类扩展手形。每个匹配模式输出预抓、闭合、收紧三组 `T_world_palm` 与 SDK 顺序的 22 维绝对关节角（弧度）。输出同时保留模式编号、样本来源及尺寸是否超出插值范围。

`--goal palm_wrap` 要求模板包含中央掌面及至少两根手指的近节或中节，按多指包络的覆盖及该形状的历史记录排序。`--goal any` 可查询其他接触目标的通用候选。需将 DexBridge 的 `src` 加入 Python 搜索路径，并备好与索引匹配的数值缓存。以下示例从已有的一帧头部网格开始。

```bash
export PYTHONPATH="$DEXBRIDGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m sim.isaac.catalog --assets "$RUA_SIM_ASSETS" --case tube \
  --index "$DEXBRIDGE_ROOT/configs/skills/sharpa_grasp_templates.json" \
  --head-mesh "$EPISODE/attempt_01/head_mesh.npz" --table-z "$TABLE_Z_METRES" \
  --goal palm_wrap --shortlist 3 --out runs/tube-catalog
```

到达预抓位后，加入 `--wrist-mesh "$EPISODE/attempt_01/wrist_mesh.npz"` 并指定新输出目录以复算。已有网格世界位姿时可直接传 `--world-from-mesh pose.json`，文件内容为 4×4 刚体变换矩阵。自定义物体可用既有格式的 `--config` 提供网格及工位参数。

手头只有 SAM3D 的 GLB 时，可用 Trimesh 直接读取原网格。提供每个模型单位对应多少米，以及缩放后的模型坐标到世界坐标的刚体变换。以下命令仅计算候选抓姿，可以在没有 Isaac Sim 的机器上运行。

```bash
python -m sim.isaac.catalog --index "$DEXBRIDGE_ROOT/configs/skills/sharpa_grasp_templates.json" \
  --mesh bottle.glb --meters-per-unit "$METERS_PER_UNIT" --world-from-mesh pose.json \
  --goal palm_wrap --out runs/bottle-from-glb
```

`candidates.json` 列出整库筛选结果。对直立圆柱侧壁，工具另存 `case.json`，将候选手形及关节种子交给控制器。用 `python -m sim.isaac --assets "$RUA_SIM_ASSETS" --config runs/tube-catalog/case.json --headless --device cuda:0 --out runs/tube-native --video` 验证。控制器在本回合重新观测、校正腕姿、检查净空及 IK。生成配置会启用 `require_palm_hold`，要求在悬空保持的每个物理步都测到中央掌面和至少两根非拇指承力，其中至少一根的近节或中节接触。只有两根非拇指承力时，还须测到拇指接触。失去支撑时控制器重新尝试，离线评分会逐步复核接触、离桌高度和滑移。

## 如何核对一次抓取

- `observer_frames.mp4` 是固定侧视连续原速录像，`frames.mp4` 是同一回合的头部视角。`frames.json` 将画面与仿真时间关联。
- `attempt_01/` 等目录保存各次尝试的头部和腕部 RGB-D、相机位姿及观测网格。`events.json` 和 `attempts.json` 记录到达、曝光、手形、释放和重试。
- `physics.json` 保存逐步的物体位姿、离桌间隙及每个手部刚体对目标的载荷。螺丝刀另存各刚体最强接触点和法向。`sim.isaac.score` 把这些点随物体刚体运动变回观测时的手柄坐标，按柄长、高度和横向半径核对每个触点。截面径向容差为 6 mm，对握要求拇指和至少两根其他手指分布在中心线两侧、间距超过 10 mm。早期录像若缺少截面数据，评分可从同一输入网格及当时记录的腕部配准位姿重建，需设置 `RUA_SIM_ASSETS`。每指各节法向载荷之和达到 0.08 N 计作承力。
- 保持期共 480 个物理步，物体最低抬升须达到 9 cm，离桌至少 5 mm，相对掌体平移不超过 30 mm、转角不超过 0.5 rad。真机抓取的执行入口和 25 cm 目标见[项目首页](../../README.md)。

CPU 回归可运行 `python -m pytest -q sim/isaac/tests`。依赖为 NumPy、SciPy、Pillow 和 pytest。

## 同一工位的仿真验证

下表给出默认物体摆位的单回合实测值。最低抬升取悬空保持期间逐步记录的最小值。

| 物体 | 成功时是第几次尝试 | 最低抬升 | 连续保持 |
| --- | ---: | ---: | ---: |
| 螺丝刀 | 1 | 9.06 cm | 五指均在手柄上，480 / 480 步 |
| 瓶子 | 1 | 10.31 cm | 掌面和手指包络，480 / 480 步 |
| 球筒 | 2 | 10.23 cm | 掌面和手指包络，480 / 480 步 |
| 横放羽毛球 | 1 | 9.76 cm | 2 s |
| 扁盒 | 2 | 9.74 cm | 2 s |

螺丝刀还在两个摆位做了复测。相对默认位置，桌面平移 `(+12, −8) mm`、偏航 `+12°` 时最低抬升 9.01 cm。平移 `(−4, +5) mm`、偏航 `−6°` 时为 9.02 cm。两个回合都由相机重新定位，并在保持的每个物理步记录到五指在手柄上承力。结果对应这三个摆位和同一件 100 g 螺丝刀。

整库查询另做了两次原生复验。按场景假定的 19 cm 尺寸标定物体后，瓶子得到 11 个全掌候选，球筒得到 12 个。将各自的前三种手形交给控制器，两者都以 `1_Large_Diameter` 完成首次抓取，使用的姿态模式分别为 `stout_cylinder-27` 和 `slender_cylinder-82`。瓶子最低抬升 10.04 cm，中央掌面法向载荷中位数 4.35 N。球筒对应为 10.15 cm 和 2.26 N。两个回合都通过了 480 步掌面包络检查。

在 i9-14900KF 上缓存网格特征和模板后，对每个物体改变世界位姿查询 21 次，瓶子和球筒的中位查询时间分别为 64 ms 和 43 ms。计时包含全部手形排序、模式匹配、三阶段掌体位姿和手指关节角计算。SAM3D 重建、相机配准、机械臂求解和物理执行各自独立计时。
