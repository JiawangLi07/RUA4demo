# 天机右手的 Isaac Sim 抓取

随机桌面入口 `tabletop_native` 根据 D455 和 D405 的 RGB-D、米制网格与机器人 CAD 计算预抓掌位、接触掌位和 22 关节闭合轨迹。试提 2 cm 后用相机检查目标，再将目标抬离桌面 5 cm。手型也可以在 CPU 上独立求解，命令见下方「直接在 CPU 上算掌位和指角」。

## 单件接触反馈对照

早期单件入口 `head_wrist_trial` 使用仿真接触反馈闭合并重试。以下单件结果和视频使用这一入口。

| 场景名 | 选取的接触区 | 持稳时测量的接触 |
| --- | --- | --- |
| `screwdriver` | 从横截面中选取连续的粗手柄，沿柄长分配指位 | 同一物理步有拇指和至少两根其他手指同时承力。独立评分还逐步核对接触点位于手柄两侧。 |
| `bottle`、`tube` | 直立瓶身或筒壁 | 分开记录中央掌体、掌骨、指腹、近节和中节。瓶子的掌面载荷控制会调节拇指根部。 |
| `shuttle` | 横放羽毛球较窄的球托端 | 拇指和对握指的指腹接触。羽毛裙按刚体建模。 |
| `box` | 扁盒的侧边 | 第一种手形未能维持对握时，重新观察并换手形。 |

## 输入和运行

实测环境为 Isaac Sim 5.1、Isaac Lab 0.53.1、NVIDIA CUDA，物理频率 240 Hz，录像 15 fps。使用带有 Isaac Lab、NumPy、SciPy 和 Pillow 的运行时 Python。编码 MP4 需要系统命令 `ffmpeg`。以下命令从 RUA4demo 仓库根目录运行。

完整仿真使用部署工位的场景、机械手 CAD、启动手形和物体刚体网格。仿真使用当前帧的原生 RGB-D 和实例掩膜，将米制 SAM3D 先验配准到观测。相机内参和手眼关系从载入的 USD 场景读取。外部输入目录结构如下。

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

`sim.isaac.catalog` 默认使用仓库自带的 DexBridge v3 手形库。它按米制网格的截面、三轴尺寸和桌面朝向查询 33 类编号手形及 3 类扩展手形。每个匹配模式输出预抓、闭合、收紧三组 `T_world_palm` 与 SDK 顺序的 22 维绝对关节角（弧度）。输出同时保留模式编号、样本来源及尺寸是否超出插值范围。

`--goal palm_wrap` 要求模板包含中央掌面及至少两根手指的近节或中节，按多指包络的覆盖及该形状的历史记录排序。`--goal any` 可查询其他接触目标的通用候选。横放羽毛球可直接使用 `--prior shuttle` 在 CPU 上复算。仓库内附两帧样例、索引和全部配套缓存。[羽毛球示例与文件清单](../../DexBridge/README.md)

下面从工位回合的一帧头部网格生成球筒候选和可执行的场景配置。

```bash
python -m sim.isaac.catalog --assets "$RUA_SIM_ASSETS" --case tube \
  --head-mesh "$EPISODE/attempt_01/head_mesh.npz" --table-z "$TABLE_Z_METRES" \
  --goal palm_wrap --shortlist 3 --out runs/tube-catalog
```

到达预抓位后，加入 `--wrist-mesh "$EPISODE/attempt_01/wrist_mesh.npz"` 并指定新输出目录以复算。已有网格世界位姿时可直接传 `--world-from-mesh pose.json`，文件内容为 4×4 刚体变换矩阵。自定义物体可用既有格式的 `--config` 提供网格及工位参数。

手头只有 SAM3D 的 GLB 时，可用 Trimesh 直接读取原网格。提供每个模型单位对应多少米，以及缩放后的模型坐标到世界坐标的刚体变换。以下命令仅计算候选抓姿，可以在没有 Isaac Sim 的机器上运行。

```bash
python -m sim.isaac.catalog \
  --mesh bottle.glb --meters-per-unit "$METERS_PER_UNIT" --world-from-mesh pose.json \
  --goal palm_wrap --out runs/bottle-from-glb
```

`candidates.json` 列出整库筛选结果。相同手形可以有多个已记录的朝向和尺寸。库会筛除与桌面支撑方向不相容的模式，在相容模式里按尺寸插值关节角和掌体位姿。尺寸超出该模式覆盖范围时，取尺寸最近的实测样本并标记 `size_or_bbox_clamped`。将物体坐标系下的候选变换到当前观测所在的世界坐标系，得到 `T_world_palm = T_world_mesh @ T_mesh_palm`。

用工位场景和候选生成的 `case.json` 可直接交给 Isaac，例如 `python -m sim.isaac --assets "$RUA_SIM_ASSETS" --config runs/tube-catalog/case.json --headless --device cuda:0 --out runs/tube-native --video`。原生控制器按接触任务使用候选。

- 直立圆柱的全掌模式取候选手形和关节种子，在当前筒壁上重算掌位。生成配置启用 `require_palm_hold`，要求保持期每步都有中央掌面和至少两根非拇指承力，其中至少一根的近节或中节接触。只有两根非拇指承力时，还须测到拇指。
- 通用对握模式用 `--goal any`。配置保留候选抓位相对物体的变换、预形角、收紧角和模式号。每次头部或腕部曝光后，重新把变换映射到实测物体上，按局部截面宽度和指腹位置拟合候选，并检查路径净空及机械臂 IK。手柄模式沿粗柄分配多根手指，再按实测手柄轴向拟合掌位。两种模式都在物理回合中核对承力。记录的三阶段姿态是规划种子，实际执行目标还取决于本次观测和反馈。
- 直接运行内置 `--case` 时用的是该场景配置的手形组。要复现整库筛选的结果，应使用上一步输出的 `--config`。

## 失抓后的就近重试

若试提时物体已离桌，先带着物体下降到支撑面，再松手。控制器用新头部深度更新物体位置，沿刚才的接近方向反向搜索。候选从当前张手位置起，最远 4.8 cm，逐个检查整手的 1 mm 净空、IK 和腕部相机能看到的表面。已经有净空时可以停在张手处。物体连续 0.15 s 静稳并与手无接触后，下一次尝试从当前实际手位选择新的手形，可以就地旋转和预形，再用 D405 重新曝光。短路径不足以清空手指时，先退到上一观测位。该处仍不能让物体自由静稳时才回到本回合的起始手位。每回合都保留失败过程和物体被推移后的位置。

同一摆位在 R6000 上做了新旧控制器对照。位移是从第一次失抓到物体自由静稳期间，实测掌心相对失抓点的最大直线位移，包含放回动作。保持高度取第二次抓取最后 480 个物理步的最小值。

| 物体 | 失抓 → 换用手形 | 旧版最大位移 | 新版最大位移 | 再观测时距起始手位 | 保持高度 |
| --- | --- | ---: | ---: | ---: | ---: |
| 球筒 | 小直径环握 → 全掌抓 | 14.29 cm | 3.26 cm | 11.68 cm | 10.27 cm |
| 扁盒 | 平行伸指 → 大直径环握 | 22.62 cm | 8.33 cm | 17.88 cm | 9.68 cm |

两组对照都沿用同一组预设候选，单独比较失败恢复的变化。若把运动一直计到第二次 D405 曝光，球筒的最大位移是 6.82 cm，扁盒是 8.35 cm。扁盒在张手静稳后移动 7.78 cm，掌体转向约 99°，给下一种手形留出空间。球筒在保持期的 480 步都有掌面和多指包络承力。摄像机可见性及掌位跟踪轨迹见 `events.json` 和 `tracking.json`，`python -m sim.isaac.score <回合目录>` 会输出实测位移、累计路径和最终距起始手位的距离。

## 如何核对一次抓取

- `observer_frames.mp4` 是固定侧视连续原速录像，`frames.mp4` 是同一回合的头部视角。`frames.json` 将画面与仿真时间关联。
- `attempt_01/` 等目录保存各次尝试的头部和腕部 RGB-D、相机位姿及观测网格。`events.json` 和 `attempts.json` 记录到达、曝光、手形、释放和重试。
- `physics.json` 保存逐步的物体位姿、离桌间隙及每个手部刚体对目标的载荷。螺丝刀另存各刚体最强接触点和法向。`sim.isaac.score` 把这些点随物体刚体运动变回观测时的手柄坐标，按柄长、高度和横向半径核对每个触点。截面径向容差为 6 mm，对握要求拇指和至少两根其他手指分布在中心线两侧、间距超过 10 mm。早期录像若缺少截面数据，评分可从同一输入网格及当时记录的腕部配准位姿重建，需设置 `RUA_SIM_ASSETS`。每指各节法向载荷之和达到 0.08 N 计作承力。
- 保持期共 480 个物理步，物体最低抬升须达到 9 cm，离桌至少 5 mm，相对掌体平移不超过 30 mm、转角不超过 0.5 rad。真机抓取的执行入口和 25 cm 目标见[项目首页](../../README.md)。

CPU 回归可运行 `python -m pytest -q sim/isaac/tests`。依赖为 NumPy、SciPy、Pillow 和 pytest。

## 单件接触反馈对照的仿真验证

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

横放羽毛球也用 `--goal any` 从 36 类中筛出了 22 个相容手形。在同一段原生回合中，指尖捏持和三指手形先后失败，第三次选中 `27_Quadpod` 模式 `unknown-77`。保持期最低抬升 9.68 cm，滑移峰值 3.55 mm。实际承力的是拇指、食指和中指，三指均持续 480 步。对扁盒运行相同的四候选筛选时，没有一个候选通过接触区和路径检查。

在 i9-14900KF 上缓存网格特征和模板后，对每个物体改变世界位姿查询 21 次，瓶子和球筒的中位查询时间分别为 64 ms 和 43 ms。计时包含全部手形排序、模式匹配、三阶段掌体位姿和手指关节角计算。SAM3D 重建、相机配准、机械臂求解和物理执行各自独立计时。

## 随机桌面的双目观测与连续抓取

同一张桌上摆放 14 个独立刚体。每个随机种子重新抽取平面位置和朝向，初始凸包之间至少留 12 mm。羽毛球横放。双臂从后撤的竖掌姿态开始，掌根在工位坐标系的 `(0.24, ±0.31, 0.95)` 米，指尖向上，左右掌相对。第一帧头部相机在这个姿态下曝光。当前由右手执行，左手保持起姿。

物体目录中的每一项只提供一份米制网格。控制器用头部 D455 的深度估计桌面、分割前景，结合尺寸与外观为目标配准。D405 接近后重新采样表面。前往腕部观察位与最终预抓位时，目标网格和邻物点云均参与避碰，沿插值后的关节路径检查手部碰撞几何与臂链包围盒。到达预抓位后才允许手指向目标表面施加闭合量。对侧面未观测到的区域继续使用配准网格，每个接触点会记录来源。当前配准支持给定静置朝向附近的俯仰与横滚、任意桌面平移与偏航。

两次观测各自查询同一份 36 类手形库。腕部阶段按截面选择可接近的两侧或可包络的侧壁。普通物体的每个候选先选定物体表面的目标点与法向，再固定这些目标，同时优化六自由度掌位与 22 个手指关节。候选之间比较对握方向、假设摩擦锥对重力的支持、手部离桌空间、邻物和臂的可达性。高侧壁按有限侧壁拟合，在满足臂可达的候选中加入掌面、近节和中节接触，带局部粗柄的细长物体沿等长截面分配拇指和多根对握指。手柄候选同时比较指腹拟合、重力支撑与沿柄方向的力臂。接近手柄时，将每个指腹从张开位置到其目标截面的运动按同一进度插值。在每个掌位用指腹雅可比矩阵求关节角，让拇指和对侧手指同步就位。质心取均匀密度凸包的体积加权估计，排序也考虑包围盒中心。闭合阶段在固定掌位下，沿接触点法向分 18 次增加几何预紧，用手部雅可比矩阵求 22 个关节的连续解。预紧距离取决于两侧间距或局部手柄直径。离桌时按下侧表面再算持物指角。头部和腕部图像测量目标是否跟手以及底部的运动。两厘米试提后，视觉高度估计可触发小幅补偿，再计时保持。失败会在目标附近张手、局部退开并重新观察。

### 直接在 CPU 上算掌位和指角

`tabletop_synthesize` 和原生回合使用同一套接触目标分配、关节求解和机械臂可达性检查。输入是一份米制网格、已标定的手 CAD、相机估计的物体位姿和桌面高度。也可以提供分割后的腕部深度点云进行配准。执行端根据已发出的轨迹计算名义腕相机外参，仿真渲染时相机随实际腕部安装位置运动。手型计算和手指闭合均独立于关节编码器、触觉和物体真值。

```bash
export RUA_SIM_ASSETS=/path/to/replay
OPENBLAS_NUM_THREADS=1 python -m sim.isaac.tabletop_synthesize \
  --registry "$RUA_SIM_ASSETS/anon4.json" --object part_fcbc3201 \
  --table-z 0.7165 --wrist-points object_in_station_metres.npy --out grasp.json
```

如已有网格在工位中的位姿，可改用 `--mesh-pose pose.json`。文件为从米制网格局部系到工位坐标系的 4×4 齐次矩阵。`--obstacles other_points.npy` 可提供邻物点。输出包含预抓与接触掌位、22 关节的名称和弧度值、开手至贴合的两组指角、从贴合到预紧的 19 个路点、试提时的持物角度，以及各接触区的目标点和法向。带粗柄的物体还输出同步接近阶段的 19 个掌位和 19 组关节角（`palm_approach_path`、`q_approach_path`）。`q_touch` 保存拟合的参考手型，接近末点衔接闭合轨迹首点。无可达候选时命令以状态 2 返回。此计算使用 NumPy、SciPy 和读取工位 CAD 的 OpenUSD。抓取的实际抬升高度需要物理执行验证。

### 用一个新网格运行

重放包提供 `shared_postures.json.gz` 和工位资产。仓库自带 DexBridge 选择器与训练目录。设置 `export DEXBRIDGE_ROOT="$PWD/DexBridge"` 后，可以用 `PYTHONPATH="$DEXBRIDGE_ROOT/src:$PWD" python -m sim.isaac.tabletop_catalog --index "$DEXBRIDGE_ROOT/configs/skills/sharpa_grasp_templates.json" --out shared_postures.json.gz` 生成这份共享库。运行时只需导出的文件。

```bash
export RUA_SIM_ASSETS="/path/to/new-table"
mkdir -p "$RUA_SIM_ASSETS"
tar -xzf /path/to/replay/real2sim_0417.tgz -C "$RUA_SIM_ASSETS"
python -m sim.isaac.tabletop_registry --mesh object.glb --meters-per-unit 1 --up-axis z \
  --catalog /path/to/replay/shared_postures.json.gz --out "$RUA_SIM_ASSETS"
python -m sim.isaac.tabletop_layout --base sim/isaac/config/station.json \
  --objects "$RUA_SIM_ASSETS/objects.json" --seed 41 --out table41.json
"$ISAAC_PYTHON" -m sim.isaac.tabletop_native --scene table41.json \
  --library "$RUA_SIM_ASSETS/registry.json" --out runs/table41 --headless --device cuda:0
OPENBLAS_NUM_THREADS=1 python -m sim.isaac.tabletop_score --scene table41.json --run runs/table41
python -m sim.isaac.tabletop_video --scene table41.json --run runs/table41 --speed 2 --out table41-2x.mp4
```

可重复传入 `--mesh` 构建多物体场景。GLB、OBJ 和 STL 的导入使用 Trimesh 4，需传入每模型单位对应的米数。示例中的 1 适用于原单位已经是米的网格。`--up-axis` 指定静置时向上的轴。可直接使用已标定的多凸包 NPZ。普通网格导入器为每个有体积的连通块生成凸包，凹槽、把手孔或弯曲物体可用预分解 NPZ 保留物理细节。仿真质量默认为每件 80 g，静动摩擦均为 0.6；这些是场景假设。只含腕部连接件的实物尺寸未知。

### 重放和验收

在原始散摆回放包上，将 `RUA_SIM_ASSETS` 指向解包后的目录，并向原生入口传入 `scene41.json` 或 `scene73.json` 以及 `--library anon14.json`。四个独立构造的网格使用 `scene317.json` 或 `scene318.json` 以及 `anon4.json`。单件螺丝刀使用 `scene1001.json --library anon14.json --target part_763ed53f`。清单中的匿名标识关联几何。场景中的 `evaluation_aliases` 仅供离线评分与视频字幕使用。历史单件参考清单 `library.json` 另存以复现早期对照。原生入口输出目录应是新目录。视频工具可读取回合内双路 PNG，或对应的 `head.mp4`、`observer.mp4`。

`observations/` 保存决策时的 RGB、深度、相机标定与已发出的关节指令。每帧 `joint_source=command`。`events.jsonl` 记录求得的掌位、指角、接触区、各阶段和失败原因。`physics_audit.npz` 保存独立评分用的逐步真值。随机桌面的默认目标为 5 cm。持稳要求指定物体的本体抬升与最低点离桌同时达到目标高度、连续 2 s，期间至少 90% 的时间有手部载荷（超过 0.05 N），相对掌位平移漂移不超过 3 cm，旋转漂移不超过 0.5 rad。仿真使用理想 RGB-D，桌上所有物体在重试和切换间保持物理状态。视觉判定可与离线评分逐项对照。评分分别给出是否到达高度、是否存在连续持稳的 2 秒，以及指令保持结束前的最后 2 秒是否仍满足全部条件。新版独立记录器还保留逐手部刚体的目标载荷，可单独核查掌面与近节实际承力。执行与评分均支持 `--lift-height-m`。历史回合保留原判据，另评 5 cm 时可加 `--lift-height-m 0.05 --out score_5cm.json`。
