# 横放羽毛球的抓姿选择

从 RUA4demo 仓库根目录运行以下命令。示例使用一次 Isaac Sim 回合中头部 D455 和腕部 D405 先后观测到的羽毛球表面，坐标系为工位世界系，单位为米。该示例的桌面高度是 0.7165 m。

```bash
python -m pip install -r DexBridge/requirements-selector.txt
python -m sim.isaac.catalog --prior shuttle \
  --head-mesh sim/isaac/examples/shuttle/head_mesh.npz \
  --wrist-mesh sim/isaac/examples/shuttle/wrist_mesh.npz \
  --table-z 0.7165 --goal any --out runs/shuttle-wrist
```

第一次头部选姿时省略 `--wrist-mesh`，并换用新的输出目录。接近后加入腕部观测，读取新输出的 `candidates.json`。两份输入 NPZ 各含 `vertices_world_m`，形状为 N×3。部署时换成当前目标的世界系观测和当前桌面高度。已有配准结果时也可使用 `--world-from-mesh pose.json`，其中矩阵将侧放先验的米制坐标变换到世界系。

该双视角样例检查 36 类手形，返回 22 个与观测形状及桌面支撑方向相容的候选。每项含 `posture`、`mode_id` 和三个阶段 `pregrasp`、`grasp`、`squeeze`。每个阶段的 `T_world_palm` 是 4×4 掌心位姿，`q_rad` 是按输出 `joint_names` 排列的 22 个关节角。抓起和保持需要另做执行验证。[Isaac Sim 执行说明](../sim/isaac/README.md)

| 仓库中的文件 | 用途 |
|---|---|
| [选择器源码](src/dexbridge/skills/grasp/) | `shapes.py` 和 `silhouette.py` 提取几何与轮廓特征，`postures.py` 排序并计算三阶段姿态，`templates.py` 和 `artifacts.py` 加载并验证索引组件。 |
| [v3 索引](configs/skills/sharpa_grasp_templates.json) · [数值缓存](build/dexonomy/artifacts/) | 38 个 gzip JSON 文件合计 91.46 MiB，分别是 36 份手形模式、1 份排序数据和 1 份形状证据。索引的 `types[*].modes_file`、`selection_file`、`visual_shape_evidence_file` 给出每份文件的相对路径、字节数和 SHA-256；缓存文件名就是完整的 SHA-256。 |
| [横放羽毛球米制模型](../sam3d_glb/shuttle_side_prior.npz) | 与头部及腕部点云配准。模型由 [shuttle 的配置](../sim/isaac/config/objects.json) 固定，SHA-256 为 `df92cba6b33463ece2a305a8a8d520c5f48cf1d0de6cc910a4ef671c33a435d0`。 |
| [两帧样例观测](../sim/isaac/examples/shuttle/) | 取自 `suitea_shuttle/attempt_01`，可在 CPU 上从点云配准一直复算到三个阶段的候选。 |

源码取自 DexBridge `02c21966a113229dc060890ec870a410d4c871ab`。上游索引的 SHA-256 为 `441cad50c6ca1700ba0239829042799d794bc31184ab487fb4c1b8684d928e7a`。仓库版本将缓存随索引一起分发，省去了上游私有归档地址和恢复命令，所有组件保留原始字节与散列。缓存路径相对于包含 `configs/skills/` 的 DexBridge 根目录。
