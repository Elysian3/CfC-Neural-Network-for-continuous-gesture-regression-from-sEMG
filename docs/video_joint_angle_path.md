# 视频关节角度路径与风险决策（2026-08-14 外部调研汇总）

## 0. TL;DR

- 相机伪标签首选：**MediaPipe Hand Landmarker（单 RGB）**，但**只直接信任部分自由度**。
- 单目 RGB 动态关节角总体 RMSE 约 **22.5°**（vs Vicon）；受控静态下拇指屈曲可到 mean error **-2.13±2.81°（ICC 0.97）**。误差集中在**深度方向、拇指旋转、握拳遮挡**。
- 因此建议：**目标空间从 5 DoA 改为视频可原生表达的关节角空间（15-22 DoF）**，录制时同时保存 21 个原始 landmarks，让"预测 landmarks(A)"与"预测关节角(B)"两种目标空间可以后验切换，不必现在赌。
- 视频获取按 A→B→C 升级：单 RGB（先试）→ RGB-D/双相机 → Leap 2 / DIY 手套真值。
- 同步用"主机时间戳 + 事件 marker + 事后线性插值"，必须实测 EMG 链路与相机取帧两条延迟，并做 0-100ms 机电延迟扫描。

## 1. 目标空间：5 DoA → 全角度的决策

### 为什么"全角度"方向是对的

单目视频最不可靠的恰好是 5 DoA 里的两个：`thumb_rotation`（2D 几何角不可解，需专门 ML 回归）与握拳时的 `ring_little_flexion`。换成全关节角/landmarks 目标后，视频算不准的自由度可以按置信度降权或丢弃帧，而不是硬凑成一个 5 维标签污染回归。

### 三个候选目标空间

| 目标空间 | 维数 | 优点 | 风险 | 源域数据 |
|---|---|---|---|---|
| A. 21 landmarks | 63（x,y,z） | 视频输出零转换；虚拟手直接渲染 | 坐标是图像系，含手部全局位姿与尺度，跨 session 分布漂移；不是姿态内在量 | DB2 无视频，源域预训练断档 |
| **B. 通用关节角（推荐）** | 15-22 | 视角/尺度不变；EgoEMG 先例（22-DoF 目标）；DB2 22 列手套可近似映射，保留源域预训练 | 需要定义每个角的计算式与 glove↔video 标定 | 可保留 34 人 DB2 |
| C. glove 22 列 | 22 | DB2 零转换 | 视频→手套传感器值映射更难 | 保留 |

**关键工程决策（避免现在赌）**：录制时保存 **原始 21 landmarks + 时间戳 + 置信度**，训练目标随时可以从 landmarks 后验计算（A 或 B）。先做 B 的小规模实验，同时用同一份数据验证 A；数据只录一次。

## 2. 视频角度获取路径与升级路线

| 级别 | 方案 | 关键证据 | 成本/风险 |
|---|---|---|---|
| **A（先试）** | MediaPipe 单 RGB + 规范采集协议（手背朝向、40-70cm、慢速 ramp、置信度丢弃、低通） | 动态 RMSE 22.5°（vs Vicon）；受控 thumb flexion -2.13±2.81° ICC 0.97 | ~0 元；thumb_rotation 与握拳不可靠 |
| **B（视觉兜底）** | RealSense D405/D435 + GMH-D；或双 RGB 相机 + ATHENA 三角化 | GMH-D 把单指敲击指尖 3D 距离误差从 3.49cm 降到 0.10cm；D405 近距深度误差 0.15-0.17mm | $300-540；librealsense 无手部 SDK，需结合 GMH-D |
| **C（真值兜底）** | Leap Motion Controller 2（停产，二手）；LucidGloves/finger-tracker DIY；5DT/Manus | Leap 关节角 RMSE 14.8°，120fps；LucidGloves ~$60 | 采购风险/DIY 标定/商用贵 |

## 3. 精度证据与风险矩阵

| 风险 | 量级（来源） | 影响 | 缓解 |
|---|---|---|---|
| 单目动态关节角总误差 | RMSE 22.5°，r=0.45（MediaPipe vs Vicon, [Sensors 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473350/)） | 全部 5 DoA | 协议放缓、静态+慢速优先、升级 B/C |
| thumb_rotation 不可 2D 直接算 | 专用 ML 回归 RMSE 4.58°，直接几何不可用（[PMC11069636](https://pmc.ncbi.nlm.nih.gov/articles/PMC11069636/)） | thumb_rotation | 侧视相机/RGB-D/手套 |
| 握拳遮挡（MCP/PIP） | OpenPose MCP flexion 平均差 6.82°，MAE 11.93°；静态小指 MCP MAE 15-25°（[PMC9632818](https://pmc.ncbi.nlm.nih.gov/articles/PMC9632818/), [PMC10450288](https://pmc.ncbi.nlm.nih.gov/articles/PMC10450288/)） | index/middle/ring_little | 丢弃遮挡帧、只采张开手型、侧视相机 |
| MediaPipe z 非公制深度 | 原论文：landmark MSE 13.4% palm size；z 仅合成数据监督（[arXiv 2006.10214](https://arxiv.org/abs/2006.10214)） | 3D 角与 thumb rotation | 不用 z 算关键角，或上 RGB-D |
| 帧时间戳不可靠 | `CAP_PROP_POS_MSEC` 实时流不可靠（[OpenCV Issue #8834](https://github.com/opencv/opencv/issues/8834)）；MediaPipe 处理 >33ms（[Issue #3482](https://github.com/google-ai-edge/mediapipe/issues/3482)） | EMG-video 对齐 | `cap.read()` 后立即打主机时间戳；不用帧属性 |
| EMG-视频生理延迟 | 机电延迟 20-50ms，范围 10-100ms（[PubMed 25566091](https://pubmed.ncbi.nlm.nih.gov/25566091/)） | 标签相位错位 | 离线扫描 0-100ms 偏移选最优 |
| 相机帧率欠采样 | 30fps=33ms/帧，快速手指运动欠采样 | 快速 transition | 60fps 全局快门优先 |
| 开源角度库不可靠 | 现成"手指角度库"几乎全 0-star demo | 直接使用会引入隐藏错误 | 自写 landmark→角度计算 + 测试 |

## 4. 同步与采集协议（可直接照做）

1. 单 Windows PC：统一 `time.perf_counter()` 打点，另存 `time.time()` 墙钟。
2. EMG：ESP32 每包带递增 sample counter；PC 在 `serial.read()` 返回后立即记录接收时刻；用序号检测丢包；实测并减去采集链路延迟。
3. 视频：`cap.read()` 返回后立即打点；每帧保存 (frame_index, host_time, landmarks, visibility)。
4. 事件 marker：ESP32 驱动相机可见 LED + EMG 包内 marker flag；屏幕提示配黑白 flash 冗余标记。
5. 对齐：EMG 窗口中心对应视频帧时间（EgoEMG 的 center-frame 监督思想）；标签为帧时刻的关节角，EMG 窗口 `[t-W/2, t+W/2]`；离线线性插值到公共时间线。
6. 协议：5s 动作 + 3s 休息 × 6 次（DB4/5 参考）；每个自由度含慢速 ramp / 保持 / 快速 transition；前后各 5-10s 静息基线。
7. 两条延迟必须实测：EMG 链路（注入电脉冲→PC 收到包）、相机链路（拍毫秒计时器/LED→cap.read 返回）。
8. 参照先例：[EgoEMG](https://github.com/zhenqis123/EgoEMG)（22-DoF 关节角 + timestamps + stale/delta_ms 诊断）、[EMG2Pose](https://github.com/facebookresearch/emg2pose)（软件时间戳 <10ms 验证）。

## 5. 验证门槛（Gate，不达标触发升级）

- **角度精度**：thumb_flexion 与开放手 index/middle：RMSE ≤ 5° 且 ICC ≥ 0.9（Koo & Li excellent）；5-8° 仅作降权伪标签。
- **thumb_rotation / 握拳段**：若 RMSE > 10° 或 ICC < 0.75 → 不允许进训练标签，升级路径 B（RGB-D/双相机）或 C（Leap/手套）。
- **同步**：对齐残差 ≤ 1/2 帧周期（60fps 下 ≤8ms 目标）；EMG 丢包率记录并报告。
- **EMD 扫描**：0-100ms 网格，选使验证 R² 最优的偏移并报告。
- 以上 gate 的测量数据本身保存为 artifacts，进入 `log/`。

## 6. 资源清单

### 仓库/工具
- [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe)（Apache-2.0，主路径）
- [handy-hand-tracking](https://pypi.org/project/handy-hand-tracking/)（仅读 API 用法，角度函数不可信）
- [facebookresearch/frankmocap](https://github.com/facebookresearch/frankmocap)、[InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M)、[FreiHAND](https://github.com/lmb-freiburg/freihand)（已归档/科研用途）
- [geopavlakos/hamer](https://github.com/geopavlakos/hamer)、[WiLoR](https://github.com/rolpotamias/WiLoR)（离线高精度参考；MANO 另有许可）
- [GMH-D](https://github.com/gianluca-amprimo/GMH-D)、[ATHENA](https://github.com/neural-control-and-computation-lab/athena)（RGB-D/多相机）
- [EgoEMG](https://github.com/zhenqis123/EgoEMG)、[EMG2Pose](https://github.com/facebookresearch/emg2pose)、[SeeEMG](https://ieee-dataport.org/documents/seeemg)（协议与同步参照）
- [LucidGloves](https://github.com/LucidVR/lucidgloves)、[finger-tracker](https://github.com/max-titov/finger-tracker)、[Leap Python bindings](https://github.com/ultraleap/leapc-python-bindings)（兜底真值）

### 关键论文
- MediaPipe Hands: [arXiv:2006.10214](https://arxiv.org/abs/2006.10214)（landmark MSE 13.4% palm size；z 为相对深度）
- OpenPose 手指运动学验证: [PMC9632818](https://pmc.ncbi.nlm.nih.gov/articles/PMC9632818/)（遮挡任务 5.03°/6.82° 平均差）
- MediaPipe 拇指运动捕获: [PMC12349048](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349048/)（-2.13±2.81°，ICC 0.97）
- 手机 ROM 自动测量: [PMC10450288](https://pmc.ncbi.nlm.nih.gov/articles/PMC10450288/)（-2.21±9.29°；40.6% 参数达 ±5° LoA）
- 拇指旋转 ML 回归: [PMC11069636](https://pmc.ncbi.nlm.nih.gov/articles/PMC11069636/)（LightGBM RMSE 4.58°）
- 商业相机 vs Vicon: [PMC12473350](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473350/)（MediaPipe 22.5°；Leap 14.8°）
- GMH-D: [arXiv:2308.01088](https://arxiv.org/abs/2308.01088)（深度增强后指尖 3D 误差 3.49→0.10cm）
- EgoEMG: [arXiv:2605.05712](https://arxiv.org/abs/2605.05712)（EMG+vision 同步数据集与 joint-angle 目标）
- ICC 门槛: [Koo & Li 2016](https://pubmed.ncbi.nlm.nih.gov/27330520/)；量角器误差参照: [Reissner 2019](https://link.springer.com/article/10.1186/s13018-019-1177-y)

## 7. 待拍板

1. 目标空间：按 **B（通用关节角 15-22 DoF）** 设计，录制同时保存 21 landmarks 以便后验切换 A —— 是否同意？
2. 视频获取：从 **A（单 RGB + 规范协议 + gate）** 开始，gate 不达标再升级 —— 是否同意？
3. 上位机 Phase 1 仍按 Python 栈（信号 + MediaPipe 双视图 + 3D 虚拟手）？
4. 采集安全 gate（电池供电/元件核对）仍未关闭，进入人体录制前必须关闭。
