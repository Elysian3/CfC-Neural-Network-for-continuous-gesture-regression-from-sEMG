• 截至 2026-04-25，这个仓库的现役主线已经不是早期那种 stimulus/restimulus 分类链了，而是“一条面向连续角度回归的传统特征工程 +
  CfC”链路。当前真正起作用的代码主要在 /D:/Project Antikythera/src/dataflow/datapreprocess.py:15、/D:/Project Antikythera/src/
  dataflow/SwRectify.py:162、/D:/Project Antikythera/src/dataflow/feature_extraction.py:285、/D:/Project Antikythera/src/deep
  learning/train.py:787，以及两个实验入口 /D:/Project Antikythera/src/deep learning/run_db2_subject_adaptation.py:293 和 /D:/
  Project Antikythera/src/deep learning/run_db2_single_subject.py:156。src/hardwareOperation/load-model.cpp 目前还是空的，占位而
  已。

  当前项目设定

  - 目标任务现在是 sEMG -> 连续关节角度 回归，不是分类。
  - 数据主源是 NinaPro DB2，load_data() 会把 emg / glove / inclin / stimulus / restimulus ... 都读出来，但主动训练链只真正用 emg +
    一个连续目标族，/D:/Project Antikythera/src/dataflow/datapreprocess.py:15。
  - target_source 设计成二选一：要么 glove，要么 inclin，不混合；当前所有活跃实验都在用 glove，/D:/Project Antikythera/src/
    dataflow/feature_extraction.py:258。
  - 预处理固定为 FS=2000、去均值、50 Hz notch、20-450 Hz 四阶 Butterworth 带通，/D:/Project Antikythera/src/dataflow/
    datapreprocess.py:7、/D:/Project Antikythera/src/dataflow/datapreprocess.py:60。
  - 滑窗固定为 200 ms 窗、50 ms 步长，/D:/Project Antikythera/src/dataflow/SwRectify.py:30。
  - 每个窗同时保留 unrectified 和 rectified EMG；目标对齐方式支持 last / center / mean，当前默认是 last，target_offset_samples=0，/
    D:/Project Antikythera/src/dataflow/SwRectify.py:123、/D:/Project Antikythera/src/dataflow/SwRectify.py:162。
  - 特征固定为 MAV, MAVS, WL, ZC, SSC，即每通道 5 个特征；12 通道时总输入维度是 60，/D:/Project Antikythera/src/dataflow/
    feature_extraction.py:33、/D:/Project Antikythera/src/dataflow/feature_extraction.py:165。
  - ZC/SSC 的阈值不是学习得到的，而是按通道平均幅值乘 0.01 的启发式阈值。
  - CfC 训练是 many-to-one：一个长度为 seq_len=8 的窗口序列，只监督最后一个窗口对应的角度；也就是“它确实在用每个序列最后一个角度做
    预测”，/D:/Project Antikythera/src/deep learning/train.py:169、/D:/Project Antikythera/src/deep learning/train.py:351。
  - 当前固定超参数在 /D:/Project Antikythera/src/deep learning/train.py:40、/D:/Project Antikythera/src/deep learning/train.py:84：
    hidden_units=64、batch_size=128、lr=1e-3、weight_decay=1e-5、max_epochs=20、patience=5、clip=1.0、num_workers=0。
  - split 有两类。blocked_time 是在同一批录波内按时间切 70/15/15，并留 16 个窗口的 gap，/D:/Project Antikythera/src/deep learning/
    train.py:289、/D:/Project Antikythera/src/deep learning/train.py:454；recording 是按整段录波划分，/D:/Project Antikythera/src/
    deep learning/train.py:224。

  当前逻辑链

  1. .mat 文件先由 load_data() 读入原始字段，/D:/Project Antikythera/src/dataflow/datapreprocess.py:15。
  2. preprocess_emg() 做滤波，得到干净 EMG，/D:/Project Antikythera/src/dataflow/datapreprocess.py:60。
  3. run_feature_pipeline() 先选一个目标族和目标列，例如 glove[:, 5]，/D:/Project Antikythera/src/dataflow/
     feature_extraction.py:258、/D:/Project Antikythera/src/dataflow/feature_extraction.py:285。
  4. sliding_window() 把 EMG 切成重叠窗口，并把连续角度按窗口末端或中心对齐，/D:/Project Antikythera/src/dataflow/
     SwRectify.py:162。
  5. extract_emg_features() 把每个窗口变成 60 维特征向量，/D:/Project Antikythera/src/dataflow/feature_extraction.py:165。
  6. train.py 再把这些窗口拼成长度 8 的短序列，/D:/Project Antikythera/src/deep learning/train.py:351 或按 blocked-time 建 train/
     val/test，/D:/Project Antikythera/src/deep learning/train.py:454。
  7. 只用训练集统计量做 z-score 归一化，/D:/Project Antikythera/src/dataflow/feature_extraction.py:344、/D:/Project Antikythera/
     src/deep learning/train.py:576。
  8. CfCRegressor 输出最后一个时间步，evaluate_split() 反归一化后计算 MAE / RMSE / R²，/D:/Project Antikythera/src/deep learning/
     train.py:605、/D:/Project Antikythera/src/deep learning/train.py:787。
  9. 当前有两条实验线：跨受试者预训练 + 1 epoch 适配，/D:/Project Antikythera/src/deep learning/run_db2_subject_adaptation.py:293；
     以及单受试者可行性验证，/D:/Project Antikythera/src/deep learning/run_db2_single_subject.py:156。

  现在实际做到哪

  - 跨受试者主线仍然偏弱。当前保存结果里，S32 上 glove[:,10] 的 zero-shot R²=0.221，1 epoch adaptation 后 R²=0.270，/D:/Project
    Antikythera/log/db2_subject_adaptation/summary.json。
  - 单受试者 S1 下，如果沿用默认目标 target_column=10，测试集 R²=-0.104，/D:/Project Antikythera/log/db2_single_subject/
    S1_target10/summary.json。
  - 单受试者 S1 下，如果换成 target_column=5，测试集 R²=0.550，已经过了你设的 0.5 门槛，/D:/Project Antikythera/log/
    db2_single_subject/S1_target5/summary.json、/D:/Project Antikythera/log/db2_single_subject/S1_target5/
    S1_target5_prediction.png。
  - 这说明仓库目前已经证明了“某些 DoF 上，单受试者内的传统特征 + CfC 是可行的”；但它还没有证明“默认目标定义是对的”，更没有证明“跨受
    试者可部署”。

  现在的问题

  - 最大问题不是模型先天不行，而是“目标定义还没稳定”。train.py 的默认配置仍是 target_columns=(10,)，但当前最像样的单受试者结果来自
    column 5。如果主目标 DoF 不先定死，后面所有比较都会失真。
  - 跨受试者泛化仍然明显不足。R² 0.27 只能算有信号，不够做可靠连续控制。
  - 过拟合/分布漂移都存在。很多实验最佳 epoch 很早，常常第 1 或第 2 轮就到顶，之后训练 loss 继续降，但 val/test 不再改善。
  - 特征工程仍然偏“轻量但启发式”。MAV/MAVS/WL/ZC/SSC 这套足够快，但 ZC/SSC 的阈值还没有严谨标定。
  - 时间对齐仍未真正研究。target_offset_samples 现在一直是 0，等于默认“EMG 与角度没有延迟”，这对真实肌电到运动关系往往过于理想。
  - 当前 single-subject 的 blocked_time 只能证明“同人、同批录波内部按时间分块可学”，不能证明“跨 session / 跨 exercise 的强鲁棒性”。
  - 仓库说明文档和代码现状不完全同步。CLAUDE.md 还保留了旧的 src/models、分类/下肢等表述；datapreprocess.py 的 docstring 里还有 20-
    50 Hz 的文字错误，但代码常量实际上是 20-450 Hz。
  - 注释存在编码污染和少量历史残留，影响整体专业度。
  - 部署链几乎还没开始。load-model.cpp 为空，没有导出、量化、边端推理延迟、RAM 占用等实际闭环。
  - 测试覆盖主要还是合成数据单元测试，/D:/Project Antikythera/tests/test_regression_pipeline.py:17、/D:/Project Antikythera/tests/
    test_cfc_training.py:21、/D:/Project Antikythera/tests/test_single_subject_training.py:15 能锁住逻辑，但还没有自动化的真实 DB2
    回归 smoke test。

  一句话总结：这个项目当前已经有一条清晰、可运行的“传统特征工程 + CfC 连续回归”主线，单受试者可行性已经被 S1 / glove[:,5] / R²=0.55
  证明；但它仍卡在三个核心问题上：目标 DoF 尚未定死、跨受试者泛化明显不足、部署链还没真正落地。

  今天和下周目标（2026-04-26 起）

  - 第一目标是把具体 DoF 定死。这里的 channel 暂按目标通道/DoF 理解，也就是 glove 或 inclin 的 target_column，不是 12 个 EMG 电极通道。
  - 今天先做 DoF 筛选闭环：固定数据源、split、窗口、特征、seq_len、seed 和 metric，对候选 target_column 做同协议横向扫描，输出每个 DoF 的 MAE / RMSE / R²、最佳 epoch、预测图和失败原因。
  - DoF 定死标准不能只看单次最高 R²；至少要同时满足：test R² 达标、val/test 不明显背离、预测曲线相位合理、不同 subject 或 recording 下不完全崩掉。
  - 当前临时强候选是 glove[:,5]，因为 S1 single-subject blocked_time 已有 test R²=0.550；glove[:,10] 暂时只能作为旧默认基线，不应继续默认代表主目标。
  - 下周目标是在所有入选 DoF/target channels 上，让传统特征 + CfC 至少达到 RNN baseline 的能力。这里的“达到”定义为：同一数据、同一 split、同一特征输入、同一 metric 下，CfC 的 test R² 不低于 RNN，且 MAE/RMSE 不显著更差。
  - 如果当前仓库没有稳定 RNN baseline，必须先补一个最小可比 RNN/GRU/LSTM baseline；否则“达到 RNN 能力”没有可验证含义。
  - 每个 channel 的最终表格必须包含：target_source、target_column、split 策略、seq_len、target_offset_samples、CfC 指标、RNN 指标、差值、是否通过、主要风险。
  - 本周不再把模型结构调参放在第一优先级；先把目标定义、baseline 对照和 channel 级别验收矩阵做稳。否则后面的优化都会建立在不稳定目标上。
