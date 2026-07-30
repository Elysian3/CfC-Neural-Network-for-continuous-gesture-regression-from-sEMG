# Graph Report - .  (2026-07-22)

## Corpus Check
- 68 files · ~4,675,700 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 523 nodes · 1108 edges · 24 communities (23 shown, 1 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 184 edges (avg confidence: 0.73)
- Token cost: 41,000 input · 17,645 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Subject Adaptation Pipelines|Subject Adaptation Pipelines]]
- [[_COMMUNITY_CfC Fine-tuning & ATL|CfC Fine-tuning & ATL]]
- [[_COMMUNITY_Training Infrastructure|Training Infrastructure]]
- [[_COMMUNITY_ESP32-S3 Firmware|ESP32-S3 Firmware]]
- [[_COMMUNITY_Pipeline Architecture Concepts|Pipeline Architecture Concepts]]
- [[_COMMUNITY_Hardware Preflight Validation|Hardware Preflight Validation]]
- [[_COMMUNITY_CfC Model Architecture|CfC Model Architecture]]
- [[_COMMUNITY_Semantic DoF Screening|Semantic DoF Screening]]
- [[_COMMUNITY_EMG Feature Extraction|EMG Feature Extraction]]
- [[_COMMUNITY_DoA Mapping (GloVe)|DoA Mapping (GloVe)]]
- [[_COMMUNITY_Weight Export & Quantization|Weight Export & Quantization]]
- [[_COMMUNITY_Sliding Window Rectification|Sliding Window Rectification]]
- [[_COMMUNITY_Single Subject Training|Single Subject Training]]
- [[_COMMUNITY_OMC Internal State|OMC Internal State]]
- [[_COMMUNITY_OMC Internal State|OMC Internal State]]
- [[_COMMUNITY_OMC Internal State|OMC Internal State]]
- [[_COMMUNITY_OMC Internal State|OMC Internal State]]
- [[_COMMUNITY_Data Preprocessing|Data Preprocessing]]
- [[_COMMUNITY_OMC Internal State|OMC Internal State]]
- [[_COMMUNITY_Feature Normalization|Feature Normalization]]
- [[_COMMUNITY_Target Column Selection|Target Column Selection]]
- [[_COMMUNITY_OMC Internal State|OMC Internal State]]

## God Nodes (most connected - your core abstractions)
1. `run_protocol()` - 28 edges
2. `Any` - 27 edges
3. `SequenceSplit` - 27 edges
4. `train_cfc_regressor()` - 25 edges
5. `run_candidate_selection()` - 21 edges
6. `run_subject_adaptation()` - 19 edges
7. `run_doa5_subject_adaptation()` - 19 edges
8. `run_preflight()` - 17 edges
9. `evaluate_split()` - 16 edges
10. `ndarray` - 15 edges

## Surprising Connections (you probably didn't know these)
- `Binary domain classifier for GRL-based domain adaptation.` --rationale_for--> `DenseCfC Linear Regressor`  [EXTRACTED]
  src/deep learning/train.py → src/deep learning/AGENTS.md
- `Dense CfC encoder with an explicit linear DoA readout head.` --rationale_for--> `DenseCfC Linear Regressor`  [EXTRACTED]
  src/deep learning/train.py → src/deep learning/AGENTS.md
- `load_subject_filtered_split()` --calls--> `sliding_window()`  [INFERRED]
  src/deep learning/run_db2_paper_cfc_finetune.py → src/dataflow/SwRectify.py
- `load_subject_filtered_split()` --calls--> `load_data()`  [INFERRED]
  src/deep learning/run_db2_paper_cfc_finetune.py → src/dataflow/datapreprocess.py
- `run_protocol()` --calls--> `load_data()`  [INFERRED]
  src/deep learning/run_db2_paper_cfc_finetune.py → src/dataflow/datapreprocess.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Signal Processing Pipeline Stages** — dataflow_agents_datapreprocess, dataflow_agents_swrectify, dataflow_agents_feature_extraction, dataflow_agents_doa_mapping_stage [EXTRACTED 1.00]
- **EMG Feature Set (5 Features)** — dataflow_agents_mav, dataflow_agents_mavs, dataflow_agents_wl, dataflow_agents_zc, dataflow_agents_ssc [EXTRACTED 1.00]
- **Training Protocol Phases** — deep_learning_agents_pretrain, deep_learning_agents_finetune, deep_learning_agents_atl [EXTRACTED 1.00]

## Communities (24 total, 1 thin omitted)

### Community 0 - "Subject Adaptation Pipelines"
Cohesion: 0.06
Nodes (77): doa5_mapping_metadata(), Return JSON-safe metadata for reports and experiment summaries., load_sequence_split_from_files(), main(), make_jsonable(), parse_args(), plot_subject_adaptation(), Sort subject ids numerically instead of lexicographically. (+69 more)

### Community 1 - "CfC Fine-tuning & ATL"
Cohesion: 0.07
Nodes (60): DataLoader, _atl_training_epoch(), _augment_sequence_batch(), _compute_dann_lambda(), _compute_doa_metrics_from_glove(), concat_splits(), copy_split_by_indices(), _discover_subjects() (+52 more)

### Community 2 - "Training Infrastructure"
Cohesion: 0.06
Nodes (61): load_subject_exercise_split(), discover_target_files(), Group usable DB2 recordings by subject id., _append_blocked_ranges(), _append_sequences_from_window_range(), _append_sequences_to_split(), apply_target_normalizer(), build_action_stratified_sequence_splits() (+53 more)

### Community 3 - "ESP32-S3 Firmware"
Cohesion: 0.10
Nodes (23): cfc_inference_int8(), cfc_lut_init(), cfc_step(), fc_int8(), lut_lookup(), mulaw_encode(), extract_rms_features(), ringbuf_init() (+15 more)

### Community 4 - "Pipeline Architecture Concepts"
Cohesion: 0.09
Nodes (31): Data Container, Signal Processing Pipeline, Data Preprocessing Stage, DoA Mapping Stage, 18 DoF to 5 DoA Reduction Rationale, Feature Extraction Stage, Feature Selection Rationale, EMG Filter Design Rationale (+23 more)

### Community 5 - "Hardware Preflight Validation"
Cohesion: 0.16
Nodes (28): checkpoint_sha256(), CheckpointBundle, compute_error_report(), compute_size_stats(), dequantize_weights(), dequantize_weights_per_channel(), estimate_core_sram_kb(), estimate_dense_macs() (+20 more)

### Community 6 - "CfC Model Architecture"
Cohesion: 0.10
Nodes (21): Closed-form Continuous-time (CfC), DenseCfC Linear Regressor, build_cfc_regressor(), CfCRegressor, Thin PyTorch wrapper around pre-built sequence arrays., Binary domain classifier for GRL-based domain adaptation., Small many-to-one CfC regressor.      The CfC layer emits an output at every tim, Dense CfC encoder with an explicit linear DoA readout head. (+13 more)

### Community 7 - "Semantic DoF Screening"
Cohesion: 0.17
Nodes (24): apply_validation_first_decisions(), best_epoch_value(), build_candidate_matrix(), build_overall_summary(), candidate_output_dir(), is_stable(), main(), metric_value() (+16 more)

### Community 8 - "EMG Feature Extraction"
Cohesion: 0.16
Nodes (21): extract_emg_features(), mean_absolute_value(), mean_absolute_value_slope(), feature_extraction.py - window-level EMG features for continuous targets =======, Compute the window-to-window change in MAV.      This measures how quickly activ, Compute waveform length for each window and channel., Compute RMS amplitude for every window and channel., Count zero crossings per window and channel.      A crossing is counted only whe (+13 more)

### Community 9 - "DoA Mapping (GloVe)"
Cohesion: 0.14
Nodes (20): apply_linear_doa_mapping(), build_doa5_matrix(), _build_terms_from_official_matrix(), glove_to_doa(), MappingTerm, Build the documented 5 x 22 provisional DoA mapping matrix., Apply an explicit linear mapping from raw glove channels to semantic DoAs., Extract the 13 non-zero-weight glove columns from the full 22-column array. (+12 more)

### Community 10 - "Weight Export & Quantization"
Cohesion: 0.17
Nodes (20): _array_literal(), compute_error(), _compute_mulaw_stats_from_data(), generate_normalization_h(), generate_weights_h(), main(), quantize_per_channel(), Export DenseCfC weights as per-channel INT8 C header files.  Generates ``weights (+12 more)

### Community 11 - "Sliding Window Rectification"
Cohesion: 0.18
Nodes (18): _1d_to_2d_of_target_array(), align_targets_to_windows(), _normalize_target_names(), print_window_summary(), SwRectify.py - regression-oriented EMG windowing and target alignment ==========, Choose one representative sample index for each window.      The anchor is the e, Align one target vector to each EMG window.      Modes:     - `last`: use the la, Slice a continuous EMG recording into overlapping windows.      Compared with th (+10 more)

### Community 12 - "Single Subject Training"
Cohesion: 0.17
Nodes (18): discover_subject_recordings(), main(), make_jsonable(), parse_args(), Sort one subject's recordings by exercise number, then by the full file name., Return all usable recordings for one subject.      Only files that contain bot, Extract the checkpoint selected by validation MAE., Convert numpy-like or path-like values into JSON-safe objects. (+10 more)

### Community 13 - "OMC Internal State"
Cohesion: 0.15
Nodes (12): active, last_checked_at, max_reinforcements, _meta, mode, sessionId, written_at, reinforcement_count (+4 more)

### Community 14 - "OMC Internal State"
Cohesion: 0.17
Nodes (11): active, awaiting_confirmation, awaiting_confirmation_set_at, iteration, last_checked_at, linked_ultrawork, max_iterations, project_path (+3 more)

### Community 15 - "OMC Internal State"
Cohesion: 0.20
Nodes (9): active, awaiting_confirmation, awaiting_confirmation_set_at, last_checked_at, original_prompt, project_path, reinforcement_count, session_id (+1 more)

### Community 16 - "OMC Internal State"
Cohesion: 0.33
Nodes (8): last_emitted_at_ms, message, entries, 1bb6914bc42b891c26d46c8ae2fd4f44071f59862e89dc550f4c82877b700216, 445ed27a3872b681d98190bae61ccb84954e1bc4e140df5370be958dee776b3a, 79a93d4a2f8f50b95f852280616242fee1855dc99a3c75211917f55e72e95fae, updated_at, version

### Community 17 - "Data Preprocessing"
Cohesion: 0.29
Nodes (7): load_data(), preprocess_emg(), Load data from a .mat file using scipy.io.loadmat.      Parameters     ---------, Apply the full preprocessing pipeline to raw EMG signals.      Pipeline (in orde, Execute the full feature pipeline from one .mat file to aligned features.      T, run_feature_pipeline(), ndarray

### Community 18 - "OMC Internal State"
Cohesion: 0.33
Nodes (5): error, retry_count, timestamp, tool_input_preview, tool_name

### Community 19 - "Feature Normalization"
Cohesion: 0.33
Nodes (6): apply_feature_normalizer(), fit_feature_normalizer(), prepare_regression_data(), Fit normalization statistics for a feature matrix.      Training code should fit, Apply precomputed feature normalization statistics., Convert one feature set into `(x, y)` regression arrays.      This function does

### Community 20 - "Target Column Selection"
Cohesion: 0.50
Nodes (4): Normalize target column selection into a validated list of indices., Select one continuous target family from the loaded .mat recording.      We inte, _resolve_target_columns(), _select_targets()

## Knowledge Gaps
- **54 isolated node(s):** `ndarray`, `Figure`, `Optimizer`, `Module`, `Figure` (+49 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train_cfc_regressor()` connect `Training Infrastructure` to `Subject Adaptation Pipelines`, `CfC Fine-tuning & ATL`, `CfC Model Architecture`, `DoA Mapping (GloVe)`, `Single Subject Training`, `Feature Normalization`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **Why does `run_single_subject_experiment()` connect `Single Subject Training` to `Training Infrastructure`, `Semantic DoF Screening`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Why does `DenseCfC Linear Regressor` connect `CfC Model Architecture` to `CfC Fine-tuning & ATL`, `Training Infrastructure`, `Pipeline Architecture Concepts`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Are the 44 inferred relationships involving `ValueError` (e.g. with `apply_linear_doa_mapping()` and `glove_to_doa()`) actually correct?**
  _`ValueError` has 44 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `run_protocol()` (e.g. with `load_data()` and `fit_feature_normalizer()`) actually correct?**
  _`run_protocol()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Any` (e.g. with `SequenceSplit` and `WeightedSmoothL1Loss`) actually correct?**
  _`Any` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `SequenceSplit` (e.g. with `Namespace` and `SequenceSplit`) actually correct?**
  _`SequenceSplit` has 17 INFERRED edges - model-reasoned connections that need verification._