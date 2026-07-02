# Graph Report - src  (2026-06-09)

## Corpus Check
- Corpus is ~19,757 words - fits in a single context window. You may not need a graph.

## Summary
- 338 nodes · 764 edges · 11 communities
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 112 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Training Infrastructure|Training Infrastructure]]
- [[_COMMUNITY_Experiment Config & Utilities|Experiment Config & Utilities]]
- [[_COMMUNITY_Signal Processing & Feature Extraction|Signal Processing & Feature Extraction]]
- [[_COMMUNITY_ATL & Domain Adaptation|ATL & Domain Adaptation]]
- [[_COMMUNITY_Data Loading & Experiment IO|Data Loading & Experiment I/O]]
- [[_COMMUNITY_Semantic Validation|Semantic Validation]]
- [[_COMMUNITY_Single-Subject Feasibility|Single-Subject Feasibility]]
- [[_COMMUNITY_Model Architectures (CfCDenseCfC)|Model Architectures (CfC/DenseCfC)]]
- [[_COMMUNITY_Sliding Window & Target Alignment|Sliding Window & Target Alignment]]
- [[_COMMUNITY_INT8 Quantization & Deployability|INT8 Quantization & Deployability]]

## God Nodes (most connected - your core abstractions)
1. `Any` - 27 edges
2. `SequenceSplit` - 26 edges
3. `train_cfc_regressor()` - 24 edges
4. `run_candidate_selection()` - 21 edges
5. `run_protocol()` - 20 edges
6. `run_subject_adaptation()` - 19 edges
7. `run_doa5_subject_adaptation()` - 19 edges
8. `evaluate_split()` - 16 edges
9. `ndarray` - 14 edges
10. `DomainDiscriminator` - 14 edges

## Surprising Connections (you probably didn't know these)
- `load_subject_filtered_split()` --calls--> `sliding_window()`  [INFERRED]
  src/deep learning/run_db2_paper_cfc_finetune.py → src/dataflow/SwRectify.py
- `load_recording_features()` --calls--> `run_feature_pipeline()`  [INFERRED]
  src/deep learning/train.py → src/dataflow/feature_extraction.py
- `run_protocol()` --calls--> `fit_feature_normalizer()`  [INFERRED]
  src/deep learning/run_db2_paper_cfc_finetune.py → src/dataflow/feature_extraction.py
- `train_cfc_regressor()` --calls--> `fit_feature_normalizer()`  [INFERRED]
  src/deep learning/train.py → src/dataflow/feature_extraction.py
- `normalize_sequence_inputs()` --calls--> `apply_feature_normalizer()`  [INFERRED]
  src/deep learning/train.py → src/dataflow/feature_extraction.py

## Import Cycles
- None detected.

## Communities (11 total, 0 thin omitted)

### Community 0 - "Training Infrastructure"
Cohesion: 0.06
Nodes (61): _append_blocked_ranges(), _append_sequences_from_window_range(), _append_sequences_to_split(), apply_target_normalizer(), build_action_stratified_sequence_splits(), build_best_cfc_config(), build_blocked_sequence_splits(), build_sequence_split() (+53 more)

### Community 1 - "Experiment Config & Utilities"
Cohesion: 0.09
Nodes (55): doa5_mapping_metadata(), Return JSON-safe metadata for reports and experiment summaries., adaptation_trainable_prefixes(), apply_adaptation_mode(), assert_selection_summary_is_uncontaminated(), _autoncp_layer_count(), build_protocol_context(), _candidate_by_config() (+47 more)

### Community 2 - "Signal Processing & Feature Extraction"
Cohesion: 0.07
Nodes (48): load_data(), preprocess_emg(), Load data from a .mat file using scipy.io.loadmat.      Parameters     ---------, Apply the full preprocessing pipeline to raw EMG signals.      Pipeline (in orde, apply_linear_doa_mapping(), build_doa5_matrix(), _build_terms_from_official_matrix(), MappingTerm (+40 more)

### Community 3 - "ATL & Domain Adaptation"
Cohesion: 0.12
Nodes (36): _atl_training_epoch(), _augment_sequence_batch(), _compute_dann_lambda(), concat_splits(), copy_split_by_indices(), fine_tune_head(), freeze_for_linear_head(), load_subject_exercise_split() (+28 more)

### Community 4 - "Data Loading & Experiment I/O"
Cohesion: 0.10
Nodes (32): discover_target_files(), load_sequence_split_from_files(), main(), make_jsonable(), parse_args(), plot_subject_adaptation(), Sort subject ids numerically instead of lexicographically., Group usable DB2 recordings by subject id. (+24 more)

### Community 5 - "Semantic Validation"
Cohesion: 0.17
Nodes (24): apply_validation_first_decisions(), best_epoch_value(), build_candidate_matrix(), build_overall_summary(), candidate_output_dir(), is_stable(), main(), metric_value() (+16 more)

### Community 6 - "Single-Subject Feasibility"
Cohesion: 0.17
Nodes (18): discover_subject_recordings(), main(), make_jsonable(), parse_args(), Sort one subject's recordings by exercise number, then by the full file name., Return all usable recordings for one subject.      Only files that contain bot, Extract the checkpoint selected by validation MAE., Convert numpy-like or path-like values into JSON-safe objects. (+10 more)

### Community 7 - "Model Architectures (CfC/DenseCfC)"
Cohesion: 0.12
Nodes (8): CfCRegressor, DenseCfCLinearRegressor, Thin PyTorch wrapper around pre-built sequence arrays., Small many-to-one CfC regressor.      The CfC layer emits an output at every tim, Dense CfC encoder with an explicit linear DoA readout head., Return (prediction, pre_dropout_state) tuple.          The pre-dropout state is, SequenceRegressionDataset, Tensor

### Community 8 - "Sliding Window & Target Alignment"
Cohesion: 0.18
Nodes (17): _1d_to_2d_of_target_array(), align_targets_to_windows(), _normalize_target_names(), print_window_summary(), SwRectify.py - regression-oriented EMG windowing and target alignment ==========, Choose one representative sample index for each window.      The anchor is the e, Align one target vector to each EMG window.      Modes:     - `last`: use the la, Slice a continuous EMG recording into overlapping windows.      Compared with th (+9 more)

### Community 9 - "INT8 Quantization & Deployability"
Cohesion: 0.27
Nodes (10): compute_size_stats(), dequantize_weights(), load_model(), main(), quantize_weights_int8(), INT8 quantization feasibility test for AutoNCP CfC model.  Tests manual weight q, Quantize weight tensors to INT8 with per-tensor symmetric quantization.      Ret, Dequantize INT8 weights back to FP32 for inference. (+2 more)

## Knowledge Gaps
- **5 isolated node(s):** `ndarray`, `Figure`, `Optimizer`, `Figure`, `Module`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train_cfc_regressor()` connect `Training Infrastructure` to `Experiment Config & Utilities`, `Signal Processing & Feature Extraction`, `ATL & Domain Adaptation`, `Data Loading & Experiment I/O`, `Single-Subject Feasibility`?**
  _High betweenness centrality (0.255) - this node is a cross-community bridge._
- **Why does `run_single_subject_experiment()` connect `Single-Subject Feasibility` to `Training Infrastructure`, `Semantic Validation`?**
  _High betweenness centrality (0.215) - this node is a cross-community bridge._
- **Why does `main()` connect `Semantic Validation` to `Single-Subject Feasibility`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Any` (e.g. with `SequenceSplit` and `WeightedSmoothL1Loss`) actually correct?**
  _`Any` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `SequenceSplit` (e.g. with `Any` and `DataLoader`) actually correct?**
  _`SequenceSplit` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `train_cfc_regressor()` (e.g. with `run_single_subject_experiment()` and `run_subject_adaptation()`) actually correct?**
  _`train_cfc_regressor()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `run_candidate_selection()` (e.g. with `doa5_mapping_metadata()` and `resolve_device()`) actually correct?**
  _`run_candidate_selection()` has 3 INFERRED edges - model-reasoned connections that need verification._