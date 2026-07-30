<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-23 | Updated: 2026-05-23 -->

# tests

## Purpose

Unit and integration tests for the dataflow pipeline and training protocols. Tests use synthetic data exclusively — no real recordings are required, so the suite runs without the full dataset.

## Key Files

| File | Description |
|------|-------------|
| `test_regression_pipeline.py` | Tests for `SwRectify.sliding_window`, `doa_mapping.apply_linear_doa_mapping`, `feature_extraction.extract_emg_features`, `run_feature_pipeline`, `prepare_regression_data` |
| `test_cfc_training.py` | Tests model construction, paper-protocol defaults, ATL, sequence splitting, shared normalization, metrics, and loss functions |
| `test_hardware_preflight.py` | Tests deployment metadata, INT8 preflight, SRAM estimates, canonical C ownership, and export defaults |

## First-Principles Testing Strategy

### Why synthetic data?

Real DB2 recordings are large (tens of MB each) and not committed to the repo. Tests must run in CI without dataset access. Synthetic arrays verify:
- **Shape correctness**: window geometry math is exact (integer sample counts)
- **Numerical correctness**: feature values for known inputs are computable by hand
- **Edge cases**: zero-length signals, single-window recordings, threshold behavior

### What each test file verifies

1. **test_regression_pipeline.py**: The data contract. Verifies that `sliding_window` produces correct start/end/center indices, that target alignment modes (last/center/mean) produce expected values, that the DoA mapping matrix has the right shape and produces expected outputs, and that `run_feature_pipeline` doesn't crash on synthetic data.

2. **test_cfc_training.py**: Model correctness. Verifies that
   `DenseCfCLinearRegressor` produces expected shapes, normalization round-trips,
   metrics match manual computation, losses apply weights correctly, and split
   strategies do not produce empty partitions.

3. **test_hardware_preflight.py**: Deployment-contract correctness. Verifies
   accepted metadata, deterministic golden tensors, explicit memory estimates,
   and that PC/export tooling uses `firmware/main` as the only C source.

## Working Principles (Applied to Testing)

These are the same principles from the root `AGENTS.md`, applied specifically to test code. Tests are the enforcement mechanism for every other principle.

1. **Test-Driven — This Is Where It Lives**: This directory IS Principle 1. Every test here protects a specific contract. If a test fails and you're tempted to delete it rather than fix the code: stop. The test is telling you something. Listen to it. A deleted test is a deleted safety net — the next person will fall through.

2. **Ask First, Never Guess — Test Edition**: If you're not sure what a test should assert, ask. A test that asserts `assertTrue(True)` is worse than no test — it creates a false sense of security. Every assertion must verify a named contract: "sliding_window with fs=10, window_ms=400, stride_ms=200 on 10 samples produces exactly 4 windows."

3. **Scientific Rigor — Test Edition**: Tests are reproducibility artifacts. Each test function name must describe what contract it verifies. Use `test_last_sample_alignment` not `test_alignment_1`. The test body should make the expected value computable by a reader without running the code — hand-computed expected values, not regression-tested magic numbers.

4. **Dual Role**: If you see code that has no test coverage, flag it. If you see a test that only checks the happy path, ask: what happens with zero-length input? Single-sample input? NaN input? The test suite should be adversarial — it should try to break the code.

5. **Engineering Five Steps — Test Edition**:
   - **Question**: What contract does this test verify? If you can't state it, you're not testing — you're just running code.
   - **Delete**: Dead tests (testing removed functions, testing obsolete behavior) are lies. Remove them.
   - **Simplify**: One test method = one contract. Don't test five unrelated things in one method.
   - **Accelerate**: Tests must run fast (<5 seconds total). Synthetic data. No file I/O. No GPU. No real recordings.
   - **Automate**: `pytest tests/` is the automation. If it can't be run with one command, it's not automated.

6. **Explain Every Action**: Every test assertion should have a comment explaining the expected value. "np.testing.assert_array_equal(windows['window_start_indices'], np.array([0, 2, 4, 6]))  # 10 samples, 4-sample window, 2-sample stride → 4 windows starting at 0,2,4,6."

7. **Strict Intent Attribution**: Test coverage decisions must be owned, not deflected. "I skipped testing the edge case because the current synthetic data generator can't produce NaN windows" — this is the agent owning a gap. "The test coverage is complete" when known edge cases are untested is a false claim. In testing, misattribution of completeness is as damaging as a false positive — it teaches the wrong safety boundary.

8. **Hardware-gated design (Principle 8):** Any model construction test must include an assertion that the parameter count does not exceed the ESP32-S3 memory budget (~400 KB FP32, or equivalently ~100,000 parameters). This is a compile-time check: if a model architecture change accidentally doubles the parameter count, the test must catch it before the experiment runs.

## For AI Agents

### Working In This Directory
- Use `unittest.TestCase` (standard library) — no pytest-specific features.
- Tests import code by manipulating `sys.path` to add `src/dataflow` or `src/deep learning`.
- Always use synthetic arrays created with `np.arange`, `np.zeros`, etc. — never reference real data files.
- Mock heavy operations (file I/O, model training) with `unittest.mock.patch` when testing protocol-level logic.
- For new pipeline features, add tests to `test_regression_pipeline.py`.
- For new training protocols, add tests to `test_cfc_training.py` or create a new test file.

### Testing Requirements
- Run with: `pytest tests/` from project root (or `python -m pytest tests/`)
- All tests must pass before committing.
- Test files are named `test_*.py` for auto-discovery.

## Dependencies

### Internal
- `src/dataflow/` — all pipeline modules under test
- `src/deep learning/` — training modules under test

### External
- `unittest` (stdlib) — test framework
- `numpy` — synthetic data generation
- `torch` — model construction tests
