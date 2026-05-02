from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DOA5_MAPPING_NAME = "doa5"
DOA5_MAPPING_VERSION = "doa5_db8_official_direct_with_db2_fallback_v2"
DOA5_NAMES = (
    "thumb_rotation",
    "thumb_flexion",
    "index_flexion",
    "middle_flexion",
    "ring_little_flexion",
)


@dataclass(frozen=True)
class MappingTerm:
    """One nonzero term in a provisional linear DoA mapping row."""

    doa_name: str
    glove_column: int
    weight: float
    rationale: str


# Official DB8 supplementary mapping is given as A.T with shape 18 x 5:
# x is the calibrated CyberGlove II vector, y is the five-DoA hand vector, and
# y = A x. DB8 recordings use this matrix directly. The 22-column DB2 fallback
# remaps these same DB8 channels through the anatomical sensor positions shown
# in the local DB2/DB8 glove maps.
DB8_OFFICIAL_A_T = np.array(
    [
        [0.6390, 0.0, 0.0, 0.0, 0.0],
        [0.3830, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [-0.6390, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.6, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.1667],
        [0.0, 0.0, 0.0, 0.0, 0.3333],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.1667],
        [0.0, 0.0, 0.0, 0.0, 0.3333],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.1900, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
DB8_OFFICIAL_W = DB8_OFFICIAL_A_T.T.copy()


# One-based DB8 channel -> one-based DB2 channel. DB2 glove channels 7, 10, 14,
# and 18 are fingertip sensors marked n/a in the DB8 figure, so they receive no
# official DB8 contribution.
DB8_TO_DB2_CHANNELS = (
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    9,
    11,
    12,
    13,
    15,
    16,
    17,
    19,
    20,
    21,
    22,
)


def _build_terms_from_official_matrix() -> tuple[MappingTerm, ...]:
    terms: list[MappingTerm] = []
    for db8_index, db2_channel in enumerate(DB8_TO_DB2_CHANNELS):
        for doa_index, weight in enumerate(DB8_OFFICIAL_A_T[db8_index]):
            if np.isclose(float(weight), 0.0):
                continue
            terms.append(
                MappingTerm(
                    DOA5_NAMES[doa_index],
                    db2_channel - 1,
                    float(weight),
                    f"official DB8 A.T row {db8_index + 1} remapped to DB2 glove channel {db2_channel}",
                )
            )
    return tuple(terms)


DOA5_TERMS = _build_terms_from_official_matrix()


DOA5_SOURCE_NOTES = (
    "Data_Sheet_1.PDF supplementary methods define y = A x from 18 CyberGlove II channels to five IH2 Azzurra hand DoAs.",
    "DB8 recordings with 18 glove channels use the official matrix directly.",
    "DB8_glove.png and DB2_glove.png are only used for the DB2 fallback remap onto DB2's 22-channel glove layout.",
    "For the DB2 fallback, DB2 channels 7, 10, 14, and 18 are fingertip channels marked n/a in the DB8 figure and are intentionally unused.",
)


def build_doa5_matrix() -> np.ndarray:
    """Build the documented 5 x 22 provisional DoA mapping matrix."""
    matrix = np.zeros((len(DOA5_NAMES), 22), dtype=np.float32)
    row_by_name = {name: index for index, name in enumerate(DOA5_NAMES)}
    for term in DOA5_TERMS:
        matrix[row_by_name[term.doa_name], term.glove_column] = np.float32(term.weight)
    return matrix


DOA5_W = build_doa5_matrix()


def apply_linear_doa_mapping(glove_raw: np.ndarray, mapping: str = DOA5_MAPPING_NAME) -> np.ndarray:
    """
    Apply an explicit linear mapping from raw glove channels to semantic DoAs.

    The MVP mapping is provisional and should be validated against target plots
    and downstream control behavior before being treated as durable metadata.
    """
    if mapping != DOA5_MAPPING_NAME:
        raise ValueError(f"unsupported DoA mapping: {mapping}")

    glove = np.asarray(glove_raw, dtype=np.float32)
    if glove.ndim == 1:
        glove = glove[np.newaxis, :]
    if glove.ndim != 2:
        raise ValueError(f"glove_raw must be 1-D or 2-D, got {glove.ndim}-D")
    if glove.shape[1] == DB8_OFFICIAL_W.shape[1]:
        return (glove @ DB8_OFFICIAL_W.T).astype(np.float32)
    if glove.shape[1] != DOA5_W.shape[1]:
        raise ValueError(
            f"glove_raw must have {DB8_OFFICIAL_W.shape[1]} DB8 columns "
            f"or {DOA5_W.shape[1]} DB2 columns, got {glove.shape[1]}"
        )

    return (glove @ DOA5_W.T).astype(np.float32)


def doa5_mapping_metadata() -> dict:
    """Return JSON-safe metadata for reports and experiment summaries."""
    return {
        "mapping_name": DOA5_MAPPING_NAME,
        "mapping_version": DOA5_MAPPING_VERSION,
        "target_names": list(DOA5_NAMES),
        "db2_fallback_matrix": DOA5_W.tolist(),
        "db8_direct_matrix": DB8_OFFICIAL_W.tolist(),
        "terms": [
            {
                "doa_name": term.doa_name,
                "glove_column": term.glove_column,
                "weight": term.weight,
                "rationale": term.rationale,
            }
            for term in DOA5_TERMS
        ],
        "db8_official_a_t": DB8_OFFICIAL_A_T.tolist(),
        "db8_to_db2_channels_1based": list(DB8_TO_DB2_CHANNELS),
        "source_notes": list(DOA5_SOURCE_NOTES),
        "status": "official_db8_matrix_direct_for_18_channel_db8_with_db2_fallback",
    }
