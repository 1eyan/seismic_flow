from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union


@dataclass(frozen=True)
class SegyProfile:
    """Runtime schema for interpreting SEG-Y headers and H5 trace keys."""

    name: str
    description: str
    key_columns: Tuple[str, ...]
    byte_pos: Mapping[str, int]
    h5_fallback: Mapping[str, str]
    gather_modes: Mapping[str, Tuple[str, ...]]
    sort_keys: Tuple[str, ...]
    default_gather_mode: str = "survey_shot"
    default_header_mode: str = "fixed"

    @property
    def all_h5_key_fields(self) -> Dict[str, Optional[str]]:
        fields = set(self.key_columns)
        for group_fields in self.gather_modes.values():
            fields.update(group_fields)
        return {field: self.h5_fallback.get(field) for field in sorted(fields)}


_H5_FALLBACK = {
    "shot_no": "shot_stake",
    "shot_stake": "shot_no",
    "recv_no": "recv_stake",
    "recv_stake": "recv_no",
}


SW06 = SegyProfile(
    name="sw06",
    description="Dongfang synthetic sw06 style headers",
    key_columns=("shot_line", "shot_no", "recv_line", "recv_no"),
    byte_pos={
        "shot_line": 221,
        "shot_no": 25,
        "recv_line": 61,
        "recv_no": 65,
        "shot_stake": 225,
        "recv_stake": 229,
        "shot_x": 73,
        "shot_y": 77,
        "rec_x": 81,
        "rec_y": 85,
        "cmp": 193,
        "cmp_line": 189,
        "offset": 37,
    },
    h5_fallback=_H5_FALLBACK,
    gather_modes={
        "survey_shot": ("recv_line", "shot_line", "shot_no"),
        "shot": ("shot_line", "shot_no"),
        "receiver": ("recv_line", "recv_no"),
        "survey_line": ("recv_line",),
    },
    sort_keys=("shot_line", "shot_no", "recv_line", "recv_no"),
    default_gather_mode="survey_shot",
)


FIELD1031 = SegyProfile(
    name="field1031",
    description="Field 1031 fixed headers",
    key_columns=("shot_line", "shot_stake", "recv_line", "recv_stake"),
    byte_pos={
        "shot_line": 17,
        "shot_no": 25,
        "recv_line": 61,
        "recv_stake": 65,
        "shot_x": 73,
        "shot_y": 77,
        "rec_x": 81,
        "rec_y": 85,
        "shot_stake": 21,
        "recv_no": 41,
        "cmp": 193,
        "cmp_line": 189,
        "offset": 37,
    },
    h5_fallback=_H5_FALLBACK,
    gather_modes={
        "survey_recv": ("shot_line", "recv_line", "recv_stake"),
        "survey_rec": ("shot_line", "recv_line", "recv_stake"),
        "survey_shot": ("recv_line", "shot_line", "shot_stake"),
        "shot": ("shot_line", "shot_stake"),
        "receiver": ("recv_line", "recv_stake"),
        "survey_line": ("recv_line",),
    },
    sort_keys=("recv_line", "recv_stake", "shot_line", "shot_stake"),
    default_gather_mode="survey_recv",
)


SEGC3 = SegyProfile(
    name="segc3",
    description="Self-computed mode: only coordinate byte positions; line/stake derived from scaled coords",
    key_columns=("shot_line", "shot_stake", "recv_line", "recv_stake"),
    byte_pos={
        "shot_x": 73,
        "shot_y": 77,
        "rec_x": 81,
        "rec_y": 85,
    },
    h5_fallback=_H5_FALLBACK,
    gather_modes={
        "survey_shot": ("recv_line", "shot_line", "shot_stake"),
        "shot": ("shot_line", "shot_stake"),
        "receiver": ("recv_line", "recv_stake"),
        "survey_line": ("recv_line",),
    },
    sort_keys=("recv_line", "recv_stake", "shot_line", "shot_stake"),
    default_gather_mode="survey_shot",
    default_header_mode="self_computed",
)


PROFILES = {
    SW06.name: SW06,
    FIELD1031.name: FIELD1031,
    SEGC3.name: SEGC3,
}
DEFAULT_PROFILE_NAME = SW06.name


def profile_names() -> Tuple[str, ...]:
    return tuple(sorted(PROFILES))


def get_segy_profile(name: Optional[str] = None) -> SegyProfile:
    key = (name or DEFAULT_PROFILE_NAME).strip().lower()
    try:
        return PROFILES[key]
    except KeyError as exc:
        valid = ", ".join(profile_names())
        raise ValueError(f"Unknown SEG-Y profile {name!r}; choose one of: {valid}") from exc


def parse_sort_keys(value: Optional[Union[Iterable[str], str]], profile: SegyProfile) -> Tuple[str, ...]:
    if value is None:
        return tuple(profile.sort_keys)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return tuple(profile.sort_keys)
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(value)


# ---------------------------------------------------------------------------
# Shared constants (profile-independent)
# ---------------------------------------------------------------------------

# Coordinate column mapping used by dataset code: {field_name: axis_index}
COORD_COL: Dict[str, int] = {"sx": 0, "sy": 1, "rx": 2, "ry": 3}

# Default trace sort order for patches
TRACE_SORT_KEYS: Tuple[str, ...] = ("rx", "ry", "sx", "sy")

# Number of coordinate dimensions
N_COORD_DIMS: int = 4

# H5 dataset keys written by convert_tool
DATASET_KEYS_FIXED = [
    "data", "sx", "sy", "rx", "ry",
    "delta", "t0",
    "shot_line", "shot_no", "recv_line", "recv_no",
    "shot_stake", "recv_stake", "cmp", "cmp_line", "offset",
    "trace_idx",
]

# Default metric weights for 4-axis spatial distance (sampler)
METRIC_WEIGHTS = [1.0, 1.0, 0.5, 0.5]


def print_profile_info(profile: Optional[SegyProfile] = None) -> None:
    """Debug helper: print profile state."""
    if profile is None:
        profile = get_segy_profile()
    print(f"[segy_schema] profile: {profile.name!r}")
    print(f"[segy_schema] description: {profile.description}")
    print(f"[segy_schema] key_columns: {profile.key_columns}")
    print(f"[segy_schema] byte_pos: {dict(profile.byte_pos)}")
    print(f"[segy_schema] sort_keys: {profile.sort_keys}")
    print(f"[segy_schema] gather_modes: {dict(profile.gather_modes)}")
    print(f"[segy_schema] available: {profile_names()}")
