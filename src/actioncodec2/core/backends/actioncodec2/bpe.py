'Strict ``bpe.json`` I/O for both tokenization schemes.'

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import TypeVar

import numpy as np

from ._validation import active_dimensions
from .bpe_vocabulary import BPEVocabulary
from .configuration import ActionCodec2Config, ConfigError, _integer
from .native_serialization import NativeCacheMixin, decode_native, encode_native
from .packing import EncodingSearch
from .setbpe.spatial import SpatialVocabulary
from .setbpe.temporal import TemporalVocabulary

BPE_NAME = "bpe.json"
SerializerT = TypeVar("SerializerT", bound="_Serializer")


def _rules(payload: object, width: int, path: str) -> tuple[tuple[int, ...], ...]:
    """Parse an ordered merge list without accepting lossy coercions."""
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ConfigError(f"{path} must be a list")
    output = []
    for rank, raw_rule in enumerate(payload):
        rule_path = f"{path}[{rank}]"
        if not isinstance(raw_rule, Sequence) or isinstance(raw_rule, (str, bytes, bytearray)):
            raise ConfigError(f"{rule_path} must be a list of {width} integers")
        if len(raw_rule) != width:
            raise ConfigError(f"{rule_path} must contain exactly {width} integers")
        output.append(
            tuple(_integer(value, f"{rule_path}[{index}]") for index, value in enumerate(raw_rule))
        )
    return tuple(output)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ConfigError(f"{BPE_NAME} contains duplicate field {key!r}")
        output[key] = value
    return output


def _validate_rule_references(
    *,
    atom_count: int,
    lookahead: int,
    spatial_merges: tuple[tuple[int, int], ...],
    temporal_merges: tuple[tuple[int, int, int], ...],
) -> None:
    """Give artifact-level errors before constructing vocabulary metadata."""
    for rank, (left, right) in enumerate(spatial_merges):
        child = atom_count + rank
        if left >= child or right >= child:
            raise ConfigError(f"spatial_merges[{rank}] must reference already-created tokens")
    spatial_size = atom_count + len(spatial_merges)
    for rank, (left, right, offset) in enumerate(temporal_merges):
        child = spatial_size + rank
        if left >= child or right >= child:
            raise ConfigError(f"temporal_merges[{rank}] must reference already-created tokens")
        if not 1 <= offset < lookahead:
            raise ConfigError(
                f"temporal_merges[{rank}] offset must be positive and smaller than lookahead"
            )


def _read_payload(directory: str | Path) -> dict[str, object]:
    """Parse ``bpe.json`` from an artifact directory, rejecting duplicate keys."""
    path = Path(directory) / BPE_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as error:
        raise ConfigError(f"cannot parse {path}: {error}") from error


def _dense_cells(
    cells: np.ndarray,
    bins: Sequence[int],
    active_dims: Sequence[int] | None,
    *,
    ndim: int = 2,
) -> tuple[np.ndarray, tuple[int, ...] | None]:
    'Validate scalar/batched trajectories and expand their final dimension.'
    raw = np.asarray(cells)
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError("cells must contain integer indices")
    values = np.asarray(raw, dtype=np.int64)
    if values.ndim != ndim or min(values.shape) < 1:
        raise ValueError("cells must have non-empty shape [T, D_active]")
    if active_dims is None:
        if values.shape[-1] > len(bins):
            raise ValueError(f"cells width exceeds codebook dimension {len(bins)}")
        if values.shape[-1] < len(bins):
            dimensions = tuple(range(values.shape[-1]))
            dense = np.full((*values.shape[:-1], len(bins)), -1, dtype=np.int64)
            dense[..., : values.shape[-1]] = values
            values = dense
        else:
            dimensions = None
    else:
        dimensions = active_dimensions(active_dims, len(bins))
        if len(dimensions) != values.shape[-1]:
            raise ValueError("active_dims must have one entry per compact cell coordinate")
        if np.any(values < 0):
            raise ValueError("compact cells must not contain -1")
        dense = np.full((*values.shape[:-1], len(bins)), -1, dtype=np.int64)
        dense[..., dimensions] = values
        values = dense
    counts = np.asarray(tuple(int(value) for value in bins), dtype=np.int64)
    if np.any(values < -1) or np.any(values >= counts[None, :]):
        raise ValueError("cells contain an invalid active or absent index")
    return values, dimensions


class _Serializer(NativeCacheMixin):
    """Common cell validation, inference dispatch, and strict artifact I/O."""

    _merge_fields: tuple[str, ...]

    def _init_dimensions(self, bins: Sequence[int], lookahead: int) -> None:
        self.encoding_search = EncodingSearch()
        self.lookahead = _integer(lookahead, "lookahead", positive=True)
        self.bins = tuple(
            _integer(value, f"bins[{index}]", positive=True) for index, value in enumerate(bins)
        )
        if not self.bins:
            raise ConfigError("bins must contain at least one cell slot")

    def configure_encoding(
        self,
        *,
        spatial_search: bool = False,
        temporal_search: bool = False,
        search_window: int = 32,
        search_beam: int = 4,
    ):
        'Select independent runtime packing stages; both default to disabled.'
        options = EncodingSearch(spatial_search, temporal_search, search_window, search_beam)
        options.validate_codec(self.scheme, self.bins)
        self.encoding_search = options
        return self

    def encode_batch(self, cells: np.ndarray, *, active_dims=None) -> list[list[int]]:
        values = np.asarray(cells)
        if values.ndim != 3 or min(values.shape) < 1:
            raise ValueError("cells must have non-empty shape [B,T,D]")
        dense, _ = _dense_cells(values, self.bins, active_dims, ndim=3)
        result = encode_native(self, dense)
        return (
            result
            if result is not None
            else [self.encode(row, active_dims=active_dims) for row in values]
        )

    def decode_batch(self, streams, *, active_dims=None) -> np.ndarray:
        dims = active_dimensions(active_dims, len(self.bins))
        result = decode_native(self, streams, dims, validate=True)
        if result is not None:
            return result
        rows = []
        for stream in streams:
            row = []
            for token in stream:
                if isinstance(token, bool) or not isinstance(token, Integral):
                    raise ValueError("tokens must be integers")
                if not 0 <= int(token) < self.vocab_size:
                    raise ValueError("token is outside the vocabulary")
                row.append(int(token))
            if not row:
                raise ValueError("token stream must not be empty")
            rows.append(row)
        if not rows:
            raise ValueError("streams must not be empty")
        result = decode_native(self, rows, dims)
        if result is not None:
            return result
        cells = [self.decode(row, active_dims=active_dims) for row in rows]
        if len({len(row) for row in cells}) != 1:
            raise ValueError("all token streams must decode to the same length")
        return np.stack(cells)

    def encode(
        self,
        cells: np.ndarray,
        *,
        active_dims: Sequence[int] | None = None,
    ) -> list[int]:
        'Serialize one non-empty trajectory without emitting absent slots.'
        values, _ = _dense_cells(cells, self.bins, active_dims)
        result = encode_native(self, values[None])
        if result is not None:
            return result[0]
        return self.vocabulary.encode(values)

    def decode(
        self,
        tokens: Sequence[int],
        *,
        active_dims: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Recover dense cells, writing ``-1`` in semantically absent slots."""
        if len(tokens) < 1:
            raise ValueError("token stream must not be empty")
        checked = []
        for index, token in enumerate(tokens):
            if isinstance(token, bool) or not isinstance(token, Integral):
                raise ValueError(f"tokens[{index}] must be an integer")
            value = int(token)
            if not 0 <= value < self.vocab_size:
                raise ValueError(f"tokens[{index}] is outside the vocabulary")
            checked.append(value)
        result = decode_native(self, [checked], active_dimensions(active_dims, len(self.bins)))
        return (
            result[0]
            if result is not None
            else self.vocabulary.decode(checked, active_dims=active_dims)
        )

    def to_dict(self) -> dict[str, object]:
        """Return only the merge lists; physical metadata lives in config.yaml."""
        return {
            field: [list(rule) for rule in getattr(self, field)] for field in self._merge_fields
        }

    @classmethod
    def from_dict(
        cls: type[SerializerT], payload: object, *, bins: Sequence[int], lookahead: int
    ) -> SerializerT:
        """Validate the scheme's exact artifact fields before constructing it."""
        if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
            raise ConfigError(f"{BPE_NAME} must contain one JSON object")
        allowed = set(cls._merge_fields)
        for fields, message in (
            (set(payload) - allowed, "contains unknown field(s)"),
            (allowed - set(payload), "is missing field(s)"),
        ):
            if fields:
                raise ConfigError(f"{BPE_NAME} {message}: {', '.join(sorted(fields))}")
        return cls(bins=bins, lookahead=lookahead, **payload)

    @classmethod
    def from_pretrained(
        cls: type[SerializerT],
        directory: str | Path,
        *,
        bins: Sequence[int],
        lookahead: int,
    ) -> SerializerT:
        return cls.from_dict(_read_payload(directory), bins=bins, lookahead=lookahead)

    def save_pretrained(self, directory: str | Path) -> Path:
        """Write ``bpe.json`` and return its path."""
        path = Path(directory) / BPE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


class SetBPESerializer(_Serializer):
    'Lossless mapping between ``[T,D]`` physical cells and token IDs.'

    scheme = "set_bpe"
    _merge_fields = ("spatial_merges", "temporal_merges")

    def __init__(
        self,
        *,
        bins: Sequence[int],
        lookahead: int,
        spatial_merges: Sequence[Sequence[int]],
        temporal_merges: Sequence[Sequence[int]],
    ) -> None:
        self._init_dimensions(bins, lookahead)
        self.spatial_merges = _rules(spatial_merges, 2, "spatial_merges")
        self.temporal_merges = _rules(temporal_merges, 3, "temporal_merges")
        _validate_rule_references(
            atom_count=sum(self.bins),
            lookahead=self.lookahead,
            spatial_merges=self.spatial_merges,
            temporal_merges=self.temporal_merges,
        )
        try:
            self.spatial = SpatialVocabulary(self.bins, self.spatial_merges)
            self.vocabulary = TemporalVocabulary(self.spatial, self.lookahead, self.temporal_merges)
        except ValueError as error:
            raise ConfigError(str(error)) from error
        self.patterns = self.vocabulary.patterns
        self.root = self.vocabulary.root
        self.vocab_size = self.vocabulary.vocab_size
        self.token_cell_counts = np.asarray(
            [
                sum(
                    int(self.spatial.support[spatial_token]).bit_count()
                    for _, spatial_token in self.patterns[token]
                )
                for token in range(self.vocab_size)
            ],
            dtype=np.int64,
        )


class BPESerializer(_Serializer):
    'Lossless mapping between ``[T,D]`` physical cells and plain-BPE ids.'

    scheme = "bpe"
    _merge_fields = ("merges",)

    def __init__(
        self,
        *,
        bins: Sequence[int],
        lookahead: int,
        merges: Sequence[Sequence[int]],
    ) -> None:
        self._init_dimensions(bins, lookahead)
        self.merges = _rules(merges, 2, "merges")
        atom_count = sum(self.bins)
        for rank, (left, right) in enumerate(self.merges):
            child = atom_count + rank
            if left >= child or right >= child:
                raise ConfigError(f"merges[{rank}] must reference already-created tokens")
        try:
            self.vocabulary = BPEVocabulary(self.bins, self.merges)
        except ValueError as error:
            raise ConfigError(str(error)) from error
        self.max_token_cells = self.lookahead * len(self.bins)
        self.token_cell_counts = self.vocabulary.token_cells
        oversized = int(np.count_nonzero(self.token_cell_counts > self.max_token_cells))
        if oversized:
            raise ConfigError(
                f"{oversized} merge rule(s) cover more than the {self.max_token_cells} "
                "primitive cells allowed by lookahead"
            )
        self.vocab_size = self.vocabulary.vocab_size


SERIALIZERS: dict[str, type[SetBPESerializer] | type[BPESerializer]] = {
    SetBPESerializer.scheme: SetBPESerializer,
    BPESerializer.scheme: BPESerializer,
}


def serializer_class(config: ActionCodec2Config) -> type[SetBPESerializer] | type[BPESerializer]:
    """Return the serializer the configured tokenization scheme requires."""
    scheme = config.set_bpe.scheme
    if scheme == "unified_set_bpe":
        from .unified_set_bpe import UnifiedSetBPESerializer

        return UnifiedSetBPESerializer
    if scheme not in SERIALIZERS:
        raise ConfigError(
            f"set_bpe.scheme={scheme!r} is not an ActionCodec2 serializer; "
            "load sparse_set_bpe artifacts with ActionCodec2Sparse"
        )
    return SERIALIZERS[scheme]
