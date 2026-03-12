from __future__ import annotations

import datetime
import json
import math
import shutil
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf

from app.config import TRAINED_MODELS_DIR, resolve_data_root, resolve_runtime_path, runtime_workdir, workspace_paths
from app.services.training_preset_service import inspect_config_architecture


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _safe_name(value: str, fallback: str = "item") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (value or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


class LibraryService:
    def __init__(self) -> None:
        self._write_lock = threading.RLock()
        self.reconfigure()

    def reconfigure(self) -> None:
        paths = workspace_paths()
        self.data_root = paths["data_root"]
        self.db_path = paths["db_path"]
        self.datasets_root = paths["datasets_root"]
        self.dataset_files_root = paths["dataset_files_root"]
        self.dataset_segments_root = paths["dataset_segments_root"]
        self.models_root = paths["models_root"]
        self.jobs_root = paths["jobs_root"]
        self.dataset_versions_root = self.data_root / "datasets" / "versions"
        for path in (
            self.data_root,
            self.db_path.parent,
            self.datasets_root,
            self.dataset_files_root,
            self.dataset_segments_root,
            self.dataset_versions_root,
            self.models_root,
            self.jobs_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._migrate_legacy_models()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dataset_files (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    duration_seconds REAL NOT NULL DEFAULT 0,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    sample_rate INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS dataset_segments (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    dataset_file_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    start_ms INTEGER NOT NULL DEFAULT 0,
                    end_ms INTEGER NOT NULL DEFAULT 0,
                    duration_seconds REAL NOT NULL DEFAULT 0,
                    energy_db REAL NOT NULL DEFAULT 0,
                    sample_rate INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    source_kind TEXT NOT NULL DEFAULT 'auto',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
                    FOREIGN KEY(dataset_file_id) REFERENCES dataset_files(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    segment_count INTEGER NOT NULL DEFAULT 0,
                    total_duration REAL NOT NULL DEFAULT 0,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS dataset_version_segments (
                    id TEXT PRIMARY KEY,
                    dataset_version_id TEXT NOT NULL,
                    dataset_file_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    start_ms INTEGER NOT NULL DEFAULT 0,
                    end_ms INTEGER NOT NULL DEFAULT 0,
                    duration_seconds REAL NOT NULL DEFAULT 0,
                    energy_db REAL NOT NULL DEFAULT 0,
                    sample_rate INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(dataset_version_id) REFERENCES dataset_versions(id) ON DELETE CASCADE,
                    FOREIGN KEY(dataset_file_id) REFERENCES dataset_files(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'training',
                    default_version_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_models_name_speaker
                ON models(name, speaker);

                CREATE TABLE IF NOT EXISTS model_versions (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    dataset_version_id TEXT,
                    training_mode TEXT NOT NULL DEFAULT 'new',
                    f0_predictor TEXT NOT NULL DEFAULT 'rmvpe',
                    device_preference TEXT NOT NULL DEFAULT 'auto',
                    device_used TEXT,
                    config_path TEXT NOT NULL,
                    inference_checkpoint_path TEXT NOT NULL,
                    step_count INTEGER NOT NULL DEFAULT 0,
                    loss_disc REAL,
                    loss_gen REAL,
                    parent_version_id TEXT,
                    parent_checkpoint_id TEXT,
                    source TEXT NOT NULL DEFAULT 'training',
                    legacy_checkpoint_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(model_id) REFERENCES models(id) ON DELETE CASCADE,
                    FOREIGN KEY(dataset_version_id) REFERENCES dataset_versions(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    model_version_id TEXT NOT NULL,
                    step INTEGER NOT NULL DEFAULT 0,
                    generator_path TEXT NOT NULL,
                    discriminator_path TEXT,
                    kind TEXT NOT NULL DEFAULT 'auto',
                    is_final INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(model_version_id) REFERENCES model_versions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL UNIQUE,
                    dataset_version_id TEXT,
                    model_id TEXT,
                    model_version_id TEXT,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    progress REAL NOT NULL DEFAULT 0,
                    device_used TEXT,
                    summary_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS inference_runs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL UNIQUE,
                    model_version_id TEXT,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    progress REAL NOT NULL DEFAULT 0,
                    device_used TEXT,
                    summary_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(connection, "model_versions", "main_preset_id", "TEXT")
            self._ensure_column(connection, "model_versions", "speech_encoder", "TEXT")
            self._ensure_column(connection, "model_versions", "use_tiny", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(connection, "model_versions", "ssl_dim", "INTEGER")
            self._ensure_column(connection, "model_versions", "gin_channels", "INTEGER")
            self._ensure_column(connection, "model_versions", "filter_channels", "INTEGER")
            self._ensure_column(connection, "model_versions", "diffusion_status", "TEXT NOT NULL DEFAULT 'not_trained'")
            self._ensure_column(connection, "model_versions", "diffusion_model_path", "TEXT")
            self._ensure_column(connection, "model_versions", "diffusion_config_path", "TEXT")
            self._ensure_column(connection, "model_versions", "diffusion_step_count", "INTEGER")
            self._ensure_column(connection, "model_versions", "diffusion_updated_at", "TEXT")
            self._ensure_column(connection, "model_versions", "diffusion_params_json", "TEXT")

    def _table_columns(self, connection: sqlite3.Connection, table_name: str) -> set[str]:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row["name"]) for row in rows}

    def _ensure_column(self, connection: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
        if column_name in self._table_columns(connection, table_name):
            return
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def _load_config_architecture_metadata(self, config_value: Optional[str]) -> Dict[str, Any]:
        if not config_value:
            return {
                "main_preset_id": None,
                "speech_encoder": "vec768l12",
                "use_tiny": False,
                "ssl_dim": 768,
                "gin_channels": 768,
                "filter_channels": 768,
            }
        config_path = self.absolute_path(config_value) if not Path(config_value).is_absolute() else Path(config_value)
        if not config_path.exists():
            return {
                "main_preset_id": None,
                "speech_encoder": "vec768l12",
                "use_tiny": False,
                "ssl_dim": 768,
                "gin_channels": 768,
                "filter_channels": 768,
            }
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config_data = json.load(handle)
        except Exception:
            return {
                "main_preset_id": None,
                "speech_encoder": "vec768l12",
                "use_tiny": False,
                "ssl_dim": 768,
                "gin_channels": 768,
                "filter_channels": 768,
            }
        return inspect_config_architecture(config_data)

    def _enrich_model_version(self, version: Dict[str, Any]) -> Dict[str, Any]:
        architecture = self._load_config_architecture_metadata(version.get("config_path"))
        version["main_preset_id"] = version.get("main_preset_id") or architecture["main_preset_id"]
        version["speech_encoder"] = version.get("speech_encoder") or architecture["speech_encoder"]
        use_tiny_value = version.get("use_tiny")
        if use_tiny_value in (None, ""):
            version["use_tiny"] = bool(architecture["use_tiny"])
        else:
            version["use_tiny"] = bool(use_tiny_value)
        version["ssl_dim"] = int(version.get("ssl_dim") or architecture["ssl_dim"])
        version["gin_channels"] = int(version.get("gin_channels") or architecture["gin_channels"])
        version["filter_channels"] = int(version.get("filter_channels") or architecture["filter_channels"])
        status = str(version.get("diffusion_status") or "").strip() or "not_trained"
        if status == "trained" and version.get("diffusion_model_path"):
            diffusion_candidate = self.absolute_path(version["diffusion_model_path"])
            if not diffusion_candidate.exists():
                status = "not_trained"
        version["diffusion_status"] = status
        version["has_diffusion"] = bool(status == "trained" and version.get("diffusion_model_path"))
        if version.get("diffusion_params_json"):
            try:
                version["diffusion_params"] = json.loads(version["diffusion_params_json"])
            except json.JSONDecodeError:
                version["diffusion_params"] = None
        else:
            version["diffusion_params"] = None
        return version

    def _relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.data_root).as_posix()

    def absolute_path(self, value: str) -> Path:
        return (self.data_root / value).resolve()

    def _fetch_all(self, query: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def _fetch_one(self, query: str, params: Iterable[Any] = ()) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(query, tuple(params)).fetchone()
        return dict(row) if row else None

    def _execute(self, query: str, params: Iterable[Any] = ()) -> None:
        with self._write_lock:
            with self._connect() as connection:
                connection.execute(query, tuple(params))
                connection.commit()

    def _executemany(self, query: str, params: Iterable[Iterable[Any]]) -> None:
        with self._write_lock:
            with self._connect() as connection:
                connection.executemany(query, [tuple(item) for item in params])
                connection.commit()

    def _copy_file(self, source_path: Path, target_dir: Path, target_name: Optional[str] = None) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / (target_name or source_path.name)
        shutil.copy2(source_path, destination)
        return destination

    def _get_dataset_row(self, dataset_id: str) -> Dict[str, Any]:
        dataset = self._fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        if not dataset:
            raise FileNotFoundError(f"数据集 {dataset_id} 不存在。")
        return dataset

    def get_dataset_file(self, file_id: str) -> Dict[str, Any]:
        item = self._fetch_one("SELECT * FROM dataset_files WHERE id = ?", (file_id,))
        if not item:
            raise FileNotFoundError(f"音频文件 {file_id} 不存在。")
        return item

    def _get_segment_row(self, segment_id: str) -> Dict[str, Any]:
        segment = self._fetch_one("SELECT * FROM dataset_segments WHERE id = ?", (segment_id,))
        if not segment:
            raise FileNotFoundError(f"音频片段 {segment_id} 不存在。")
        return segment

    def get_segment(self, segment_id: str) -> Dict[str, Any]:
        return self._get_segment_row(segment_id)

    def get_dataset_segment(self, segment_id: str) -> Dict[str, Any]:
        return self._get_segment_row(segment_id)

    def get_dataset_file(self, file_id: str) -> Dict[str, Any]:
        file_row = self._fetch_one("SELECT * FROM dataset_files WHERE id = ?", (file_id,))
        if not file_row:
            raise FileNotFoundError(f"音频文件 {file_id} 不存在。")
        return file_row

    def _get_dataset_version_row(self, version_id: str) -> Dict[str, Any]:
        version = self._fetch_one("SELECT * FROM dataset_versions WHERE id = ?", (version_id,))
        if not version:
            raise FileNotFoundError(f"数据集版本 {version_id} 不存在。")
        return version

    def _get_model_row(self, model_id: str) -> Dict[str, Any]:
        model = self._fetch_one("SELECT * FROM models WHERE id = ?", (model_id,))
        if not model:
            raise FileNotFoundError(f"模型 {model_id} 不存在。")
        return model

    def get_model_version(self, version_id: str) -> Dict[str, Any]:
        version = self._fetch_one("SELECT * FROM model_versions WHERE id = ?", (version_id,))
        if not version:
            raise FileNotFoundError(f"模型版本 {version_id} 不存在。")
        return self._enrich_model_version(version)

    def get_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        checkpoint = self._fetch_one("SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,))
        if not checkpoint:
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} 不存在。")
        return checkpoint

    def create_dataset(self, *, name: str, speaker: str, description: str = "") -> Dict[str, Any]:
        dataset_id = uuid.uuid4().hex
        now = _now()
        safe_name = name.strip() or speaker.strip() or "dataset"
        speaker_name = speaker.strip() or "speaker"
        self._execute(
            """
            INSERT INTO datasets (id, name, speaker, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, safe_name, speaker_name, description.strip(), now, now),
        )
        return self.get_dataset(dataset_id)

    def list_dataset_files(self, dataset_id: str) -> List[Dict[str, Any]]:
        return self._fetch_all(
            """
            SELECT * FROM dataset_files
            WHERE dataset_id = ?
            ORDER BY created_at ASC
            """,
            (dataset_id,),
        )

    def list_dataset_segments(self, dataset_id: str) -> List[Dict[str, Any]]:
        return self._fetch_all(
            """
            SELECT * FROM dataset_segments
            WHERE dataset_id = ?
            ORDER BY dataset_file_id ASC, start_ms ASC, display_name ASC
            """,
            (dataset_id,),
        )

    def list_dataset_versions(self, dataset_id: str) -> List[Dict[str, Any]]:
        return self._fetch_all(
            """
            SELECT * FROM dataset_versions
            WHERE dataset_id = ?
            ORDER BY created_at DESC
            """,
            (dataset_id,),
        )

    def list_datasets(self) -> List[Dict[str, Any]]:
        datasets = self._fetch_all(
            """
            SELECT
                d.*,
                COALESCE((SELECT COUNT(*) FROM dataset_files f WHERE f.dataset_id = d.id), 0) AS file_count,
                COALESCE((SELECT COUNT(*) FROM dataset_segments s WHERE s.dataset_id = d.id), 0) AS segment_count,
                COALESCE((SELECT COUNT(*) FROM dataset_segments s WHERE s.dataset_id = d.id AND s.enabled = 1), 0) AS enabled_segment_count,
                COALESCE((SELECT COUNT(*) FROM dataset_versions v WHERE v.dataset_id = d.id), 0) AS version_count,
                (SELECT status FROM training_runs r
                 JOIN dataset_versions v ON v.id = r.dataset_version_id
                 WHERE v.dataset_id = d.id
                 ORDER BY r.created_at DESC LIMIT 1) AS last_training_status
            FROM datasets d
            ORDER BY d.updated_at DESC, d.created_at DESC
            """
        )
        for item in datasets:
            item["files"] = self.list_dataset_files(item["id"])
            item["versions"] = self.list_dataset_versions(item["id"])
        return datasets

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        dataset = self._get_dataset_row(dataset_id)
        dataset["files"] = self.list_dataset_files(dataset_id)
        dataset["segments"] = self.list_dataset_segments(dataset_id)
        dataset["versions"] = self.list_dataset_versions(dataset_id)
        dataset["segment_count"] = len(dataset["segments"])
        dataset["enabled_segment_count"] = sum(1 for item in dataset["segments"] if item["enabled"])
        dataset["file_count"] = len(dataset["files"])
        return dataset

    def add_dataset_file(self, *, dataset_id: str, source_path: Path, original_name: str) -> Dict[str, Any]:
        self._get_dataset_row(dataset_id)
        file_id = uuid.uuid4().hex
        safe_name = Path(original_name).name
        file_dir = self.dataset_files_root / dataset_id
        target_path = self._copy_file(source_path, file_dir, f"{file_id}_{safe_name}")
        info = sf.info(str(target_path))
        now = _now()
        record = {
            "id": file_id,
            "dataset_id": dataset_id,
            "original_name": safe_name,
            "storage_path": self._relative_path(target_path),
            "duration_seconds": round(float(info.duration), 4),
            "size_bytes": int(target_path.stat().st_size),
            "sample_rate": int(info.samplerate),
            "created_at": now,
        }
        self._execute(
            """
            INSERT INTO dataset_files (id, dataset_id, original_name, storage_path, duration_seconds, size_bytes, sample_rate, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["dataset_id"],
                record["original_name"],
                record["storage_path"],
                record["duration_seconds"],
                record["size_bytes"],
                record["sample_rate"],
                record["created_at"],
            ),
        )
        self._execute("UPDATE datasets SET updated_at = ? WHERE id = ?", (now, dataset_id))
        return record

    def _clear_dataset_segments(self, dataset_id: str) -> None:
        for segment in self.list_dataset_segments(dataset_id):
            self.absolute_path(segment["storage_path"]).unlink(missing_ok=True)
        self._execute("DELETE FROM dataset_segments WHERE dataset_id = ?", (dataset_id,))

    def _load_mono_audio(self, path: Path) -> tuple[np.ndarray, int]:
        audio, sample_rate = sf.read(str(path), always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)

    def _voice_regions(self, chunks: Dict[str, Dict[str, Any]], sample_rate: int, merge_gap_ms: int) -> List[tuple[int, int]]:
        merge_gap_samples = int(sample_rate * merge_gap_ms / 1000.0)
        ordered = [chunks[key] for key in sorted(chunks.keys(), key=int)]
        ranges = []
        index = 0
        while index < len(ordered):
            current = ordered[index]
            start, end = (int(value) for value in current["split_time"].split(","))
            if current["slice"]:
                index += 1
                continue
            region_start = start
            region_end = end
            cursor = index + 1
            while cursor + 1 < len(ordered):
                silence = ordered[cursor]
                next_voice = ordered[cursor + 1]
                silence_start, silence_end = (int(value) for value in silence["split_time"].split(","))
                next_start, next_end = (int(value) for value in next_voice["split_time"].split(","))
                if silence["slice"] and not next_voice["slice"] and (silence_end - silence_start) <= merge_gap_samples:
                    region_end = next_end
                    cursor += 2
                    continue
                break
            ranges.append((region_start, region_end))
            index = cursor
        return ranges

    def _split_region(self, start_sample: int, end_sample: int, sample_rate: int, max_seconds: float) -> List[tuple[int, int]]:
        max_samples = int(sample_rate * max_seconds)
        result = []
        cursor = start_sample
        while cursor < end_sample:
            next_end = min(cursor + max_samples, end_sample)
            result.append((cursor, next_end))
            cursor = next_end
        return result

    def segmentize_dataset(
        self,
        dataset_id: str,
        *,
        min_keep_seconds: float = 1.5,
        max_segment_seconds: float = 6.0,
        merge_gap_ms: int = 300,
        energy_floor_db: float = -45.0,
    ) -> Dict[str, Any]:
        self._get_dataset_row(dataset_id)
        files = self.list_dataset_files(dataset_id)
        if not files:
            raise RuntimeError("数据集里还没有可分段的音频。")

        self._clear_dataset_segments(dataset_id)
        created_rows = []
        now = _now()
        with runtime_workdir():
            from inference import slicer

            for file_row in files:
                file_path = self.absolute_path(file_row["storage_path"])
                audio, sample_rate = self._load_mono_audio(file_path)
                chunks = slicer.cut(str(file_path), db_thresh=-40, min_len=1800)
                voice_ranges = self._voice_regions(chunks, sample_rate, merge_gap_ms=merge_gap_ms)
                segment_index = 0
                for region_start, region_end in voice_ranges:
                    for chunk_start, chunk_end in self._split_region(region_start, region_end, sample_rate, max_segment_seconds):
                        clip = audio[chunk_start:chunk_end]
                        duration_seconds = float((chunk_end - chunk_start) / sample_rate) if sample_rate else 0.0
                        if duration_seconds < min_keep_seconds:
                            continue
                        rms = float(np.sqrt(np.mean(np.square(clip)))) if clip.size else 0.0
                        energy_db = 20.0 * math.log10(max(rms, 1e-8))
                        if energy_db < energy_floor_db:
                            continue
                        segment_id = uuid.uuid4().hex
                        segment_index += 1
                        display_name = f"{Path(file_row['original_name']).stem}_seg{segment_index:03d}.wav"
                        segment_dir = self.dataset_segments_root / dataset_id / file_row["id"]
                        segment_path = segment_dir / f"{segment_id}_{display_name}"
                        segment_dir.mkdir(parents=True, exist_ok=True)
                        sf.write(str(segment_path), clip, sample_rate)
                        created_rows.append(
                            (
                                segment_id,
                                dataset_id,
                                file_row["id"],
                                display_name,
                                self._relative_path(segment_path),
                                int(round(chunk_start * 1000 / sample_rate)),
                                int(round(chunk_end * 1000 / sample_rate)),
                                round(duration_seconds, 4),
                                round(energy_db, 2),
                                sample_rate,
                                1,
                                "auto",
                                now,
                                now,
                            )
                        )

        if created_rows:
            self._executemany(
                """
                INSERT INTO dataset_segments (
                    id, dataset_id, dataset_file_id, display_name, storage_path, start_ms, end_ms,
                    duration_seconds, energy_db, sample_rate, enabled, source_kind, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                created_rows,
            )

        self._execute("UPDATE datasets SET updated_at = ? WHERE id = ?", (_now(), dataset_id))
        return {
            "dataset_id": dataset_id,
            "generated_segments": len(created_rows),
            "enabled_segments": len(created_rows),
            "segments": self.list_dataset_segments(dataset_id),
        }

    def set_segment_enabled(self, segment_id: str, enabled: bool) -> Dict[str, Any]:
        segment = self._get_segment_row(segment_id)
        now = _now()
        self._execute(
            "UPDATE dataset_segments SET enabled = ?, updated_at = ? WHERE id = ?",
            (1 if enabled else 0, now, segment_id),
        )
        self._execute("UPDATE datasets SET updated_at = ? WHERE id = ?", (now, segment["dataset_id"]))
        return self._get_segment_row(segment_id)

    def list_dataset_version_segments(self, version_id: str) -> List[Dict[str, Any]]:
        return self._fetch_all(
            """
            SELECT * FROM dataset_version_segments
            WHERE dataset_version_id = ?
            ORDER BY created_at ASC, display_name ASC
            """,
            (version_id,),
        )

    def get_dataset_version_segment(self, segment_id: str) -> Dict[str, Any]:
        segment = self._fetch_one("SELECT * FROM dataset_version_segments WHERE id = ?", (segment_id,))
        if not segment:
            raise FileNotFoundError(f"数据集版本片段 {segment_id} 不存在。")
        return segment

    def create_dataset_version(self, dataset_id: str, *, label: str = "", notes: str = "") -> Dict[str, Any]:
        dataset = self._get_dataset_row(dataset_id)
        segments = [item for item in self.list_dataset_segments(dataset_id) if item["enabled"]]
        if len(segments) < 3:
            raise RuntimeError("当前启用的训练片段不足 3 条，请先审核或补充音频。")

        version_id = uuid.uuid4().hex
        now = _now()
        version_label = label.strip() or f"v{len(self.list_dataset_versions(dataset_id)) + 1}"
        version_dir = self.dataset_versions_root / dataset_id / version_id / "segments"
        version_dir.mkdir(parents=True, exist_ok=True)
        total_duration = 0.0
        version_rows = []
        for index, segment in enumerate(segments, start=1):
            source_path = self.absolute_path(segment["storage_path"])
            target_path = self._copy_file(source_path, version_dir, f"{index:04d}_{segment['display_name']}")
            total_duration += float(segment["duration_seconds"])
            version_rows.append(
                (
                    uuid.uuid4().hex,
                    version_id,
                    segment["dataset_file_id"],
                    segment["display_name"],
                    self._relative_path(target_path),
                    segment["start_ms"],
                    segment["end_ms"],
                    segment["duration_seconds"],
                    segment["energy_db"],
                    segment["sample_rate"],
                    now,
                )
            )

        self._execute(
            """
            INSERT INTO dataset_versions (id, dataset_id, label, speaker, segment_count, total_duration, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                dataset_id,
                version_label,
                dataset["speaker"],
                len(version_rows),
                round(total_duration, 4),
                notes.strip(),
                now,
                now,
            ),
        )
        self._executemany(
            """
            INSERT INTO dataset_version_segments (
                id, dataset_version_id, dataset_file_id, display_name, storage_path, start_ms, end_ms,
                duration_seconds, energy_db, sample_rate, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            version_rows,
        )
        self._execute("UPDATE datasets SET updated_at = ? WHERE id = ?", (now, dataset_id))
        return self.get_dataset_version(version_id)

    def get_dataset_version(self, version_id: str) -> Dict[str, Any]:
        version = self._get_dataset_version_row(version_id)
        version["segments"] = self.list_dataset_version_segments(version_id)
        return version

    def _get_or_create_model(self, name: str, speaker: str, source: str) -> Dict[str, Any]:
        safe_name = _safe_name(name, "model")
        safe_speaker = speaker.strip() or "speaker"
        existing = self._fetch_one("SELECT * FROM models WHERE name = ? AND speaker = ?", (safe_name, safe_speaker))
        if existing:
            return existing
        model_id = uuid.uuid4().hex
        now = _now()
        self._execute(
            """
            INSERT INTO models (id, name, speaker, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (model_id, safe_name, safe_speaker, source, now, now),
        )
        return self._get_model_row(model_id)

    def list_checkpoints(self, model_version_id: str) -> List[Dict[str, Any]]:
        return self._fetch_all(
            """
            SELECT * FROM checkpoints
            WHERE model_version_id = ?
            ORDER BY step DESC, created_at DESC
            """,
            (model_version_id,),
        )

    def list_model_versions(self, model_id: str) -> List[Dict[str, Any]]:
        versions = self._fetch_all(
            """
            SELECT * FROM model_versions
            WHERE model_id = ?
            ORDER BY created_at DESC
            """,
            (model_id,),
        )
        for version in versions:
            self._enrich_model_version(version)
            version["checkpoints"] = self.list_checkpoints(version["id"])
        return versions

    def list_models(self) -> List[Dict[str, Any]]:
        models = self._fetch_all(
            """
            SELECT * FROM models
            ORDER BY updated_at DESC, created_at DESC
            """
        )
        for model in models:
            model["versions"] = self.list_model_versions(model["id"])
        return models

    def get_model(self, model_id: str) -> Dict[str, Any]:
        model = self._get_model_row(model_id)
        model["versions"] = self.list_model_versions(model_id)
        return model

    def _next_model_version_label(self, model_id: str) -> str:
        count_row = self._fetch_one("SELECT COUNT(*) AS count FROM model_versions WHERE model_id = ?", (model_id,))
        return f"v{int(count_row['count']) + 1 if count_row else 1}"

    def _register_model_version(
        self,
        *,
        model: Dict[str, Any],
        dataset_version_id: Optional[str],
        training_mode: str,
        f0_predictor: str,
        device_preference: str,
        device_used: Optional[str],
        config_source_path: Path,
        inference_checkpoint_source_path: Path,
        checkpoints: List[Dict[str, Any]],
        step_count: int,
        loss_disc: Optional[float],
        loss_gen: Optional[float],
        source: str,
        parent_version_id: Optional[str],
        parent_checkpoint_id: Optional[str],
        legacy_checkpoint_path: Optional[str] = None,
        label: str = "",
    ) -> Dict[str, Any]:
        version_id = uuid.uuid4().hex
        now = _now()
        version_label = label.strip() or self._next_model_version_label(model["id"])
        version_dir = self.models_root / model["id"] / version_id
        checkpoints_dir = version_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        config_target = self._copy_file(config_source_path, version_dir, "config.json")
        final_generator_target = self._copy_file(inference_checkpoint_source_path, checkpoints_dir, Path(inference_checkpoint_source_path).name)
        checkpoint_rows = []
        for item in checkpoints:
            checkpoint_id = uuid.uuid4().hex
            generator_target = final_generator_target
            discriminator_target_rel = None
            if Path(item["generator_path"]).resolve() != inference_checkpoint_source_path.resolve():
                generator_target = self._copy_file(Path(item["generator_path"]), checkpoints_dir, Path(item["generator_path"]).name)
            if item.get("discriminator_path"):
                discriminator_target = self._copy_file(Path(item["discriminator_path"]), checkpoints_dir, Path(item["discriminator_path"]).name)
                discriminator_target_rel = self._relative_path(discriminator_target)
            checkpoint_rows.append(
                (
                    checkpoint_id,
                    version_id,
                    int(item.get("step", 0)),
                    self._relative_path(generator_target),
                    discriminator_target_rel,
                    item.get("kind", "auto"),
                    1 if item.get("is_final") else 0,
                    now,
                )
            )

        self._execute(
            """
            INSERT INTO model_versions (
                id, model_id, label, dataset_version_id, training_mode, f0_predictor, device_preference, device_used,
                config_path, inference_checkpoint_path, step_count, loss_disc, loss_gen, parent_version_id,
                parent_checkpoint_id, source, legacy_checkpoint_path, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                model["id"],
                version_label,
                dataset_version_id,
                training_mode,
                f0_predictor,
                device_preference,
                device_used,
                self._relative_path(config_target),
                self._relative_path(final_generator_target),
                step_count,
                loss_disc,
                loss_gen,
                parent_version_id,
                parent_checkpoint_id,
                source,
                legacy_checkpoint_path,
                now,
                now,
            ),
        )
        if checkpoint_rows:
            self._executemany(
                """
                INSERT INTO checkpoints (
                    id, model_version_id, step, generator_path, discriminator_path, kind, is_final, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                checkpoint_rows,
            )
        self._execute("UPDATE models SET default_version_id = ?, updated_at = ? WHERE id = ?", (version_id, now, model["id"]))
        model_version = self.get_model_version(version_id)
        model_version["checkpoints"] = self.list_checkpoints(version_id)
        return model_version

    def import_model(
        self,
        *,
        checkpoint_path: Path,
        config_path: Path,
        label: str,
        speaker: str,
        source: str = "imported",
    ) -> Dict[str, Any]:
        model = self._get_or_create_model(label, speaker, source)
        version = self._register_model_version(
            model=model,
            dataset_version_id=None,
            training_mode="import",
            f0_predictor="rmvpe",
            device_preference="auto",
            device_used=None,
            config_source_path=config_path,
            inference_checkpoint_source_path=checkpoint_path,
            checkpoints=[{"step": 0, "generator_path": str(checkpoint_path), "discriminator_path": None, "kind": "final", "is_final": True}],
            step_count=0,
            loss_disc=None,
            loss_gen=None,
            source=source,
            parent_version_id=None,
            parent_checkpoint_id=None,
            label="导入版本",
        )
        return {
            "model_id": model["id"],
            "model_name": model["name"],
            "model_version_id": version["id"],
            "label": version["label"],
            "speaker": model["speaker"],
            "checkpoint_path": version["inference_checkpoint_path"],
            "config_path": version["config_path"],
            "source": source,
        }

    def get_inference_profiles(self) -> List[Dict[str, Any]]:
        versions = self._fetch_all(
            """
            SELECT
                mv.*,
                m.name AS model_name,
                m.speaker AS model_speaker,
                m.default_version_id AS model_default_version_id
            FROM model_versions mv
            JOIN models m ON m.id = mv.model_id
            ORDER BY mv.created_at DESC
            """
        )
        profiles = []
        for version in versions:
            self._enrich_model_version(version)
            config_path = self.absolute_path(version["config_path"])
            speakers = []
            try:
                with config_path.open("r", encoding="utf-8") as handle:
                    config_data = json.load(handle)
                speakers = list((config_data.get("spk") or {}).keys())
            except Exception:
                speakers = [version["model_speaker"]] if version.get("model_speaker") else []
            profiles.append(
                {
                    "id": version["id"],
                    "label": f"{version['model_name']} / {version['label']}",
                    "name": version["model_name"],
                    "checkpoint_path": version["inference_checkpoint_path"],
                    "config_path": version["config_path"],
                    "speakers": speakers or ([version["model_speaker"]] if version.get("model_speaker") else []),
                    "speaker": version["model_speaker"] or (speakers[0] if speakers else ""),
                    "source": version["source"],
                    "created_at": version["created_at"],
                    "model_id": version["model_id"],
                    "model_version_id": version["id"],
                    "f0_predictor": version["f0_predictor"],
                    "dataset_version_id": version["dataset_version_id"],
                    "training_mode": version["training_mode"],
                    "step_count": version["step_count"],
                    "main_preset_id": version.get("main_preset_id"),
                    "speech_encoder": version.get("speech_encoder"),
                    "use_tiny": version.get("use_tiny"),
                    "ssl_dim": version.get("ssl_dim"),
                    "gin_channels": version.get("gin_channels"),
                    "filter_channels": version.get("filter_channels"),
                    "diffusion_status": version.get("diffusion_status"),
                    "diffusion_model_path": version.get("diffusion_model_path"),
                    "diffusion_config_path": version.get("diffusion_config_path"),
                    "has_diffusion": version.get("has_diffusion"),
                    "diffusion_params": version.get("diffusion_params"),
                    "is_default": bool(version["model_default_version_id"] == version["id"]),
                }
            )
        return profiles

    def find_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        for profile in self.get_inference_profiles():
            if profile["id"] == profile_id or profile["checkpoint_path"] == profile_id:
                return profile
        return None

    def create_training_run(
        self,
        *,
        task_id: str,
        dataset_version_id: Optional[str],
        model_id: Optional[str],
        model_version_id: Optional[str],
        status: str,
        stage: str,
        message: str,
        progress: float,
    ) -> None:
        run_id = uuid.uuid4().hex
        now = _now()
        self._execute(
            """
            INSERT OR REPLACE INTO training_runs (
                id, task_id, dataset_version_id, model_id, model_version_id, status, stage, message,
                progress, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, task_id, dataset_version_id, model_id, model_version_id, status, stage, message, progress, now, now),
        )

    def update_training_run(
        self,
        *,
        task_id: str,
        status: str,
        stage: str,
        message: str,
        progress: float,
        device_used: Optional[str] = None,
        model_id: Optional[str] = None,
        model_version_id: Optional[str] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        row = self._fetch_one("SELECT * FROM training_runs WHERE task_id = ?", (task_id,))
        if not row:
            return
        self._execute(
            """
            UPDATE training_runs
            SET status = ?, stage = ?, message = ?, progress = ?, device_used = COALESCE(?, device_used),
                model_id = COALESCE(?, model_id), model_version_id = COALESCE(?, model_version_id),
                summary_json = COALESCE(?, summary_json), updated_at = ?
            WHERE task_id = ?
            """,
            (
                status,
                stage,
                message,
                progress,
                device_used,
                model_id,
                model_version_id,
                json.dumps(summary, ensure_ascii=False) if summary is not None else None,
                _now(),
                task_id,
            ),
        )

    def create_inference_run(self, *, task_id: str, model_version_id: Optional[str], message: str) -> None:
        run_id = uuid.uuid4().hex
        now = _now()
        self._execute(
            """
            INSERT OR REPLACE INTO inference_runs (
                id, task_id, model_version_id, status, stage, message, progress, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, task_id, model_version_id, "queued", "queued", message, 0.0, now, now),
        )

    def update_inference_run(
        self,
        *,
        task_id: str,
        status: str,
        stage: str,
        message: str,
        progress: float,
        device_used: Optional[str] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        row = self._fetch_one("SELECT * FROM inference_runs WHERE task_id = ?", (task_id,))
        if not row:
            return
        self._execute(
            """
            UPDATE inference_runs
            SET status = ?, stage = ?, message = ?, progress = ?, device_used = COALESCE(?, device_used),
                summary_json = COALESCE(?, summary_json), updated_at = ?
            WHERE task_id = ?
            """,
            (
                status,
                stage,
                message,
                progress,
                device_used,
                json.dumps(summary, ensure_ascii=False) if summary is not None else None,
                _now(),
                task_id,
            ),
        )

    def _migrate_legacy_models(self) -> None:
        for metadata_path in TRAINED_MODELS_DIR.rglob("metadata.json"):
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception:
                continue
            checkpoint_path = resolve_runtime_path(metadata.get("checkpoint_path"))
            config_path = resolve_runtime_path(metadata.get("config_path"))
            if not checkpoint_path or not checkpoint_path.exists() or not config_path or not config_path.exists():
                continue
            existing = self._fetch_one(
                "SELECT * FROM model_versions WHERE legacy_checkpoint_path = ? OR inference_checkpoint_path = ?",
                (metadata.get("checkpoint_path"), metadata.get("checkpoint_path")),
            )
            if existing:
                continue
            model = self._get_or_create_model(metadata.get("name") or metadata.get("label") or checkpoint_path.stem, metadata.get("speaker", ""), metadata.get("source", "legacy"))
            self._register_model_version(
                model=model,
                dataset_version_id=None,
                training_mode="legacy",
                f0_predictor="rmvpe",
                device_preference="auto",
                device_used=None,
                config_source_path=config_path,
                inference_checkpoint_source_path=checkpoint_path,
                checkpoints=[{"step": 0, "generator_path": str(checkpoint_path), "discriminator_path": None, "kind": "final", "is_final": True}],
                step_count=0,
                loss_disc=None,
                loss_gen=None,
                source=metadata.get("source", "legacy"),
                parent_version_id=None,
                parent_checkpoint_id=None,
                legacy_checkpoint_path=metadata.get("checkpoint_path"),
                label=metadata.get("label") or "迁移版本",
            )

    def _register_model_version(
        self,
        *,
        model: Dict[str, Any],
        dataset_version_id: Optional[str],
        training_mode: str,
        f0_predictor: str,
        device_preference: str,
        device_used: Optional[str],
        config_source_path: Path,
        inference_checkpoint_source_path: Path,
        checkpoints: List[Dict[str, Any]],
        step_count: int,
        loss_disc: Optional[float],
        loss_gen: Optional[float],
        source: str,
        parent_version_id: Optional[str],
        parent_checkpoint_id: Optional[str],
        legacy_checkpoint_path: Optional[str] = None,
        label: str = "",
        main_preset_id: Optional[str] = None,
        speech_encoder: Optional[str] = None,
        use_tiny: Optional[bool] = None,
        ssl_dim: Optional[int] = None,
        gin_channels: Optional[int] = None,
        filter_channels: Optional[int] = None,
        diffusion_status: str = "not_trained",
        diffusion_model_source_path: Optional[Path] = None,
        diffusion_config_source_path: Optional[Path] = None,
        diffusion_step_count: Optional[int] = None,
        diffusion_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        version_id = uuid.uuid4().hex
        now = _now()
        version_label = label.strip() or self._next_model_version_label(model["id"])
        version_dir = self.models_root / model["id"] / version_id
        checkpoints_dir = version_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        config_target = self._copy_file(config_source_path, version_dir, "config.json")
        with config_target.open("r", encoding="utf-8") as handle:
            config_data = json.load(handle)
        derived_architecture = inspect_config_architecture(config_data)
        final_generator_target = self._copy_file(
            inference_checkpoint_source_path,
            checkpoints_dir,
            Path(inference_checkpoint_source_path).name,
        )
        diffusion_target_rel = None
        diffusion_config_rel = None
        if diffusion_model_source_path and diffusion_model_source_path.exists():
            diffusion_dir = version_dir / "diffusion"
            diffusion_target = self._copy_file(diffusion_model_source_path, diffusion_dir, diffusion_model_source_path.name)
            diffusion_target_rel = self._relative_path(diffusion_target)
        if diffusion_config_source_path and diffusion_config_source_path.exists():
            diffusion_dir = version_dir / "diffusion"
            diffusion_config_target = self._copy_file(diffusion_config_source_path, diffusion_dir, "config.yaml")
            diffusion_config_rel = self._relative_path(diffusion_config_target)

        checkpoint_rows = []
        final_checkpoint_id = None
        for item in checkpoints:
            checkpoint_id = uuid.uuid4().hex
            generator_target = final_generator_target
            discriminator_target_rel = None
            generator_source = Path(item["generator_path"])
            if generator_source.resolve() != inference_checkpoint_source_path.resolve():
                generator_target = self._copy_file(generator_source, checkpoints_dir, generator_source.name)
            if item.get("discriminator_path"):
                discriminator_source = Path(item["discriminator_path"])
                discriminator_target = self._copy_file(discriminator_source, checkpoints_dir, discriminator_source.name)
                discriminator_target_rel = self._relative_path(discriminator_target)
            checkpoint_rows.append(
                (
                    checkpoint_id,
                    version_id,
                    int(item.get("step", 0)),
                    self._relative_path(generator_target),
                    discriminator_target_rel,
                    item.get("kind", "auto"),
                    1 if item.get("is_final") else 0,
                    now,
                )
            )
            if item.get("is_final"):
                final_checkpoint_id = checkpoint_id

        self._execute(
            """
            INSERT INTO model_versions (
                id, model_id, label, dataset_version_id, training_mode, f0_predictor, device_preference, device_used,
                config_path, inference_checkpoint_path, step_count, loss_disc, loss_gen, parent_version_id,
                parent_checkpoint_id, source, legacy_checkpoint_path, main_preset_id, speech_encoder, use_tiny,
                ssl_dim, gin_channels, filter_channels, diffusion_status, diffusion_model_path, diffusion_config_path,
                diffusion_step_count, diffusion_updated_at, diffusion_params_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                model["id"],
                version_label,
                dataset_version_id,
                training_mode,
                f0_predictor,
                device_preference,
                device_used,
                self._relative_path(config_target),
                self._relative_path(final_generator_target),
                step_count,
                loss_disc,
                loss_gen,
                parent_version_id,
                parent_checkpoint_id,
                source,
                legacy_checkpoint_path,
                main_preset_id or derived_architecture["main_preset_id"],
                speech_encoder or derived_architecture["speech_encoder"],
                1 if (derived_architecture["use_tiny"] if use_tiny is None else bool(use_tiny)) else 0,
                int(ssl_dim or derived_architecture["ssl_dim"]),
                int(gin_channels or derived_architecture["gin_channels"]),
                int(filter_channels or derived_architecture["filter_channels"]),
                diffusion_status or "not_trained",
                diffusion_target_rel,
                diffusion_config_rel,
                diffusion_step_count,
                now if diffusion_status and diffusion_status != "not_trained" else None,
                json.dumps(diffusion_params, ensure_ascii=False) if diffusion_params is not None else None,
                now,
                now,
            ),
        )
        if checkpoint_rows:
            self._executemany(
                """
                INSERT INTO checkpoints (
                    id, model_version_id, step, generator_path, discriminator_path, kind, is_final, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                checkpoint_rows,
            )
        self._execute("UPDATE models SET default_version_id = ?, updated_at = ? WHERE id = ?", (version_id, now, model["id"]))
        model_version = self.get_model_version(version_id)
        model_version["checkpoints"] = self.list_checkpoints(version_id)
        model_version["final_checkpoint_id"] = final_checkpoint_id
        return model_version

    def import_model(
        self,
        *,
        checkpoint_path: Path,
        config_path: Path,
        label: str,
        speaker: str,
        source: str = "imported",
    ) -> Dict[str, Any]:
        model = self._get_or_create_model(label, speaker, source)
        version = self._register_model_version(
            model=model,
            dataset_version_id=None,
            training_mode="import",
            f0_predictor="rmvpe",
            device_preference="auto",
            device_used=None,
            config_source_path=config_path,
            inference_checkpoint_source_path=checkpoint_path,
            checkpoints=[
                {
                    "step": 0,
                    "generator_path": str(checkpoint_path),
                    "discriminator_path": None,
                    "kind": "final",
                    "is_final": True,
                }
            ],
            step_count=0,
            loss_disc=None,
            loss_gen=None,
            source=source,
            parent_version_id=None,
            parent_checkpoint_id=None,
            label="导入版本",
        )
        return {
            "model_id": model["id"],
            "model_name": model["name"],
            "model_version_id": version["id"],
            "label": version["label"],
            "speaker": model["speaker"],
            "checkpoint_path": version["inference_checkpoint_path"],
            "config_path": version["config_path"],
            "source": source,
        }

    def register_trained_version(
        self,
        *,
        model_name: str,
        speaker: str,
        dataset_version_id: str,
        training_mode: str,
        f0_predictor: str,
        device_preference: str,
        device_used: Optional[str],
        config_path: Path,
        checkpoints: List[Dict[str, Any]],
        final_generator_path: Path,
        step_count: int,
        loss_disc: Optional[float],
        loss_gen: Optional[float],
        parent_version_id: Optional[str],
        parent_checkpoint_id: Optional[str],
        main_preset_id: Optional[str] = None,
        speech_encoder: Optional[str] = None,
        use_tiny: Optional[bool] = None,
        ssl_dim: Optional[int] = None,
        gin_channels: Optional[int] = None,
        filter_channels: Optional[int] = None,
        diffusion_status: str = "not_trained",
        diffusion_model_path: Optional[Path] = None,
        diffusion_config_path: Optional[Path] = None,
        diffusion_step_count: Optional[int] = None,
        diffusion_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        model = self._get_or_create_model(model_name, speaker, "training")
        return self._register_model_version(
            model=model,
            dataset_version_id=dataset_version_id,
            training_mode=training_mode,
            f0_predictor=f0_predictor,
            device_preference=device_preference,
            device_used=device_used,
            config_source_path=config_path,
            inference_checkpoint_source_path=final_generator_path,
            checkpoints=checkpoints,
            step_count=step_count,
            loss_disc=loss_disc,
            loss_gen=loss_gen,
            source="training",
            parent_version_id=parent_version_id,
            parent_checkpoint_id=parent_checkpoint_id,
            main_preset_id=main_preset_id,
            speech_encoder=speech_encoder,
            use_tiny=use_tiny,
            ssl_dim=ssl_dim,
            gin_channels=gin_channels,
            filter_channels=filter_channels,
            diffusion_status=diffusion_status,
            diffusion_model_source_path=diffusion_model_path,
            diffusion_config_source_path=diffusion_config_path,
            diffusion_step_count=diffusion_step_count,
            diffusion_params=diffusion_params,
        )

    def set_default_model_version(self, model_id: str, version_id: str) -> Dict[str, Any]:
        self._get_model_row(model_id)
        self.get_model_version(version_id)
        self._execute("UPDATE models SET default_version_id = ?, updated_at = ? WHERE id = ?", (version_id, _now(), model_id))
        return self.get_model(model_id)

    def update_model_version_diffusion(
        self,
        version_id: str,
        *,
        diffusion_status: str,
        diffusion_model_path: Optional[Path] = None,
        diffusion_config_path: Optional[Path] = None,
        diffusion_step_count: Optional[int] = None,
        diffusion_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        version = self.get_model_version(version_id)
        version_dir = self.models_root / version["model_id"] / version_id / "diffusion"
        model_rel = version.get("diffusion_model_path")
        config_rel = version.get("diffusion_config_path")
        if diffusion_model_path and diffusion_model_path.exists():
            model_target = self._copy_file(diffusion_model_path, version_dir, diffusion_model_path.name)
            model_rel = self._relative_path(model_target)
        if diffusion_config_path and diffusion_config_path.exists():
            config_target = self._copy_file(diffusion_config_path, version_dir, "config.yaml")
            config_rel = self._relative_path(config_target)
        self._execute(
            """
            UPDATE model_versions
            SET diffusion_status = ?, diffusion_model_path = ?, diffusion_config_path = ?,
                diffusion_step_count = COALESCE(?, diffusion_step_count),
                diffusion_updated_at = ?, diffusion_params_json = COALESCE(?, diffusion_params_json),
                updated_at = ?
            WHERE id = ?
            """,
            (
                diffusion_status,
                model_rel,
                config_rel,
                diffusion_step_count,
                _now(),
                json.dumps(diffusion_params, ensure_ascii=False) if diffusion_params is not None else None,
                _now(),
                version_id,
            ),
        )
        return self.get_model_version(version_id)
