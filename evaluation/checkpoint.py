"""Durable per-example benchmark records and safe continuation."""

from contextlib import contextmanager
import json
import os
import sys
from pathlib import Path


@contextmanager
def exclusive_prediction_file(path):
    """Allow readers, but prevent another Windows writer (including the old script)."""
    path = Path(path)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        import msvcrt

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        handle = create(str(path.resolve()), 0xC0000000, 1, None, 4, 0x80, None)
        if handle == wintypes.HANDLE(-1).value:
            error = ctypes.get_last_error()
            if error == 32:
                raise RuntimeError("This benchmark file is already being written. Stop the existing run first.")
            raise ctypes.WinError(error)
        try:
            descriptor = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
        except BaseException:
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle(handle)
            raise
        stream = os.fdopen(descriptor, "r+b")
    else:
        import fcntl

        stream = path.open("a+b")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            raise RuntimeError("This benchmark file is already being written. Stop the existing run first.")
    try:
        yield stream
    finally:
        stream.close()


def write_json_atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


class PredictionLog:
    def __init__(self, path, expected_rows, fingerprint, artifact_paths=(), variants=("base", "lora")):
        self.path = Path(path)
        self.expected_rows = expected_rows
        self.fingerprint = fingerprint
        self.artifact_paths = list(artifact_paths)
        self.variants = tuple(variants)
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("Expected distinct model variants.")
        stem = self.path.stem.removesuffix("-predictions")
        self.metadata_path = self.path.with_name(stem + "-metadata.json")
        self.progress_path = self.path.with_name(stem + "-progress.json")
        self.records = []
        self.completed = {}
        self.peak = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._context = exclusive_prediction_file(self.path)
        self.stream = self._context.__enter__()
        try:
            self.stream.seek(0)
            existing = self.stream.read()
            if self.metadata_path.exists():
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                if metadata["fingerprint"] != self.fingerprint:
                    raise ValueError("Model, tokenizer, test data or generation settings changed; use a new --output-dir.")
            elif existing and os.name == "nt":
                # Old versions had no fingerprints. Do not adopt them after an artifact was updated.
                stat = self.path.stat()
                created = getattr(stat, "st_birthtime", stat.st_ctime)
                if any(Path(p).stat().st_mtime > created for p in self.artifact_paths):
                    raise ValueError("An artifact changed after this legacy run started; use a new --output-dir.")

            lines = existing.splitlines(keepends=True)
            offset = 0
            for number, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if number != len(lines) - 1:
                        raise ValueError(f"Invalid JSON inside the log at line {number + 1}; refusing to modify it.")
                    # Only an incomplete final write may be discarded.
                    self.stream.seek(offset)
                    self.stream.truncate()
                    print("Discarded an incomplete final record; that case will run again.", flush=True)
                    break
                self._validate_record(record)
                key = (record["variant"], record["test_index"])
                if key in self.completed:
                    raise ValueError(f"Duplicate completed case {key}; refusing to continue.")
                self.records.append(record)
                self.completed[key] = record
                offset += len(line)
            self.stream.seek(0, os.SEEK_END)
            if self.stream.tell() and not existing[:self.stream.tell()].endswith(b"\n"):
                self.stream.write(b"\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())
            if not self.metadata_path.exists():
                write_json_atomic(self.metadata_path, {
                    "fingerprint": self.fingerprint,
                    "legacy_records_adopted": bool(existing),
                })
            if self.progress_path.exists():
                previous = json.loads(self.progress_path.read_text(encoding="utf-8"))
                self.peak = previous.get("peak_allocated_vram_gib")
            self.save_progress()
            return self
        except BaseException:
            self._context.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *exception):
        return self._context.__exit__(*exception)

    def _validate_record(self, record):
        if not isinstance(record, dict) or record.get("variant") not in self.variants:
            raise ValueError("Invalid benchmark record.")
        index = record.get("test_index")
        if type(index) is not int or index not in self.expected_rows:
            raise ValueError("Saved test indices differ from this run.")
        row = self.expected_rows[index]
        expected = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                    for c in row["messages"][-1].get("tool_calls", [])]
        if record.get("prompt") != row["messages"][-2]["content"] or record.get("expected_calls") != expected:
            raise ValueError(f"Saved question/reference changed at test index {index}.")
        required = {"output_text", "latency_seconds", "generated_tokens", "hit_token_limit",
                    "valid_tool_call", "tools_correct", "full_call_exact"}
        if not required.issubset(record):
            raise ValueError(f"Incomplete benchmark record at test index {index}.")
        if record["generated_tokens"] > self.fingerprint["max_new_tokens"]:
            raise ValueError("Saved output used a different token limit.")

    def pending(self, variant):
        return [index for index in self.expected_rows if (variant, index) not in self.completed]

    def append(self, record):
        self._validate_record(record)
        key = (record["variant"], record["test_index"])
        if key in self.completed:
            raise ValueError(f"Case {key} was already completed.")
        self.stream.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.records.append(record)
        self.completed[key] = record
        self.save_progress()

    def save_progress(self, interrupted=False, peak=None):
        if peak is not None:
            self.peak = max(self.peak or 0, peak)
        counts = {variant: sum(key[0] == variant for key in self.completed) for variant in self.variants}
        complete = len(self.completed) == len(self.expected_rows) * len(self.variants)
        write_json_atomic(self.progress_path, {
            "status": "completed" if complete else "interrupted" if interrupted else "in_progress",
            "completed": counts,
            "remaining": {variant: len(self.expected_rows) - counts[variant] for variant in counts},
            "peak_allocated_vram_gib": self.peak,
        })
