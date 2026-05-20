"""
AlarmManager – báo thức cho AI voice assistant.

- Lưu/load JSON (tests/_tmp/alarms.json, hoặc đường dẫn truyền vào)
- Background thread check 30s/lần
- Hỗ trợ: 1 lần, lặp theo các thứ trong tuần (weekdays bitmap)
- Khi đến giờ → callback on_trigger(alarm) — gọi TTS + chime ở lớp tích hợp
- Tạo chuông báo gentle ramp-up (sine C-E-G) bằng pydub nếu chưa có

API chính:
    am = AlarmManager(path, on_trigger)
    am.start()
    am.add(trigger_at, weekdays, label)        # datetime, [0..6], str
    am.cancel_by_match(keyword) -> int
    am.cancel_all() -> int
    am.list_active() -> list[Alarm]
    am.format_for_prompt() -> str
    am.stop()
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

WEEKDAY_VN = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]

# Tên thứ trong tuần theo 4 ngôn ngữ (0=Thứ Hai ... 6=Chủ Nhật)
WEEKDAY_NAMES: dict[str, list[str]] = {
    "vi":  ["T2",   "T3",   "T4",   "T5",   "T6",   "T7",   "CN"],
    "en":  ["Mon",  "Tue",  "Wed",  "Thu",  "Fri",  "Sat",  "Sun"],
    "zh":  ["周一", "周二", "周三", "周四", "周五", "周六", "周日"],
    "yue": ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"],
}

# Câu báo thức khi tới giờ — placeholder {name} và {reason}
ALARM_ANNOUNCE_TEMPLATES: dict[str, str] = {
    "vi":  "Đến giờ rồi {name} ơi! {reason}.",
    "en":  "Wake up, {name}! {reason}.",
    "zh":  "起床了，{name}！{reason}。",
    "yue": "起身喇，{name}！{reason}。",
}

SUPPORTED_LANGS = list(ALARM_ANNOUNCE_TEMPLATES.keys())


# =============================================================
#  Dataclass
# =============================================================
@dataclass
class Alarm:
    id: str
    trigger_at: str             # ISO 8601, lần kế tiếp sẽ rung
    weekdays: list[int]         # rỗng = 1 lần; [0..6] (0=T2, 6=CN) = lặp
    label: str
    language: str = "vi"        # "vi" | "en" | "zh" | "yue" — ngôn ngữ khi đặt
    active: bool = True
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    # ---- helpers ----
    @property
    def trigger_dt(self) -> datetime:
        return datetime.fromisoformat(self.trigger_at)

    @property
    def is_repeating(self) -> bool:
        return len(self.weekdays) > 0

    def next_occurrence(self, after: datetime) -> datetime:
        """Lần rung kế tiếp sau `after`. Với báo thức lặp, lùi qua ngày kế tiếp khớp weekday."""
        if not self.is_repeating:
            return self.trigger_dt
        base = self.trigger_dt.time()
        cur = after.replace(
            hour=base.hour, minute=base.minute, second=base.second, microsecond=0
        )
        for delta in range(1, 8):
            cand = cur + timedelta(days=delta)
            if cand.weekday() in self.weekdays:
                return cand
        return self.trigger_dt

    def human_summary(self, language: str | None = None) -> str:
        """Tóm tắt cho hiển thị/log. `language` mặc định = ngôn ngữ alarm."""
        lang = language or self.language
        names = WEEKDAY_NAMES.get(lang, WEEKDAY_VN)
        when = self.trigger_at[:16].replace("T", " ")
        if self.is_repeating:
            wd = ",".join(names[i] for i in sorted(self.weekdays))
            return f"{when[-5:]} ({wd}) — {self.label}"
        return f"{when} — {self.label}"

    def format_announce(self, name: str = "bạn") -> str:
        """Câu thông báo khi tới giờ, theo ngôn ngữ alarm."""
        tpl = ALARM_ANNOUNCE_TEMPLATES.get(
            self.language, ALARM_ANNOUNCE_TEMPLATES["vi"]
        )
        return tpl.format(name=name, reason=self.label)


# =============================================================
#  Manager
# =============================================================
class AlarmManager:
    def __init__(
        self,
        path: Path,
        on_trigger: Callable[[Alarm], None] | None = None,
        check_interval: float = 30.0,
    ) -> None:
        self.path = path
        self.on_trigger = on_trigger or (lambda a: None)
        self.check_interval = check_interval
        self.alarms: list[Alarm] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._load()

    # ---- persistence ----
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.alarms = [Alarm(**a) for a in data]
        except Exception as e:
            print(f"[Alarm] Load fail: {e}")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = [asdict(a) for a in self.alarms]
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---- public API ----
    def add(
        self,
        trigger_at: datetime,
        weekdays: list[int] | None,
        label: str,
        language: str = "vi",
    ) -> Alarm:
        if language not in ALARM_ANNOUNCE_TEMPLATES:
            language = "vi"
        a = Alarm(
            id=f"alarm_{uuid.uuid4().hex[:8]}",
            trigger_at=trigger_at.replace(microsecond=0).isoformat(),
            weekdays=list(weekdays or []),
            label=(label or "Báo thức").strip(),
            language=language,
        )
        with self._lock:
            self.alarms.append(a)
        self._save()
        return a

    def cancel_by_match(self, keyword: str) -> int:
        """Xóa alarm có label/id chứa `keyword` (case-insensitive). Trả về số lượng đã xóa."""
        kw = keyword.strip().lower()
        if not kw:
            return 0
        with self._lock:
            before = len(self.alarms)
            self.alarms = [
                a for a in self.alarms
                if kw not in a.label.lower() and kw != a.id.lower()
            ]
            removed = before - len(self.alarms)
        if removed:
            self._save()
        return removed

    def cancel_all(self) -> int:
        with self._lock:
            n = len(self.alarms)
            self.alarms = []
        self._save()
        return n

    def list_active(self) -> list[Alarm]:
        with self._lock:
            return [a for a in self.alarms if a.active]

    def format_for_prompt(self, language: str = "vi") -> str:
        """Chuỗi ngắn để inject vào system prompt cho LLM, theo ngôn ngữ hiển thị."""
        empty_msg = {
            "vi":  "(không có báo thức nào đang đặt)",
            "en":  "(no active alarms)",
            "zh":  "（暂无闹钟）",
            "yue": "（而家無鬧鐘）",
        }.get(language, "(no active alarms)")
        active = self.list_active()
        if not active:
            return empty_msg
        return "\n".join(f"- {a.human_summary(language)}" for a in active)

    # ---- background thread ----
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="AlarmThread"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._check_due()
            except Exception as e:
                print(f"[Alarm] check error: {e}")
            self._stop.wait(timeout=self.check_interval)

    def _check_due(self) -> None:
        now = datetime.now()
        due: list[Alarm] = []
        with self._lock:
            for a in self.alarms:
                if not a.active:
                    continue
                if a.trigger_dt <= now:
                    due.append(a)
            for a in due:
                if a.is_repeating:
                    a.trigger_at = (
                        a.next_occurrence(now).isoformat(timespec="seconds")
                    )
                else:
                    a.active = False
        if due:
            self._save()
            for a in due:
                try:
                    self.on_trigger(a)
                except Exception as e:
                    print(f"[Alarm] on_trigger error for {a.id}: {e}")


# =============================================================
#  Alarm tone (sine arpeggio C-E-G, gentle ramp-in)
#  Dùng numpy + stdlib `wave`. Output WAV — ffplay phát native.
#  Không phụ thuộc pydub/audioop (Python 3.13+ bỏ audioop).
# =============================================================
def ensure_alarm_tone(path: Path, duration_seconds: int = 5) -> Path:
    """Tạo chuông báo gentle nếu chưa có. Ramp volume tăng dần ~3s đầu.
    Mặc định 5s — vừa đủ để sau đó bot đọc tên/lý do."""
    if path.exists():
        return path

    import wave
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)

    sample_rate     = 22050
    notes_hz        = [523, 659, 784, 659]   # C5 – E5 – G5 – E5
    note_dur_s      = 0.32
    gap_dur_s       = 0.12
    note_fade_in_s  = 0.04
    note_fade_out_s = 0.12

    n_note = int(note_dur_s * sample_rate)
    n_gap  = int(gap_dur_s  * sample_rate)
    n_in   = int(note_fade_in_s  * sample_rate)
    n_out  = int(note_fade_out_s * sample_rate)

    # Envelope cho mỗi note (fade in/out để tránh click)
    envelope = np.ones(n_note, dtype=np.float32)
    envelope[:n_in]    = np.linspace(0.0, 1.0, n_in,  dtype=np.float32)
    envelope[-n_out:]  = np.linspace(1.0, 0.0, n_out, dtype=np.float32)

    t = np.arange(n_note, dtype=np.float32) / sample_rate

    # Một chu kỳ pattern = các note + gap
    chunks: list[np.ndarray] = []
    for f in notes_hz:
        tone = 0.3 * np.sin(2.0 * np.pi * f * t).astype(np.float32) * envelope
        chunks.append(tone)
        chunks.append(np.zeros(n_gap, dtype=np.float32))
    pattern = np.concatenate(chunks)

    # Lặp đủ độ dài, sau đó crop
    n_total = int(duration_seconds * sample_rate)
    n_reps  = max(1, n_total // len(pattern) + 1)
    audio   = np.tile(pattern, n_reps)[:n_total]

    # Ramp volume tăng dần ~3s đầu, fade out 0.2s cuối
    ramp_n = min(int(3.0 * sample_rate), n_total // 2)
    ramp = np.concatenate(
        [np.linspace(0.0, 1.0, ramp_n, dtype=np.float32),
         np.ones(n_total - ramp_n, dtype=np.float32)]
    )
    audio *= ramp
    fade_out_n = int(0.2 * sample_rate)
    if fade_out_n < n_total:
        audio[-fade_out_n:] *= np.linspace(1.0, 0.0, fade_out_n, dtype=np.float32)

    # Giảm 10 dB tổng (≈ x0.316) cho dễ chịu
    audio *= 0.316

    # Convert float [-1,1] → int16 PCM
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path
