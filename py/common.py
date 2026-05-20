"""
Helpers dùng chung cho 4 test scripts:
  - Đọc API keys từ .env
  - Bảng cấu hình 4 ngôn ngữ
  - Ghi âm WAV 16 kHz mono qua PyAudio
  - Phát MP3 qua pygame.mixer
"""

from __future__ import annotations

import json
import os
import sys
import time
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

# Đường dẫn gốc dự án (chứa .env)
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")


def require_keys(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        sys.exit(f"[ERROR] Thiếu key trong .env: {', '.join(missing)}")


# =============================================================
#  4 ngôn ngữ – cấu hình STT / LLM / TTS / câu test
# =============================================================
# Voice TTS – dùng edge-tts (Microsoft Edge Online TTS, miễn phí, không cần API key).
# Tên voice xem thêm: `edge-tts --list-voices`
LANGS: Dict[str, Dict[str, str]] = {
    "vi": {
        "name": "Tiếng Việt",
        "whisper": "vi",
        "voice_id": "vi-VN-NamMinhNeural",  # nam, Việt Nam
        "system_prompt": (
            "Bạn là người bạn AI vui vẻ của trẻ em 6-12 tuổi. "
            "Xưng 'mình' hoặc 'tớ', gọi người dùng là 'bạn' — như bạn bè ngang hàng, "
            "không xưng chị/em hay thầy/cô. "
            "Giải thích đơn giản, thân thiện, dùng ví dụ gần gũi. "
            "Trả lời ngắn gọn, không quá 3 câu, tiếng Việt tự nhiên và vui vẻ."
        ),
        "test_question": "Tại sao trời lại có mưa?",
        "tts_sample": "Chào bạn! Tớ là người bạn AI của bạn đây. Hôm nay mình cùng học cho vui nhé!",
    },
    "en": {
        "name": "English",
        "whisper": "en",
        "voice_id": "en-US-AriaNeural",  # nữ, Mỹ
        "system_prompt": (
            "You are a cheerful AI buddy for kids aged 6-12. "
            "Talk like a friend the same age — casual, warm, and fun, not like a teacher. "
            "Explain things simply using examples kids can relate to. "
            "Keep replies under 3 short sentences, natural and playful."
        ),
        "test_question": "Why does it rain?",
        "tts_sample": "Hey there! I'm your AI buddy. Wanna learn something fun together today?",
    },
    "zh": {
        "name": "普通话",
        "whisper": "zh",
        "voice_id": "zh-CN-XiaoxiaoNeural",  # nữ, Đại Lục
        "system_prompt": (
            "你是一个开朗的AI小伙伴，陪6到12岁的小朋友玩和学习。"
            "说话像同龄好朋友一样，轻松、温暖、有趣，不要像老师那样。"
            "用简单的方式解释，举贴近生活的例子。"
            "回答不超过3句话，要自然、开心，用普通话。"
        ),
        "test_question": "为什么会下雨？",
        "tts_sample": "嗨！我是你的AI小伙伴。今天我们一起玩着学，好不好？",
    },
    "yue": {
        "name": "粵語",
        "whisper": "zh",  # Whisper không hỗ trợ "yue" → fallback "zh"
        "voice_id": "zh-HK-HiuMaanNeural",  # nữ, Hong Kong (Cantonese)
        "system_prompt": (
            "你係一個開心活潑嘅AI好朋友，陪6至12歲嘅細路仔玩同學嘢。"
            "講嘢要好似同齡朋友咁輕鬆、溫暖、好玩，唔好似老師咁。"
            "用簡單嘅方法解釋，舉貼近生活嘅例子。"
            "答得簡短啲，唔好超過3句，用廣東話，自然又開心。"
        ),
        "test_question": "點解會落雨㗎？",
        "tts_sample": "Hi！我係你嘅AI好朋友嚟㗎。今日我哋一齊玩住學嘢，好唔好啊？",
    },
}

ALL_LANG_CODES = list(LANGS.keys())


# =============================================================
#  AUDIO – ghi âm + phát
# =============================================================
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM
CHUNK = 1024

# Thư mục tạm cho file âm thanh
TMP_DIR = ROOT / "tests" / "_tmp"
TMP_DIR.mkdir(exist_ok=True)


def record_wav(seconds: int, out_path: Path | None = None) -> Path:
    """Ghi âm từ mic mặc định bằng sounddevice, lưu WAV 16-bit mono 16 kHz."""
    import sounddevice as sd  # import muộn → script chỉ test LLM/TTS không cần lib này

    out_path = out_path or (TMP_DIR / "record.wav")
    print(f"  🎙  Ghi âm {seconds}s... (nói ngay)")

    # sd.rec() không block; gọi sd.wait() khi xong.
    # frames=int(...), dtype='int16' → array shape (frames, channels)
    recording = sd.rec(
        frames=int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )

    # Progress bar trong khi sounddevice ghi nền
    bar_width = 20
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break
        filled = int(min(elapsed / seconds, 1.0) * bar_width)
        sys.stdout.write(
            f"\r  [{'▮' * filled}{' ' * (bar_width - filled)}] {elapsed:4.1f}/{seconds}s"
        )
        sys.stdout.flush()
        time.sleep(0.05)
    sd.wait()
    sys.stdout.write(f"\r  [{'▮' * bar_width}] done.{' ' * 8}\n")

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(recording.tobytes())
    return out_path


# =============================================================
#  VAD RECORD — ghi âm động, tự dừng khi hết tiếng nói (Silero VAD)
# =============================================================
# Silero VAD yêu cầu chunk 512 samples ở 16 kHz (= 32 ms).
_VAD_CHUNK = 512
_VAD_MODEL = None  # lazy-load 1 lần dùng lại các lượt sau


def _get_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        from silero_vad import load_silero_vad

        print("  ⏳ Load Silero VAD model (lần đầu, ~1-2s)...")
        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def record_vad(
    max_seconds: int = 15,
    silence_ms: int = 1200,
    min_speech_ms: int = 300,
    start_timeout_s: float = 5.0,
    threshold: float = 0.5,
    out_path: Path | None = None,
) -> Path:
    """Ghi âm động bằng Silero VAD.

    Quy tắc dừng:
      - Nếu đã có speech và im lặng liên tục `silence_ms` → dừng.
      - Hết `max_seconds` → dừng cứng.
      - Nếu sau `start_timeout_s` mà chưa nghe thấy gì → dừng (không nói thì thôi).

    Tham số:
      max_seconds   : trần thời lượng (an toàn).
      silence_ms    : im lặng bao lâu sau khi đã nói thì coi là xong.
      min_speech_ms : phải nói ít nhất bấy nhiêu mới cho phép dừng.
      threshold     : ngưỡng xác suất speech của Silero (0..1).
    """
    import sounddevice as sd
    import numpy as np
    import torch

    model = _get_vad_model()
    model.reset_states()

    out_path = out_path or (TMP_DIR / "record.wav")
    chunk_ms = _VAD_CHUNK * 1000 // SAMPLE_RATE  # 32 ms
    max_chunks = (max_seconds * SAMPLE_RATE) // _VAD_CHUNK
    silence_chunks_to_stop = silence_ms // chunk_ms
    min_speech_chunks = min_speech_ms // chunk_ms
    start_timeout_chunks = int(start_timeout_s * 1000 / chunk_ms)

    print(
        f"  🎙  Nói đi... (tự dừng sau {silence_ms}ms im lặng, tối đa {max_seconds}s)"
    )

    audio_chunks: list[np.ndarray] = []
    speech_chunks_total = 0
    silence_streak = 0
    speech_started = False
    last_speech_state: bool | None = None

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=_VAD_CHUNK,
    ) as stream:
        for i in range(max_chunks):
            block, _overflow = stream.read(_VAD_CHUNK)
            block = block.reshape(-1)
            audio_chunks.append(block.copy())

            audio_f = block.astype(np.float32) / 32768.0
            with torch.no_grad():
                prob = model(torch.from_numpy(audio_f), SAMPLE_RATE).item()
            is_speech = prob >= threshold

            if is_speech:
                speech_started = True
                speech_chunks_total += 1
                silence_streak = 0
            elif speech_started:
                silence_streak += 1

            # In trạng thái khi thay đổi
            if is_speech != last_speech_state:
                marker = "🟢 nghe..." if is_speech else (
                    "⚪ im..." if speech_started else "… chờ..."
                )
                sys.stdout.write(f"\r  {marker:<14}")
                sys.stdout.flush()
                last_speech_state = is_speech

            # Điều kiện dừng
            if speech_started:
                if (
                    silence_streak >= silence_chunks_to_stop
                    and speech_chunks_total >= min_speech_chunks
                ):
                    break
            else:
                if i >= start_timeout_chunks:
                    sys.stdout.write("\r  (không nghe thấy gì)        \n")
                    break

    audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.int16)
    duration = len(audio) / SAMPLE_RATE
    sys.stdout.write(f"\r  ✅ Ghi {duration:.2f}s, nói {speech_chunks_total*chunk_ms/1000:.2f}s\n")

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return out_path


def play_audio(path: Path) -> None:
    """Phát MP3/WAV qua ffplay (đi kèm ffmpeg). Block đến khi phát xong.
    Yêu cầu ffmpeg trong PATH — kiểm tra: `ffmpeg -version`."""
    import subprocess

    subprocess.run(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
        check=False,
    )


# =============================================================
#  TIMING
# =============================================================
@contextmanager
def stopwatch(label: str, store: dict | None = None):
    """`with stopwatch('STT', timings):` – in latency và lưu vào dict."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"  ⏱  {label:<10} {dt:6.2f}s")
    if store is not None:
        store[label] = dt


def print_header(title: str) -> None:
    bar = "═" * (len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}")


# =============================================================
#  CLEAN TEXT cho TTS — bỏ emoji + markdown ký tự
#  LLM hay sinh ra "**bold**", "😊", "_italic_"... Edge TTS đọc cả ký tự đó
#  ("asterisk asterisk bold" / tên emoji) → cần lọc trước khi gửi sang TTS.
# =============================================================
import re as _re   # alias để không xung đột nếu file khác cũng import re

# Khoảng emoji thường gặp. Tránh đụng CJK (U+4E00–9FFF) và Latin Extended (Việt).
_EMOJI_RE = _re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # symbols & pictographs, transport, emoticons, ...
    "\U00002600-\U000027BF"   # misc symbols + dingbats (★, ✓, ✨...)
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001F1E6-\U0001F1FF"   # regional indicators (cờ)
    "‍"                  # ZWJ (gắn các emoji ghép)
    "️"                  # variation selector cho emoji
    "]+",
    flags=_re.UNICODE,
)

# Markdown ký tự thường gặp: **bold**, *italic*, _emphasis_, `code`, ~strike~
_MD_RE = _re.compile(r"[\*_`~]+")

# Nhiều khoảng trắng liên tiếp → 1
_WS_RE = _re.compile(r"\s+")


def strip_for_tts(text: str) -> str:
    """Xoá emoji + ký tự markdown trước khi đưa vào TTS engine.
    Giữ nguyên tiếng Việt, Trung, dấu câu thường."""
    if not text:
        return text
    text = _EMOJI_RE.sub("", text)
    text = _MD_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# =============================================================
#  USER PROFILE
#  Lưu tên + ngôn ngữ ưa thích. Trên ESP32 → /profile.json trên SD.
#  Trong test pipeline → tests/_tmp/profile.json.
# =============================================================
PROFILE_PATH = TMP_DIR / "profile.json"

DEFAULT_PROFILE: dict = {
    "name": "bạn",
    "language": "vi",  # vi | en | zh | yue
    "city": "Ho Chi Minh City",  # TP mặc định cho weather query; user có thể đổi
}


def load_profile() -> dict:
    if not PROFILE_PATH.exists():
        return DEFAULT_PROFILE.copy()
    try:
        loaded = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        return {**DEFAULT_PROFILE, **loaded}
    except Exception as e:
        print(f"[Profile] Load fail: {e}")
        return DEFAULT_PROFILE.copy()


def save_profile(profile: dict) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
