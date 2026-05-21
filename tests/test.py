"""
AI Voice Assistant — consolidated test entry point.

Pipeline (default mode):
    python tests/test.py                       # dùng profile.language
    python tests/test.py --lang vi             # Tiếng Việt
    python tests/test.py --lang en             # English
    python tests/test.py --lang zh             # Phổ thông
    python tests/test.py --lang yue            # Quảng Đông
    python tests/test.py --lang auto           # Whisper tự nhận dạng
    python tests/test.py --loop                # chạy liên tục (Ctrl+C để thoát)
    python tests/test.py --seconds 8 --rounds 3
    python tests/test.py --no-vad              # tắt VAD, ghi cố định --seconds
    python tests/test.py --loop --seconds 20   # với VAD: 20s là TRẦN tối đa

Voice comparison (Edge TTS):
    python tests/test.py --voices              # cả vi-VN và zh-HK
    python tests/test.py --voices --locale vi-VN
    python tests/test.py --voices --no-play

Alarm tests:
    python tests/test.py --alarm                          # cả 3 suite
    python tests/test.py --alarm --suite manager
    python tests/test.py --alarm --suite parser
    python tests/test.py --alarm --suite live --in 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Cho phép `from py....` khi chạy `python tests/test.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import edge_tts
import requests
from groq import Groq

from py.alarm_manager import (
    Alarm,
    AlarmManager,
    SUPPORTED_LANGS,
    WEEKDAY_NAMES,
    ensure_alarm_tone,
)
from py.alarm_parser import build_chat_system_prompt, parse_intent
from py.memory import ConversationMemory
from py.search import brave_search, format_for_llm as format_search_for_llm
from py.sgk_teacher import (
    SGK_DIR,
    build_vision_user_content,
    find_relevant_pages,
    load_lessons,
    load_master_index,
    render_sgk_system_prompt,
)
from py.sgk_progress import SGKProgress, format_progress_note
from py.common import (
    ALL_LANG_CODES,
    DEEPSEEK_API_KEY,
    GROQ_API_KEY,
    LANGS,
    TMP_DIR,
    load_profile,
    play_audio,
    print_header,
    record_vad,
    record_wav,
    require_keys,
    stopwatch,
    strip_for_tts,
)

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
# Vision model cho SGK pipeline. DeepSeek-chat (V3) là text-only, không nhận
# image_url content → phải dùng Groq Llama-4 Scout (vision-capable, OpenAI-
# compatible) cho câu hỏi học bài có ảnh SGK đính kèm.
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
ALARMS_PATH = TMP_DIR / "alarms.json"
ALARM_TONE = TMP_DIR / "alarm_tone.wav"
CHAT_HISTORY_PATH = TMP_DIR / "chat_history.json"
SGK_PROGRESS_PATH = TMP_DIR / "sgk_progress.json"
CHAT_MAX_TURNS = 12  # số cặp gửi cho LLM mỗi lần (đĩa lưu vô tận)

# Whisper verbose_json trả về tên ngôn ngữ dạng English lowercase
WHISPER_LANG_NAME_TO_KEY: dict[str, str] = {
    "vietnamese": "vi",
    "english": "en",
    "chinese": "zh",
    "mandarin": "zh",
    "cantonese": "yue",
}


# =============================================================
#  CONFIRM TEMPLATES — câu xác nhận báo thức theo 4 ngôn ngữ
# =============================================================
CONFIRM_TEMPLATES: dict[str, dict[str, str]] = {
    "vi": {
        "create_once": "Ok, tớ sẽ nhắc bạn lúc {time} ngày {date} nhé. Nội dung: {label}.",
        "create_repeat": "Ok, tớ sẽ nhắc bạn lúc {time} các ngày {days} nhé. Nội dung: {label}.",
        "early_tip": " Đặt sớm thế này thì nhớ ngủ sớm bạn nha!",
        "missing_time": "Mình chưa hiểu rõ thời gian, bạn nói lại được không?",
        "missing_match": "Bạn muốn hủy báo thức nào? Nói rõ giúp tớ với.",
        "cancel_one_ok": "Đã hủy {n} báo thức có từ '{match}' rồi nhé.",
        "cancel_one_no": "Tớ không tìm thấy báo thức nào có từ '{match}' cả.",
        "cancel_all_ok": "Đã hủy hết {n} báo thức rồi nhé.",
        "cancel_all_no": "Hiện không có báo thức nào để hủy.",
        "list_empty": "Hiện bạn chưa đặt báo thức nào cả.",
        "list_some": "Bạn đang có {n} báo thức: {items}.",
    },
    "en": {
        "create_once": "Okay, I'll remind you at {time} on {date}. About: {label}.",
        "create_repeat": "Okay, I'll remind you at {time} on {days}. About: {label}.",
        "early_tip": " That's pretty early — don't forget to go to bed soon!",
        "missing_time": "I didn't catch the time, could you say it again?",
        "missing_match": "Which alarm should I cancel? Tell me more.",
        "cancel_one_ok": "Cancelled {n} alarm(s) matching '{match}'.",
        "cancel_one_no": "Couldn't find any alarm matching '{match}'.",
        "cancel_all_ok": "Cancelled all {n} alarms.",
        "cancel_all_no": "There are no alarms to cancel.",
        "list_empty": "You don't have any alarms set.",
        "list_some": "You have {n} alarm(s): {items}.",
    },
    "zh": {
        "create_once": "好的，我会在 {date} {time} 提醒你。内容：{label}。",
        "create_repeat": "好的，我会在 {days} 的 {time} 提醒你。内容：{label}。",
        "early_tip": " 设得这么早呀，记得早点睡觉哦！",
        "missing_time": "我没听清楚时间，你能再说一次吗？",
        "missing_match": "你想取消哪个闹钟？再说清楚一点。",
        "cancel_one_ok": "已经取消 {n} 个含‘{match}’的闹钟了。",
        "cancel_one_no": "没找到含‘{match}’的闹钟。",
        "cancel_all_ok": "已经取消了全部 {n} 个闹钟。",
        "cancel_all_no": "现在没有可以取消的闹钟。",
        "list_empty": "你还没设置任何闹钟。",
        "list_some": "你目前有 {n} 个闹钟：{items}。",
    },
    "yue": {
        "create_once": "好啊，我會喺 {date} {time} 提你。內容：{label}。",
        "create_repeat": "好啊，我會喺 {days} 嘅 {time} 提你。內容：{label}。",
        "early_tip": " 咁早就設咗呀？記得早啲訓覺啊！",
        "missing_time": "我聽唔清個時間，你可以再講多次嗎？",
        "missing_match": "你想取消邊個鬧鐘？講清楚啲。",
        "cancel_one_ok": "已經取消咗 {n} 個含「{match}」嘅鬧鐘。",
        "cancel_one_no": "搵唔到含「{match}」嘅鬧鐘。",
        "cancel_all_ok": "已經取消晒全部 {n} 個鬧鐘。",
        "cancel_all_no": "而家無鬧鐘可以取消。",
        "list_empty": "你重未設定任何鬧鐘。",
        "list_some": "你而家有 {n} 個鬧鐘：{items}。",
    },
}


# =============================================================
#  STT / LLM / TTS
# =============================================================
# Whisper `prompt` hint: bias decoder để giữ Han/Anh xen kẽ trong câu Việt.
# Vài ví dụ Han + Anh giúp Whisper biết user hay code-switch.
WHISPER_HINT = (
    "Người dùng có thể nói tiếng Việt xen với tiếng Hoa và tiếng Anh. "
    "Ví dụ: 太阳, 你好, 月亮, hello, sunshine, what's your name. "
    "Hãy giữ chữ Hán và chữ Latin nguyên vẹn, đừng phiên âm sang tiếng Việt."
)


def stt(client: Groq, wav: Path, lang_code: str) -> str:
    with open(wav, "rb") as f:
        r = client.audio.transcriptions.create(
            file=(wav.name, f.read()),
            model="whisper-large-v3",
            language=lang_code,
            prompt=WHISPER_HINT,
            response_format="json",
        )
    return (r.text or "").strip()


# Trợ từ/hư từ đặc trưng tiếng Quảng — KHÔNG dùng trong tiếng Hoa phổ thông.
# Có bất kỳ ký tự nào trong tập này → coi là Cantonese.
_YUE_MARKERS = set("嘅咁嚟喺唔啲咗佢嗰冇咁樣咩嘢點解啦㗎喎噃畀")


def _looks_like_cantonese(text: str) -> bool:
    return any(c in _YUE_MARKERS for c in text)


def stt_auto(client: Groq, wav: Path) -> tuple[str, str | None]:
    """Whisper tự nhận dạng ngôn ngữ. Trả về (transcript, lang_key_or_None).
    Whisper trả 'chinese' chung cho cả Mandarin lẫn Cantonese → cần check
    trợ từ đặc trưng để tách yue ra."""
    with open(wav, "rb") as f:
        r = client.audio.transcriptions.create(
            file=(wav.name, f.read()),
            model="whisper-large-v3",
            prompt=WHISPER_HINT,
            response_format="verbose_json",
        )
    text = (r.text or "").strip()
    detected_name = (getattr(r, "language", "") or "").lower()
    lang_key = WHISPER_LANG_NAME_TO_KEY.get(detected_name)
    # Whisper nói "chinese"/"mandarin" → check xem có phải Cantonese không
    if lang_key == "zh" and _looks_like_cantonese(text):
        print("     ↪ Detect trợ từ Cantonese → chuyển zh → yue")
        lang_key = "yue"
    return text, lang_key


def llm(
    messages: list[dict],
    temperature: float = 0.7,
    json_mode: bool = True,
) -> str:
    """messages = [{role:system, ...}, {role:user, ...}, {role:assistant, ...}, ...]
    Caller chịu trách nhiệm dựng list (system prompt + history + current user)."""
    payload: dict = {
        "model": "deepseek-chat",
        "temperature": temperature,
        "max_tokens": 700,
        "messages": messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(
        DEEPSEEK_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45,
    )
    if r.status_code >= 400:
        # In ra body để biết DeepSeek phàn nàn gì (vd. content type không hỗ trợ).
        snippet = (r.text or "")[:500]
        print(f"  ⚠ DeepSeek HTTP {r.status_code}: {snippet}")
    r.raise_for_status()
    body = r.json()
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    finish = choice.get("finish_reason")
    if not content.strip():
        print(f"  ⚠ DeepSeek content rỗng. finish_reason={finish!r}")
        print(f"      message dict: {msg!r}")
        print(f"      usage: {body.get('usage')}")
        # Đôi khi nội dung nằm ở field khác (reasoning_content...) — lấy ra dùng tạm
        for alt in ("reasoning_content", "thinking", "text"):
            val = msg.get(alt)
            if isinstance(val, str) and val.strip():
                print(f"      ↪ Dùng tạm '{alt}' làm content.")
                return val.strip()
    return content.strip()


def llm_vision(
    client: Groq,
    messages: list[dict],
    temperature: float = 0.6,
    max_tokens: int = 2000,
) -> str:
    """Vision LLM qua Groq (Llama-4 Scout). DeepSeek-chat không support image_url
    content nên SGK pipeline phải dùng provider khác. Groq client đã có sẵn
    (dùng chung với Whisper STT) → tiết kiệm 1 dependency.

    max_tokens=2000 để có chỗ ĐỌC NGUYÊN VĂN cả bài SGK khi bạn ấy yêu cầu
    "đọc hết bài" (1 trang SGK lớp 2 ~ 200-400 từ × ~1.7 token/từ tiếng Việt
    + buffer cho câu hỏi ngược cuối + JSON wrapper).

    KHÔNG dùng json_object response_format vì Groq Llama vision không bảo đảm
    hỗ trợ. System prompt đã yêu cầu output JSON → model sẽ tự tuân thủ, và
    parse_tts_segments có fallback cho plain text."""
    try:
        resp = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"  ⚠ Groq vision lỗi: {type(e).__name__}: {e}")
        return ""
    try:
        return (resp.choices[0].message.content or "").strip()
    except (AttributeError, IndexError):
        return ""


def tts(voice: str, text: str, out: Path, retries: int = 2) -> Path:
    """Edge TTS đôi khi trả NoAudioReceived do mạng/server. Retry vài lần."""
    async def _run() -> None:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(str(out))

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            asyncio.run(_run())
            return out
        except Exception as e:
            last_err = e
            if attempt < retries:
                print(f"     ↻ Edge TTS attempt {attempt + 1} fail ({type(e).__name__}), retry...")
                time.sleep(0.4 * (attempt + 1))
    raise last_err  # type: ignore[misc]


# Các chữ cái có dấu đặc trưng tiếng Việt — nếu cụm Latin trong dấu ngoặc kép
# KHÔNG chứa bất kỳ ký tự nào trong tập này, ta coi đó là tiếng Anh.
_VI_DIACRITICS = set(
    "ăâđêôơư"
    "áàảãạắằẳẵặấầẩẫậ"
    "éèẻẽẹếềểễệ"
    "íìỉĩị"
    "óòỏõọốồổỗộớờởỡợ"
    "úùủũụứừửữự"
    "ýỳỷỹỵ"
)
_VI_DIACRITICS |= {c.upper() for c in _VI_DIACRITICS}

# Bắt cụm nằm trong dấu nháy kép kiểu thẳng hoặc cong: "...", "...", '...'
_QUOTED_RE = re.compile(r'["“”„«»](.+?)["“”„«»]|\'([A-Za-z][^\']{2,})\'')


def _looks_like_english(phrase: str) -> bool:
    """Đoán cụm trong ngoặc kép là tiếng Anh: có chữ Latin và KHÔNG có dấu Việt,
    không có ký tự CJK. Cho phép số, dấu câu cơ bản, dấu apostrophe."""
    if not phrase or not any(c.isalpha() for c in phrase):
        return False
    for c in phrase:
        if c in _VI_DIACRITICS:
            return False
        if "一" <= c <= "鿿":  # CJK
            return False
    # Phải có ít nhất 1 ký tự ASCII Latin
    return any("a" <= c.lower() <= "z" for c in phrase)


def _split_embedded_english(
    segments: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Nếu segment lang=vi/zh/yue chứa cụm trong "..." trông như tiếng Anh thuần,
    tách cụm đó thành segment lang=en riêng để TTS dùng giọng Anh."""
    out: list[dict[str, str]] = []
    for seg in segments:
        if seg["lang"] == "en":
            out.append(seg)
            continue
        text = seg["text"]
        # Skip nếu text trông như JSON thô — tránh tự chặt "segments","lang","vi" thành tokens
        stripped = text.lstrip()
        if stripped.startswith(("{", "[")) and '"segments"' in stripped:
            out.append(seg)
            continue
        pieces: list[dict[str, str]] = []
        last = 0
        for m in _QUOTED_RE.finditer(text):
            phrase = m.group(1) if m.group(1) is not None else m.group(2)
            if not _looks_like_english(phrase):
                continue
            before = text[last : m.start()].strip(" \t")
            if before:
                pieces.append({"lang": seg["lang"], "text": before})
            pieces.append({"lang": "en", "text": phrase.strip()})
            last = m.end()
        if not pieces:
            out.append(seg)
            continue
        tail = text[last:].strip(" \t")
        if tail:
            pieces.append({"lang": seg["lang"], "text": tail})
        out.extend(pieces)
    return out


# Dấu thanh chỉ có trong Pinyin, KHÔNG dùng trong tiếng Việt → tín hiệu mạnh
_PINYIN_EXCLUSIVE = set("āǎēěīǐōǒūǔǖǘǚǜńňǹḿ")
_PINYIN_EXCLUSIVE |= {c.upper() for c in _PINYIN_EXCLUSIVE}

# Dấu thanh chỉ tiếng Việt mới có (Pinyin không dùng)
_VI_EXCLUSIVE = set(
    "ăâđêôơư" "ắằẳẵặấầẩẫậ" "ẻẽẹếềểễệ" "ỉĩị" "ỏõọốồổỗộớờởỡợ" "ủũụứừửữự" "ỳỷỹỵ"
)
_VI_EXCLUSIVE |= {c.upper() for c in _VI_EXCLUSIVE}

_PUNCT_TRIM = " \t.,!?;:\"'()[]{}“”‘’«»。，！？；："


def _classify_token(token: str) -> str:
    """Phân loại 1 token: vi | cjk | py | amb (Latin trung lập) | sep | other."""
    if not token or token.isspace():
        return "sep"
    word = token.strip(_PUNCT_TRIM)
    if not word:
        return "sep"
    if any(c in _VI_EXCLUSIVE for c in word):
        return "vi"
    if any("一" <= c <= "鿿" for c in word):
        return "cjk"
    if any(c in _PINYIN_EXCLUSIVE for c in word):
        return "py"
    if any(c.isalpha() for c in word):
        return "amb"
    return "other"


def _split_pinyin(segments: list[dict[str, str]]) -> list[dict[str, str]]:
    """XOÁ Pinyin/Jyutping khỏi segment vi/yue/en. Edge TTS đọc không nổi Latin
    có thanh Hán → cứ bỏ luôn cho gọn. LLM được dặn không trả Pinyin nhưng
    đây là lưới an toàn nếu nó vẫn lỡ trả về."""
    out: list[dict[str, str]] = []
    for seg in segments:
        if seg["lang"] == "zh":
            out.append(seg)
            continue
        text = seg["text"]
        tokens = re.split(r"(\s+)", text)
        n = len(tokens)
        if n <= 1:
            out.append(seg)
            continue
        classes = [_classify_token(t) for t in tokens]
        is_py = [False] * n

        for i in range(n):
            if classes[i] != "py":
                continue
            is_py[i] = True
            j = i + 1
            while j < n:
                if classes[j] == "sep":
                    j += 1
                elif classes[j] in ("py", "amb"):
                    is_py[j] = True
                    j += 1
                else:
                    break

        # Nối separator nằm giữa 2 từ đã đánh dấu py
        for i in range(1, n - 1):
            if classes[i] == "sep" and is_py[i - 1] and is_py[i + 1]:
                is_py[i] = True

        if not any(is_py):
            out.append(seg)
            continue

        # Xoá tất cả token Pinyin, gộp phần còn lại
        kept = [tok for tok, py in zip(tokens, is_py) if not py]
        cleaned = re.sub(r"\s+", " ", "".join(kept)).strip()
        if cleaned:
            out.append({"lang": seg["lang"], "text": cleaned})
    return out


def parse_tts_segments(answer: str, fallback_lang: str) -> list[dict[str, str]]:
    """Parse LLM JSON segments.
    Preferred: {"segments": [{"lang":"vi|en|zh|yue", "text":"..."}]}.
    Also accepts old raw array format.
    If the model replies as plain text, keep the old single-voice behavior."""
    fallback_lang = fallback_lang if fallback_lang in LANGS else "vi"
    raw = (answer or "").strip()
    obj_start = raw.find("{")
    obj_end = raw.rfind("}")
    arr_start = raw.find("[")
    arr_end = raw.rfind("]")
    if obj_start >= 0 and obj_end > obj_start:
        json_text = raw[obj_start : obj_end + 1]
    elif arr_start >= 0 and arr_end > arr_start:
        json_text = raw[arr_start : arr_end + 1]
    else:
        json_text = raw

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return _split_pinyin(
            _split_embedded_english([{"lang": fallback_lang, "text": raw}])
        )

    if isinstance(data, dict):
        # Ưu tiên key "segments"; nếu LLM lười trả {"answer":"..."} hay {"reply":"..."}
        # → coi như plain text rồi để splitter tự tách English ra.
        if "segments" in data:
            data = data["segments"]
        else:
            for key in ("answer", "reply", "response", "text", "message", "content"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return _split_pinyin(
                        _split_embedded_english(
                            [{"lang": fallback_lang, "text": val.strip()}]
                        )
                    )
            data = []

    if not isinstance(data, list):
        return _split_pinyin(
            _split_embedded_english([{"lang": fallback_lang, "text": raw}])
        )

    def _coerce_segments(data_obj, depth: int = 0) -> list[dict[str, str]]:
        """Convert JSON-ish data into TTS segments, unwrapping accidental nested
        {"segments": ...} strings that the model sometimes puts inside text."""
        if depth > 6:
            return []
        if isinstance(data_obj, dict):
            if "segments" in data_obj:
                return _coerce_segments(data_obj["segments"], depth + 1)
            for key in ("answer", "reply", "response", "text", "message", "content"):
                val = data_obj.get(key)
                if isinstance(val, str) and val.strip():
                    return _coerce_segments(val.strip(), depth + 1)
            return []
        if isinstance(data_obj, str):
            nested = data_obj.strip()
            if not nested.startswith(("{", "[")):
                return []
            try:
                return _coerce_segments(json.loads(nested), depth + 1)
            except json.JSONDecodeError:
                return []
        if not isinstance(data_obj, list):
            return []

        coerced: list[dict[str, str]] = []
        for item in data_obj:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue

            nested_segments = _coerce_segments(text, depth + 1)
            if nested_segments:
                coerced.extend(nested_segments)
                continue

            lang = str(item.get("lang", fallback_lang)).strip().lower()
            if lang not in LANGS:
                lang = fallback_lang
            coerced.append({"lang": lang, "text": text})
        return coerced

    segments: list[dict[str, str]] = _coerce_segments(data)

    if not segments:
        return _split_pinyin(
            _split_embedded_english([{"lang": fallback_lang, "text": raw}])
        )
    return _split_pinyin(_split_embedded_english(segments))


def segments_to_text(segments: list[dict[str, str]]) -> str:
    return " ".join(s["text"] for s in segments if s.get("text")).strip()


def _split_long_tts_text(text: str, max_chars: int = 250) -> list[str]:
    """Cắt text dài theo ranh giới câu để Edge TTS không trả NoAudioReceived.

    Edge TTS server đôi khi từ chối các text quá dài (vài trăm ký tự trở lên,
    đặc biệt với giọng vi-VN) — trả về NoAudioReceived dù retry. Cắt theo dấu
    câu (. ! ? …) trước, nếu chunk vẫn dài thì cắt tiếp theo dấu phẩy.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?…])\s+", text)
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur = cur + " " + s
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)

    # Câu siêu dài (không có dấu chấm) → cắt thêm theo dấu phẩy / chấm phẩy.
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
            continue
        parts = re.split(r"(?<=[,;])\s+", chunk)
        cur = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if not cur:
                cur = p
            elif len(cur) + 1 + len(p) <= max_chars:
                cur = cur + " " + p
            else:
                final.append(cur)
                cur = p
        if cur:
            final.append(cur)
    return final


def synth_tts_segments(
    segments: list[dict[str, str]], prefix: str
) -> list[tuple[Path, str, str]]:
    rendered: list[tuple[Path, str, str]] = []
    file_idx = 0
    for seg_idx, segment in enumerate(segments, start=1):
        lang = segment["lang"]
        voice = LANGS[lang]["voice_id"]
        spoken = strip_for_tts(segment["text"])
        if not spoken:
            continue
        # Cắt nhỏ trước khi gọi Edge TTS để tránh NoAudioReceived ở text dài.
        chunks = _split_long_tts_text(spoken)
        for chunk_idx, chunk in enumerate(chunks, start=1):
            file_idx += 1
            out = TMP_DIR / f"{prefix}_{file_idx}_{lang}.mp3"
            print(
                f"  🔊 TTS seg {seg_idx}.{chunk_idx}: lang={lang} voice={voice}"
                f" ({len(chunk)} chars)"
            )
            try:
                tts(voice, chunk, out)
            except Exception as e:
                print(
                    f"  ⚠ Edge TTS lỗi ở seg {seg_idx}.{chunk_idx} "
                    f"({type(e).__name__}: {e}) — bỏ qua chunk này."
                )
                continue
            rendered.append((out, lang, chunk))
    return rendered


# =============================================================
#  ALARM — handle intent + trigger callback
# =============================================================
def _format_weekdays(weekdays: list[int], language: str) -> str:
    names = WEEKDAY_NAMES.get(language, WEEKDAY_NAMES["en"])
    return ", ".join(names[i] for i in sorted(weekdays))


def handle_alarm_intent(
    intent: dict, alarm_mgr: AlarmManager, language: str
) -> str | None:
    """Thực thi intent báo thức. Trả về câu xác nhận hoặc None nếu là chat."""
    kind = intent.get("intent", "chat")
    if kind == "chat":
        return None

    lang = language if language in CONFIRM_TEMPLATES else "vi"
    tpl = CONFIRM_TEMPLATES[lang]

    if kind == "alarm_create":
        try:
            trigger_at = datetime.fromisoformat(intent["trigger_at"])
        except (KeyError, ValueError, TypeError):
            return tpl["missing_time"]

        weekdays = intent.get("weekdays") or []
        label = (intent.get("label") or "").strip() or "—"
        alarm_mgr.add(trigger_at, weekdays, label, language=lang)

        time_str = trigger_at.strftime("%H:%M")
        if weekdays:
            confirm = tpl["create_repeat"].format(
                time=time_str, days=_format_weekdays(weekdays, lang), label=label
            )
        else:
            confirm = tpl["create_once"].format(
                time=time_str, date=trigger_at.strftime("%d/%m"), label=label
            )
        if trigger_at.hour < 7:
            confirm += tpl["early_tip"]
        return confirm

    if kind == "alarm_cancel_one":
        match = (intent.get("match") or "").strip()
        if not match:
            return tpl["missing_match"]
        n = alarm_mgr.cancel_by_match(match)
        if n == 0:
            return tpl["cancel_one_no"].format(match=match)
        return tpl["cancel_one_ok"].format(n=n, match=match)

    if kind == "alarm_cancel_all":
        n = alarm_mgr.cancel_all()
        if n == 0:
            return tpl["cancel_all_no"]
        return tpl["cancel_all_ok"].format(n=n)

    if kind == "alarm_list":
        active = alarm_mgr.list_active()
        if not active:
            return tpl["list_empty"]
        items = "; ".join(a.human_summary(lang) for a in active)
        return tpl["list_some"].format(n=len(active), items=items)

    return None


def make_alarm_trigger_callback(profile: dict):
    """Callback chạy trong AlarmManager thread khi đến giờ."""

    def _on_trigger(alarm: Alarm) -> None:
        lang = alarm.language if alarm.language in SUPPORTED_LANGS else "vi"
        voice = LANGS.get(lang, LANGS["vi"])["voice_id"]
        name = profile.get("name", "bạn")

        print(f"\n  🔔 BÁO THỨC ({lang}): {alarm.human_summary(lang)}")
        play_audio(ensure_alarm_tone(ALARM_TONE))
        announce = alarm.format_announce(name)
        print(f"  💬 [{lang}] {announce}")
        out = TMP_DIR / f"alarm_announce_{alarm.id}.mp3"
        try:
            tts(voice, strip_for_tts(announce), out)
            play_audio(out)
        except Exception as e:
            print(f"  ⚠ Không phát được announce: {e}")

    return _on_trigger


# =============================================================
#  PIPELINE — một lượt ghi âm → xử lý → trả lời
# =============================================================
def one_round(
    client: Groq,
    lang_key: str,
    seconds: int,
    round_idx: int,
    alarm_mgr: AlarmManager,
    profile: dict,
    memory: ConversationMemory,
    sgk_progress: SGKProgress,
    use_vad: bool = True,
) -> dict:
    is_auto = lang_key == "auto"

    if is_auto:
        print_header(f"PIPELINE • AUTO-DETECT  • Lượt {round_idx}")
        print("  Gợi ý: nói bất kỳ ngôn ngữ nào (vi/en/zh/yue) — Whisper tự nhận dạng.")
    else:
        cfg_preview = LANGS[lang_key]
        print_header(f"PIPELINE • {cfg_preview['name']}  • Lượt {round_idx}")
        print("  Gợi ý: hỏi gia sư AI hoặc đặt báo thức.")

    actives = alarm_mgr.list_active()
    if actives:
        print(f"  📋 Đang có {len(actives)} báo thức:")
        for a in actives:
            print(f"     • [{a.language}] {a.human_summary(a.language)}")

    if len(memory) > 0:
        print(f"  🧠 Memory: {memory.summary()}")

    input("  ↪ Nhấn ENTER để bắt đầu (Ctrl+C để thoát)...")

    timings: dict = {}
    t_total_start = time.perf_counter()

    wav = TMP_DIR / f"pipe_{lang_key}_{round_idx}.wav"
    with stopwatch("Record", timings):
        if use_vad:
            record_vad(max_seconds=seconds, out_path=wav)
        else:
            record_wav(seconds, wav)

    detected_key: str | None = None
    with stopwatch("STT", timings):
        if is_auto:
            transcript, detected_key = stt_auto(client, wav)
        else:
            transcript = stt(client, wav, LANGS[lang_key]["whisper"])
            detected_key = lang_key

    if detected_key is None:
        detected_key = profile.get("language", "vi")
        print(f"  ⚠  Không nhận dạng được ngôn ngữ, dùng profile → '{detected_key}'")

    cfg = LANGS[detected_key]
    if is_auto:
        print(
            f"  🌐 Detected: {cfg['name']} ({detected_key}) → voice = {cfg['voice_id']}"
        )

    print(f"  👤 User : {transcript or '<empty>'}")
    if not transcript:
        print("  ⚠  Không nhận được giọng nói, bỏ qua lượt này.")
        return {}

    answer = ""
    intent_handled = False
    search_context: str | None = None
    user_city = profile.get("city", "Ho Chi Minh City")
    with stopwatch("Intent", timings):
        try:
            intent = parse_intent(
                DEEPSEEK_API_KEY,
                transcript,
                alarm_mgr.format_for_prompt(detected_key),
                user_city=user_city,
            )
        except Exception as e:
            print(f"  ⚠ Parse intent fail: {e}")
            intent = {"intent": "chat", "language": detected_key}

    intent_lang = intent.get("language") or detected_key
    print(f"  🎯 Intent: {intent.get('intent')}  (lang={intent_lang})")

    # Khi --lang bị forced (vd. vi) nhưng parse_intent nhìn transcript thấy
    # ngôn ngữ khác (vd. user nói tiếng Hoa) → chuyển detected_key + cfg theo
    # intent_lang để system prompt + voice TTS đúng ngôn ngữ phản hồi.
    if intent_lang in LANGS and intent_lang != detected_key:
        print(f"     ↪ Override response lang: {detected_key} → {intent_lang}")
        detected_key = intent_lang
        cfg = LANGS[intent_lang]

    if intent.get("intent") in ("alarm_create", "alarm_cancel_one", "alarm_cancel_all", "alarm_list"):
        confirm = handle_alarm_intent(intent, alarm_mgr, intent_lang)
        if confirm is not None:
            answer = confirm
            intent_handled = True
            cfg = LANGS.get(intent_lang, cfg)

    elif intent.get("intent") == "web_search":
        query = (intent.get("query") or transcript).strip()
        # Nếu là weather query mà không có TP nào trong query → chèn TP mặc định
        q_lower = query.lower()
        weather_kw = ("thời tiết", "weather", "天气", "天氣", "nhiệt độ", "temperature", "mưa", "rain", "nắng", "sunny")
        if any(k in q_lower for k in weather_kw) and user_city.lower() not in q_lower:
            query = f"{query} {user_city}"
        print(f"  🔎 Brave Search: {query!r}")
        with stopwatch("Search", timings):
            try:
                results = brave_search(query, count=5)
                search_context = format_search_for_llm(results)
                print(f"     → {len(results)} kết quả ({len(search_context)} chars)")
            except Exception as e:
                print(f"  ⚠ Brave Search fail: {type(e).__name__}: {e}")
                search_context = "(không tìm được kết quả)"

    # SGK lookup — chỉ làm khi intent là chat thường (không alarm/search), để
    # dành vision API cho câu hỏi học bài.
    sgk_pages: list[dict] = []
    sgk_next_lesson: dict | None = None
    sgk_master_idx: dict | None = None
    if not intent_handled and not search_context and detected_key == "vi":
        try:
            sgk_master_idx = load_master_index(SGK_DIR)
            sgk_pages = find_relevant_pages(
                transcript, sgk_master_idx, SGK_DIR, max=2
            )
        except FileNotFoundError:
            pass  # SGK chưa setup → bỏ qua, dùng chat thường
        except Exception as e:
            print(f"  ⚠ SGK lookup fail: {type(e).__name__}: {e}")
        if sgk_pages:
            head = sgk_pages[0]
            pages_str = ", ".join("trang " + str(p["page"]) for p in sgk_pages)
            print(
                f"  📖 SGK: {head['subject_name']} lớp {head['grade']}"
                f" tập {head.get('tap', '?')} — {pages_str}"
            )
            # Tính bài kế tiếp trong sách dựa vào tiến độ đã ghi nhận
            if sgk_master_idx is not None:
                try:
                    lessons = load_lessons(
                        sgk_master_idx, SGK_DIR,
                        head["subject"], head["grade"], head.get("tap"),
                    )
                    sgk_next_lesson = sgk_progress.next_lesson(
                        head["subject"], head["grade"], head.get("tap"), lessons,
                    )
                except Exception as e:
                    print(f"  ⚠ SGK next_lesson fail: {type(e).__name__}: {e}")
            prev = sgk_progress.latest_before_today()
            if prev:
                print(
                    f"  🧠 Buổi trước ({prev.get('date')}): "
                    f"{prev.get('subject')} lớp {prev.get('grade')} "
                    f"trang {prev.get('page')}"
                )
            if sgk_next_lesson:
                nxt_p = (
                    sgk_next_lesson.get("book_page")
                    or sgk_next_lesson.get("pdf_page")
                )
                print(f"  ➡️  Bài kế tiếp đề xuất: trang {nxt_p}")

    if not intent_handled:
        if sgk_pages:
            # SGK teacher mode: vision message với ảnh SGK + system prompt cô giáo
            progress_note = format_progress_note(
                sgk_progress,
                current_pages=sgk_pages,
                next_lesson=sgk_next_lesson,
            )
            sys_prompt = render_sgk_system_prompt(
                ten=profile.get("name", "bé"),
                lop=profile.get("grade", 2),
                progress_note=progress_note,
            )
            user_content = build_vision_user_content(transcript, sgk_pages)
            # History chỉ chứa text (memory.add_user lưu transcript text, không có
            # base64) → không lo blow up token cho lượt sau.
            messages = [
                {"role": "system", "content": sys_prompt},
                *memory.for_llm(),
                {"role": "user", "content": user_content},
            ]
        else:
            sys_prompt = build_chat_system_prompt(
                cfg["system_prompt"],
                datetime.now(),
                alarm_mgr.format_for_prompt(detected_key),
                user_name=profile.get("name", "bạn"),
                language=detected_key,
            )
            # Nếu có search_context → chèn vào user message để LLM tổng hợp từ kết quả thật,
            # không dùng training data (vốn outdated).
            if search_context:
                instr = {
                    "vi": "Dùng các kết quả trên để trả lời ngắn gọn, tự nhiên bằng tiếng Việt.",
                    "en": "Use the above results to answer briefly and naturally in English.",
                    "zh": "请根据上面的结果，用简短自然的中文（普通话）回答。",
                    "yue": "請根據上面嘅結果，用簡短自然嘅廣東話回答。",
                }.get(detected_key, "Dùng các kết quả trên để trả lời ngắn gọn.")
                user_content = (
                    f"{transcript}\n\n"
                    f"--- Search results (latest web) ---\n{search_context}\n---\n\n"
                    f"{instr}"
                )
                # Không gửi history khi search — context đã đủ, tránh nhiễu
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                ]
            else:
                messages = [
                    {"role": "system", "content": sys_prompt},
                    *memory.for_llm(),
                    {"role": "user", "content": transcript},
                ]
        with stopwatch("LLM", timings):
            if sgk_pages:
                # SGK pipeline → Groq vision. DeepSeek-chat không nhận image_url.
                answer = llm_vision(client, messages)
                if not (answer or "").strip():
                    print("  ⚠ Vision LLM rỗng — không có retry strategy riêng.")
            else:
                answer = llm(messages)
                if not (answer or "").strip():
                    print("  ⚠ LLM rỗng — retry 1: temp=0.2 + json_object.")
                    try:
                        answer = llm(messages, temperature=0.2)
                    except Exception as e:
                        print(f"  ⚠ Retry 1 fail: {type(e).__name__}: {e}")
                        answer = ""
                if not (answer or "").strip():
                    # DeepSeek json_object mode đôi khi bị kẹt sinh ra ' ' rồi stop.
                    # Bỏ json_mode → model tự do trả text; splitter Python sẽ tự chia segment.
                    print("  ⚠ LLM rỗng — retry 2: bỏ json_object mode.")
                    try:
                        answer = llm(messages, temperature=0.5, json_mode=False)
                    except Exception as e:
                        print(f"  ⚠ Retry 2 fail: {type(e).__name__}: {e}")
                        answer = ""
        if not (answer or "").strip():
            print("  ⚠ LLM vẫn rỗng — dùng câu xin lỗi mặc định.")
            answer = json.dumps(
                {
                    "segments": [
                        {
                            "lang": detected_key if detected_key in LANGS else "vi",
                            "text": "Xin lỗi, mình chưa nghĩ ra câu trả lời. Bạn hỏi lại giúp mình nhé.",
                        }
                    ]
                },
                ensure_ascii=False,
            )

    tts_fallback_lang = intent_lang if intent_handled else detected_key
    segments = parse_tts_segments(answer, tts_fallback_lang)
    display_answer = segments_to_text(segments)
    if not display_answer:
        # Fallback cuối: hiện raw answer để debug + đọc bằng giọng mặc định
        print(f"  ⚠ Parse cho ra rỗng. Raw LLM: {answer[:300]!r}")
        display_answer = (answer or "").strip() or "(không có nội dung)"
        segments = [{"lang": tts_fallback_lang, "text": display_answer}]

    # Ghi vào memory cho mọi turn → lưu DẠNG JSON segments để model thấy
    # history nhất quán với output format (tránh bug DeepSeek json_object trả space).
    memory.add_user(transcript)
    assistant_json = json.dumps(
        {"segments": segments}, ensure_ascii=False, separators=(",", ":")
    )
    memory.add_assistant(assistant_json)

    # Ghi tiến độ học SGK: 1 entry / trang đã hỏi trong lượt này.
    # Chỉ ghi khi (1) đã đi vào SGK path, (2) AI trả lời thực (không phải fallback
    # xin lỗi). Dedupe theo (date, subject, grade, tap, page) — học lại cùng trang
    # trong ngày sẽ không tạo entry trùng.
    if sgk_pages and display_answer and display_answer != "(không có nội dung)":
        for p in sgk_pages:
            try:
                sgk_progress.record(
                    subject=p["subject"],
                    grade=p["grade"],
                    tap=p.get("tap"),
                    page=p["page"],
                    page_label=f"trang {p['page']}",
                )
            except Exception as e:
                print(f"  ⚠ SGK record fail: {type(e).__name__}: {e}")

    print(f"  🤖 AI   : {display_answer}")

    with stopwatch("TTS", timings):
        rendered_segments = synth_tts_segments(
            segments, prefix=f"pipe_{lang_key}_{round_idx}"
        )

    print("  🔊 Đang phát...")
    with stopwatch("Playback", timings):
        for path, _, _ in rendered_segments:
            play_audio(path)

    timings["TOTAL"] = time.perf_counter() - t_total_start
    print(f"\n  Σ TOTAL    {timings['TOTAL']:6.2f}s")
    return timings


def summarize(history: list[dict]) -> None:
    if not history:
        return
    print_header("LATENCY SUMMARY")
    labels = ["Record", "STT", "Intent", "LLM", "TTS", "Playback", "TOTAL"]
    n = len(history)
    print(f"  {'Step':<10} {'avg':>8} {'min':>8} {'max':>8}   ({n} lượt)")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for k in labels:
        vals = [h[k] for h in history if k in h]
        if not vals:
            continue
        avg = statistics.mean(vals)
        print(f"  {k:<10} {avg:8.2f} {min(vals):8.2f} {max(vals):8.2f}")

    user_perceived = [
        h.get("STT", 0) + h.get("Intent", 0) + h.get("LLM", 0) + h.get("TTS", 0)
        for h in history
    ]
    if user_perceived:
        print(
            f"\n  → Thời gian chờ AI (STT+Intent+LLM+TTS): "
            f"trung bình {statistics.mean(user_perceived):.2f}s"
        )


def run_pipeline(args, profile: dict) -> None:
    require_keys("GROQ_API_KEY", "DEEPSEEK_API_KEY")
    client = Groq(api_key=GROQ_API_KEY)

    print(
        f"  👤 Profile: name={profile.get('name')!r}  lang={profile.get('language')!r}"
    )
    if args.loop:
        print("  🔁 Loop mode: Ctrl+C để thoát giữa chừng.")

    # Memory hội thoại — persist qua JSON, tự load lại khi restart
    memory = ConversationMemory(CHAT_HISTORY_PATH, max_turns=CHAT_MAX_TURNS)
    if args.forget:
        memory.clear()
        print("  🧠 Memory đã được xóa theo --forget.")
    elif len(memory) > 0:
        print(f"  🧠 Memory đã load: {memory.summary()}")
    else:
        print(f"  🧠 Memory rỗng. File: {CHAT_HISTORY_PATH}")

    # Tiến độ học SGK — persist qua JSON, để AI nhắc lại bài cũ + gợi ý bài kế.
    sgk_progress = SGKProgress(SGK_PROGRESS_PATH)
    if args.forget:
        sgk_progress.clear()
        print("  📚 SGK progress đã được xóa theo --forget.")
    elif len(sgk_progress) > 0:
        prev = sgk_progress.latest_before_today()
        if prev:
            print(
                f"  📚 SGK progress: {len(sgk_progress)} bài trên đĩa. "
                f"Buổi trước ({prev.get('date')}): {prev.get('subject')} "
                f"lớp {prev.get('grade')} trang {prev.get('page')}."
            )
        else:
            print(f"  📚 SGK progress: {len(sgk_progress)} bài (tất cả hôm nay).")
    else:
        print(f"  📚 SGK progress rỗng. File: {SGK_PROGRESS_PATH}")

    alarm_mgr = AlarmManager(
        path=ALARMS_PATH,
        on_trigger=make_alarm_trigger_callback(profile),
        check_interval=30.0,
    )
    ensure_alarm_tone(ALARM_TONE)
    alarm_mgr.start()
    print(f"  📋 AlarmManager start. Alarms file: {ALARMS_PATH}")

    history: list[dict] = []
    try:
        if args.loop:
            i = 0
            while True:
                i += 1
                result = one_round(
                    client, args.lang, args.seconds, i, alarm_mgr, profile, memory,
                    sgk_progress,
                    use_vad=not args.no_vad,
                )
                if result:
                    history.append(result)
        else:
            for i in range(1, args.rounds + 1):
                result = one_round(
                    client, args.lang, args.seconds, i, alarm_mgr, profile, memory,
                    sgk_progress,
                    use_vad=not args.no_vad,
                )
                if result:
                    history.append(result)
    except KeyboardInterrupt:
        print("\n  ⏹ Dừng theo yêu cầu.")
    finally:
        alarm_mgr.stop()

    summarize(history)
    print("\n✅ Hoàn tất pipeline test.")


# =============================================================
#  VOICES — so sánh giọng Edge TTS vi-VN và zh-HK
# =============================================================
VOICE_SAMPLES: dict[str, str] = {
    "vi-VN": "Xin chào! Tớ là người bạn AI của bạn. Hôm nay mình cùng học cho thật vui nhé!",
    "zh-HK": "你好啊！我係你嘅AI好朋友嚟㗎。今日我哋一齊玩住學嘢啦！",
}

VOICE_RECOMMENDED: dict[str, str] = {
    "vi-VN": "vi-VN-NamMinhNeural",
    "zh-HK": "zh-HK-HiuMaanNeural",
}


def _fetch_voices(locale: str) -> list[dict]:
    async def _run() -> list[dict]:
        all_voices = await edge_tts.list_voices()
        return [v for v in all_voices if v["Locale"] == locale]

    return asyncio.run(_run())


def _synth_voice(voice: str, text: str, out: Path) -> Path:
    async def _run() -> None:
        await edge_tts.Communicate(text=text, voice=voice).save(str(out))

    asyncio.run(_run())
    return out


def _voices_for_locale(locale: str, play: bool) -> dict[str, int]:
    print_header(f"VOICES • {locale}")
    voices = _fetch_voices(locale)
    if not voices:
        print(f"  ⚠ Không tìm thấy giọng cho {locale}.")
        return {}

    sample = VOICE_SAMPLES[locale]
    print(f"  Tìm thấy {len(voices)} giọng.")
    print(f"  Câu mẫu: {sample}")

    ratings: dict[str, int] = {}
    for v in voices:
        name = v["ShortName"]
        gender = v["Gender"]
        marker = " ★" if name == VOICE_RECOMMENDED.get(locale) else ""
        print(f"\n  ▶ {name}  ({gender}){marker}")

        out = TMP_DIR / f"voice_{name}.mp3"
        with stopwatch("synth"):
            _synth_voice(name, sample, out)
        print(f"     💾 {out.stat().st_size / 1024:.1f} KB")

        if not play:
            continue

        print("     🔊 phát...")
        play_audio(out)
        score = input("     ↪ Chấm 1-5 (ENTER = bỏ qua, q = thoát): ").strip().lower()
        if score == "q":
            raise KeyboardInterrupt
        if score in {"1", "2", "3", "4", "5"}:
            ratings[name] = int(score)
    return ratings


def run_voices(args) -> None:
    play = not args.no_play
    targets = ["vi-VN", "zh-HK"] if args.locale == "both" else [args.locale]

    all_ratings: dict[str, int] = {}
    try:
        for locale in targets:
            all_ratings.update(_voices_for_locale(locale, play))
    except KeyboardInterrupt:
        print("\n\n  ⏹ Đã dừng theo yêu cầu.")

    print_header("KẾT LUẬN")
    if play and all_ratings:
        ranked = sorted(all_ratings.items(), key=lambda x: -x[1])
        print(f"  {'Voice':<35} {'Điểm':>5}")
        print(f"  {'-' * 35} {'-' * 5}")
        for name, score in ranked:
            bar = "★" * score + "·" * (5 - score)
            print(f"  {name:<35} {score:>2}/5  {bar}")
        top_name, top_score = ranked[0]
        print(f"\n  🏆 Tự nhiên nhất theo bạn: {top_name}  ({top_score}/5)")
    elif play:
        print("  (Chưa đánh giá giọng nào.)")

    print("\n  Khuyến nghị mặc định (★):")
    for locale, voice in VOICE_RECOMMENDED.items():
        print(f"    {locale}  →  {voice}")

    if not play:
        n = len(list(TMP_DIR.glob("voice_*.mp3")))
        print(f"\n✅ Đã render {n} file MP3 trong {TMP_DIR}")


# =============================================================
#  ALARM TESTS — 3 suite: manager / parser / live
# =============================================================
ALARM_TEST_DIR = TMP_DIR / "_alarm_test"
ALARM_TEST_DIR.mkdir(exist_ok=True)
ALARM_TEST_PATH = ALARM_TEST_DIR / "alarms.json"
ALARM_TEST_TONE = ALARM_TEST_DIR / "alarm_tone.wav"


def _reset_alarm_file() -> None:
    if ALARM_TEST_PATH.exists():
        ALARM_TEST_PATH.unlink()


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"  ❌ FAIL: {msg}")
        sys.exit(1)
    print(f"  ✅ {msg}")


def _suite_manager() -> None:
    print_header("ALARM SUITE • MANAGER (CRUD + persistence)")
    _reset_alarm_file()
    fired: list[Alarm] = []

    am = AlarmManager(
        path=ALARM_TEST_PATH,
        on_trigger=lambda a: fired.append(a),
        check_interval=0.5,
    )

    t1 = datetime.now() + timedelta(hours=8)
    a1 = am.add(t1, [], "Uống nước", language="vi")
    _assert(len(am.list_active()) == 1, "thêm 1 alarm one-shot")
    _assert(not a1.is_repeating, "one-shot không lặp")
    _assert(a1.language == "vi", "language='vi' được lưu")

    t2 = datetime.now() + timedelta(days=1)
    a2 = am.add(
        t2.replace(hour=6, minute=30),
        [0, 1, 2, 3, 4],
        "Wake up for school",
        language="en",
    )
    _assert(a2.is_repeating, "repeating có weekdays")
    _assert(a2.language == "en", "language='en' được lưu")
    _assert(len(am.list_active()) == 2, "tổng 2 alarm")

    _assert(
        a1.format_announce("Anh").startswith("Đến giờ rồi Anh"),
        "format_announce VI đúng",
    )
    _assert(
        a2.format_announce("Anh").startswith("Wake up, Anh"), "format_announce EN đúng"
    )

    text_vi = am.format_for_prompt("vi")
    text_en = am.format_for_prompt("en")
    _assert("T2,T3,T4,T5,T6" in text_vi, "weekdays hiển thị tiếng Việt (T2..T6)")
    _assert("Mon,Tue,Wed,Thu,Fri" in text_en, "weekdays hiển thị English")

    am2 = AlarmManager(path=ALARM_TEST_PATH, on_trigger=lambda a: None)
    _assert(len(am2.list_active()) == 2, "load lại từ file vẫn còn 2")
    loaded = am2.list_active()
    _assert(loaded[1].label == "Wake up for school", "label persist đúng")
    _assert(loaded[1].language == "en", "language persist đúng")

    n = am.cancel_by_match("uống")
    _assert(n == 1, "cancel_by_match xóa đúng 1")
    _assert(len(am.list_active()) == 1, "còn 1 alarm sau khi xóa")

    n = am.cancel_all()
    _assert(n == 1, "cancel_all xóa 1 còn lại")
    _assert(len(am.list_active()) == 0, "không còn alarm nào")

    now = datetime.now()
    rep = Alarm(
        id="x",
        trigger_at=(now - timedelta(hours=1)).replace(microsecond=0).isoformat(),
        weekdays=[(now.weekday() + 1) % 7],
        label="t",
    )
    nxt = rep.next_occurrence(now)
    _assert(nxt > now, "next_occurrence > now")
    _assert(
        nxt.weekday() == (now.weekday() + 1) % 7, "next_occurrence rơi đúng weekday"
    )

    print("\n  ⏲  Đặt alarm sau 1.5s và đợi thread fire...")
    _reset_alarm_file()
    am3 = AlarmManager(
        path=ALARM_TEST_PATH,
        on_trigger=lambda a: fired.append(a),
        check_interval=0.4,
    )
    am3.start()
    am3.add(datetime.now() + timedelta(seconds=1.5), [], "ping")
    time.sleep(3.0)
    am3.stop()
    _assert(any(f.label == "ping" for f in fired), "background thread đã fire")
    print("\n  ✅ Suite manager PASS")


PARSER_CASES: list[tuple[str, str, str]] = [
    # Vietnamese
    ("đặt báo thức 7 giờ sáng mai", "alarm_create", "vi"),
    ("nhắc tao sau 30 phút uống nước", "alarm_create", "vi"),
    ("báo thức 6 giờ 30 thứ 2 đến thứ 6", "alarm_create", "vi"),
    ("hủy tất cả báo thức", "alarm_cancel_all", "vi"),
    ("hủy báo thức uống nước", "alarm_cancel_one", "vi"),
    ("tao có báo thức nào không?", "alarm_list", "vi"),
    ("tại sao trời lại có mưa?", "chat", "vi"),
    # English
    ("set an alarm for 7 am tomorrow", "alarm_create", "en"),
    ("remind me in 15 minutes to drink water", "alarm_create", "en"),
    ("set alarm 6:30 Monday to Friday", "alarm_create", "en"),
    ("cancel all alarms", "alarm_cancel_all", "en"),
    ("what alarms do I have", "alarm_list", "en"),
    ("why does it rain", "chat", "en"),
    # Mandarin
    ("设置明天早上7点闹钟", "alarm_create", "zh"),
    ("30分钟后提醒我喝水", "alarm_create", "zh"),
    ("取消所有闹钟", "alarm_cancel_all", "zh"),
    ("我有什么闹钟", "alarm_list", "zh"),
    ("为什么会下雨", "chat", "zh"),
    # Cantonese
    ("設定聽日朝早7點鬧鐘", "alarm_create", "yue"),
    ("30分鐘後提我飲水", "alarm_create", "yue"),
    ("取消晒所有鬧鐘", "alarm_cancel_all", "yue"),
    ("我有咩鬧鐘", "alarm_list", "yue"),
    ("點解會落雨", "chat", "yue"),
]


def _suite_parser() -> None:
    require_keys("DEEPSEEK_API_KEY")
    print_header("ALARM SUITE • PARSER (intent + language detection)")

    correct_intent = 0
    correct_lang = 0
    for transcript, expected_intent, expected_lang in PARSER_CASES:
        try:
            res = parse_intent(DEEPSEEK_API_KEY, transcript, "(no alarms)")
        except Exception as e:
            print(f"  ❌ {transcript!r} → exception: {e}")
            continue

        got_intent = res.get("intent")
        got_lang = res.get("language")
        intent_ok = got_intent == expected_intent
        lang_ok = got_lang == expected_lang

        marker = "✅" if (intent_ok and lang_ok) else "❌"
        print(f"  {marker} [{expected_lang}] {transcript!r}")
        extra = ""
        if got_intent == "alarm_create":
            extra = (
                f"  → at={res.get('trigger_at')}  "
                f"wd={res.get('weekdays')}  label={res.get('label')!r}"
            )
        elif got_intent == "alarm_cancel_one":
            extra = f"  → match={res.get('match')!r}"
        print(
            f"     expected: intent={expected_intent:<18} lang={expected_lang}\n"
            f"     got     : intent={got_intent:<18} lang={got_lang}{extra}"
        )

        if intent_ok:
            correct_intent += 1
        if lang_ok:
            correct_lang += 1

    n = len(PARSER_CASES)
    print(f"\n  Intent  : {correct_intent}/{n} đúng ({correct_intent / n * 100:.0f}%)")
    print(f"  Language: {correct_lang}/{n} đúng ({correct_lang / n * 100:.0f}%)")


def _suite_live(delay_seconds: int) -> None:
    print_header(f"ALARM SUITE • LIVE (alarm sau {delay_seconds}s)")
    _reset_alarm_file()
    ensure_alarm_tone(ALARM_TEST_TONE)

    def on_trigger(alarm: Alarm) -> None:
        print(f"\n  🔔 FIRE: {alarm.human_summary()}")
        print("  🎵 Phát chime ramp-up...")
        play_audio(ALARM_TEST_TONE)
        print(f'  💬 [Bot sẽ đọc]: "Đến giờ rồi bạn ơi! {alarm.label}."')

    am = AlarmManager(
        path=ALARM_TEST_PATH,
        on_trigger=on_trigger,
        check_interval=1.0,
    )
    am.start()
    am.add(datetime.now() + timedelta(seconds=delay_seconds), [], "Uống nước nhé")
    print(f"  ⏲  Đang đợi tới {delay_seconds}s... (Ctrl+C để dừng)")
    try:
        time.sleep(delay_seconds + 8)
    except KeyboardInterrupt:
        print("\n  ⏹ Dừng theo yêu cầu.")
    am.stop()

    if ALARM_TEST_PATH.exists():
        data = json.loads(ALARM_TEST_PATH.read_text(encoding="utf-8"))
        print(f"\n  alarms.json sau test: {len(data)} alarm")
        for a in data:
            print(f"     active={a['active']}  {a['label']}")


def run_alarm(args) -> None:
    suite = args.suite
    if suite in ("manager", "all"):
        _suite_manager()
    if suite in ("parser", "all"):
        _suite_parser()
    if suite in ("live", "all"):
        _suite_live(args.delay)
    print("\n✅ Hoàn tất alarm test.")


# =============================================================
#  MAIN DISPATCHER
# =============================================================
def main() -> None:
    profile = load_profile()
    default_lang = profile.get("language", "vi")

    ap = argparse.ArgumentParser(
        description="AI Voice Assistant — pipeline / voices / alarm tests",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--voices", action="store_true", help="So sánh giọng Edge TTS (vi-VN + zh-HK)."
    )
    mode.add_argument("--alarm", action="store_true", help="Chạy alarm test suites.")

    # Pipeline options
    ap.add_argument(
        "--lang",
        choices=ALL_LANG_CODES + ["auto"],
        default=default_lang,
        help=f"Pipeline: mã ngôn ngữ hoặc 'auto'. Mặc định = profile ({default_lang}).",
    )
    ap.add_argument(
        "--seconds",
        type=int,
        default=15,
        help="Pipeline: thời lượng ghi âm (với VAD: trần tối đa; không VAD: cố định).",
    )
    ap.add_argument("--rounds", type=int, default=1, help="Pipeline: số lượt.")
    ap.add_argument("--loop", action="store_true", help="Pipeline: chạy liên tục.")
    ap.add_argument(
        "--no-vad",
        action="store_true",
        help="Pipeline: tắt VAD, ghi cố định --seconds (mặc định bật VAD tự dừng).",
    )
    ap.add_argument(
        "--forget",
        action="store_true",
        help="Pipeline: xóa lịch sử hội thoại trước khi chạy.",
    )

    # Voices options
    ap.add_argument(
        "--locale",
        choices=["vi-VN", "zh-HK", "both"],
        default="both",
        help="Voices: chọn locale.",
    )
    ap.add_argument("--no-play", action="store_true", help="Voices: không phát ra loa.")

    # Alarm options
    ap.add_argument(
        "--suite",
        choices=["manager", "parser", "live", "all"],
        default="all",
        help="Alarm: chọn suite test.",
    )
    ap.add_argument(
        "--in",
        dest="delay",
        type=int,
        default=8,
        help="Alarm live: kêu sau N giây (mặc định 8).",
    )

    args = ap.parse_args()

    if args.voices:
        run_voices(args)
    elif args.alarm:
        run_alarm(args)
    else:
        run_pipeline(args, profile)


if __name__ == "__main__":
    main()
