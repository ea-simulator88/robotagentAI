"""
Phân loại ý định báo thức ĐA NGÔN NGỮ (vi / en / zh / yue) bằng DeepSeek JSON mode.

Đầu vào: transcript của Whisper (có thể là bất kỳ trong 4 ngôn ngữ).
Đầu ra: dict {
    intent: "alarm_create" | "alarm_cancel_one" | "alarm_cancel_all" | "alarm_list" | "chat",
    trigger_at: ISO datetime (alarm_create),
    weekdays:   [0..6] danh sách thứ trong tuần (rỗng = 1 lần),
    label:      mô tả việc cần nhắc,
    match:      keyword khớp label (cancel_one),
    language:   "vi" | "en" | "zh" | "yue" — ngôn ngữ phát hiện của câu nói,
}

Cũng cung cấp build_chat_system_prompt() để inject datetime + alarm list
vào system prompt của LLM theo đúng ngôn ngữ người dùng.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import requests

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

WEEKDAY_VN = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]


# =============================================================
#  Intent classifier prompt (multilingual)
# =============================================================
def _build_system_prompt(
    now: datetime,
    active_alarms_text: str,
    user_city: str = "Ho Chi Minh City",
) -> str:
    today_vn = WEEKDAY_VN[now.weekday()]
    return f"""You are an intent classifier for a multilingual voice assistant.
The user may speak Vietnamese (vi), English (en), Mandarin Chinese (zh), or Cantonese (yue).
Return ONE JSON object only — no markdown, no commentary.

Current datetime: {now.isoformat(timespec='seconds')}  (weekday: {today_vn})
User's default city: {user_city}
Active alarms:
{active_alarms_text}

INTENTS:
  - "alarm_create"       : user wants to set an alarm / reminder
  - "alarm_cancel_one"   : cancel one alarm by keyword
  - "alarm_cancel_all"   : cancel all alarms
  - "alarm_list"         : list / ask about current alarms
  - "web_search"         : user asks for REAL-TIME / CURRENT info you don't know:
                           weather, news, sports scores, stock prices, current events,
                           "today/now/this week" facts, anything time-sensitive.
  - "chat"               : anything else (general conversation, jokes, explanations,
                           translations, math, common knowledge from training data).

SCHEMA:
{{
  "intent":     "alarm_create | alarm_cancel_one | alarm_cancel_all | alarm_list | web_search | chat",
  "trigger_at": "ISO 8601 datetime (for alarm_create only, MUST be future)",
  "weekdays":   [0..6]  (0=Mon ... 6=Sun; empty = one-shot),
  "label":      "what to remind, kept in the user's language",
  "match":      "keyword to match an existing alarm (cancel_one only)",
  "query":      "search query in user's language (web_search only). For weather,
                 format like 'thời tiết <TP> <ngày>'. Resolve relative dates to
                 absolute: 'mai' → '<tomorrow date>', 'tuần này' → '<this week>'.",
  "language":   "vi | en | zh | yue — the language the user spoke in"
}}

TIME-OF-DAY conventions:
  vi : sáng = 06-11, trưa = 11-13, chiều = 13-18, tối = 18-22, đêm/khuya = 22-05
  en : morning = 06-11, noon = 11-13, afternoon = 13-18, evening = 18-22, night = 22-05
  zh : 早上/上午 = 06-11, 中午 = 11-13, 下午 = 13-18, 晚上 = 18-22, 半夜 = 22-05
  yue: 朝早 = 06-11, 晏晝 = 11-13, 下晝 = 13-18, 夜晚 = 18-22, 半夜 = 22-05

WEEKDAYS:
  vi : "thứ 2..thứ 7, chủ nhật"        → 0..6
  en : "Mon..Sun" / "Monday..Sunday"   → 0..6
  zh : "周一..周日" / "星期一..星期日"  → 0..6
  yue: "星期一..星期日"                → 0..6

  Common ranges:
  - "thứ 2 đến thứ 6" / "Monday to Friday" / "周一到周五" / "星期一至五" → [0,1,2,3,4]
  - "thứ 7 và CN" / "weekends" / "周末" / "星期六日"                     → [5,6]
  - "mỗi ngày" / "every day" / "每天" / "每日"                            → [0,1,2,3,4,5,6]
  - One-shot (no repeat mentioned)                                       → []

RELATIVE TIME:
  - "mai" / "tomorrow" / "明天" / "聽日"          → current date + 1 day
  - "sau N phút" / "in N minutes" / "N分钟后" / "N分鐘後" → now + N minutes
  - "X giờ sáng" if X<6 → next day; otherwise today (or tomorrow if already past)

IMPORTANT:
  - "language" MUST reflect what the user actually spoke.
  - To distinguish Mandarin (zh) vs Cantonese (yue):
    yue uses 嘅 咁 嚟 喺 唔 啲 咗 佢 嗰 冇 嘢 咩 點解 啦 㗎 — Mandarin doesn't.
    If transcript has ANY of these particles → language is "yue", NOT "zh".
  - trigger_at MUST be strictly after {now.isoformat(timespec='seconds')}.
  - If user mentions an hour but it has already passed today → schedule for tomorrow.

EXAMPLES:
  "đặt báo thức 7 giờ sáng mai uống thuốc"
    → {{"intent":"alarm_create", "trigger_at":"<tomorrow>T07:00:00", "weekdays":[],
        "label":"uống thuốc", "language":"vi"}}

  "set an alarm for 7am tomorrow to take medicine"
    → {{"intent":"alarm_create", "trigger_at":"<tomorrow>T07:00:00", "weekdays":[],
        "label":"take medicine", "language":"en"}}

  "设置明天早上7点闹钟，提醒我吃药"
    → {{"intent":"alarm_create", "trigger_at":"<tomorrow>T07:00:00", "weekdays":[],
        "label":"吃药", "language":"zh"}}

  "設定聽日朝早7點鬧鐘，提我食藥"
    → {{"intent":"alarm_create", "trigger_at":"<tomorrow>T07:00:00", "weekdays":[],
        "label":"食藥", "language":"yue"}}

  "báo thức 6 giờ 30 thứ 2 đến thứ 6"
    → {{"intent":"alarm_create", "trigger_at":"<next mon>T06:30:00", "weekdays":[0,1,2,3,4],
        "label":"", "language":"vi"}}

  "hủy hết báo thức" / "cancel all alarms" / "取消所有闹钟" / "取消晒所有鬧鐘"
    → {{"intent":"alarm_cancel_all", "language":"<detected>"}}

  "hủy báo thức uống nước" / "cancel the water alarm" / "取消喝水的闹钟" / "取消飲水嗰個鬧鐘"
    → {{"intent":"alarm_cancel_one", "match":"<keyword in user language>", "language":"<detected>"}}

  "tao có báo thức nào không?" / "what alarms do I have?" / "我有什么闹钟" / "我有咩鬧鐘"
    → {{"intent":"alarm_list", "language":"<detected>"}}

  "tại sao trời mưa" / "why does it rain" / "为什么下雨" / "點解會落雨"
    → {{"intent":"chat", "language":"<detected>"}}

  "thời tiết mai thế nào" / "what's the weather tomorrow"
    → {{"intent":"web_search", "query":"thời tiết ngày <tomorrow date> tại <user's city>", "language":"vi"}}

  "tin tức hôm nay" / "today's news"
    → {{"intent":"web_search", "query":"tin tức Việt Nam <today date>", "language":"vi"}}

  "giá bitcoin bây giờ" / "current bitcoin price"
    → {{"intent":"web_search", "query":"giá bitcoin USD hiện tại", "language":"vi"}}

  "Messi ghi bao nhiêu bàn tuần này"
    → {{"intent":"web_search", "query":"Messi bàn thắng tuần <this week>", "language":"vi"}}
"""


def parse_intent(
    api_key: str,
    transcript: str,
    active_alarms_text: str,
    now: datetime | None = None,
    user_city: str = "Ho Chi Minh City",
) -> dict[str, Any]:
    now = now or datetime.now()
    sys_prompt = _build_system_prompt(now, active_alarms_text, user_city)
    r = requests.post(
        DEEPSEEK_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": transcript},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 256,
        },
        timeout=30,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Parser trả JSON không hợp lệ: {content}") from e

    parsed.setdefault("intent", "chat")
    parsed.setdefault("weekdays", [])
    parsed.setdefault("label", "")
    parsed.setdefault("match", "")
    parsed.setdefault("trigger_at", "")
    parsed.setdefault("query", "")
    parsed.setdefault("language", "vi")
    if parsed["language"] not in {"vi", "en", "zh", "yue"}:
        parsed["language"] = "vi"
    return parsed


# =============================================================
#  Chat system prompt — inject datetime + alarm list theo ngôn ngữ
# =============================================================
WEEKDAY_DISPLAY = {
    "vi":  ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"],
    "en":  ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "zh":  ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"],
    "yue": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"],
}

# Khuôn nội dung "context block" theo từng ngôn ngữ
CONTEXT_TEMPLATES = {
    "vi": (
        "\n\nBối cảnh hiện tại:\n"
        "- Hiện tại là {when}.\n"
        "- Người đang nói chuyện với bạn: {name}.\n"
        "- Báo thức đang đặt:\n{alarms}\n\n"
        "Khi được hỏi 'mấy giờ rồi' / 'hôm nay thứ mấy' → trả lời dựa trên bối cảnh trên. "
        "Nếu thấy báo thức trước 7 giờ sáng → gợi ý nhẹ 'ngủ sớm nhé'."
    ),
    "en": (
        "\n\nCurrent context:\n"
        "- It is now {when}.\n"
        "- You're talking to: {name}.\n"
        "- Active alarms:\n{alarms}\n\n"
        "If asked about time or day, answer from the context. "
        "If any alarm is before 7am, gently suggest going to bed early."
    ),
    "zh": (
        "\n\n当前情况：\n"
        "- 现在是 {when}。\n"
        "- 正在和你说话的是：{name}。\n"
        "- 当前的闹钟：\n{alarms}\n\n"
        "如果被问几点了 / 今天星期几，请根据上面的情况回答。"
        "如果有闹钟设在早上7点之前，请轻轻提醒早点睡觉。"
    ),
    "yue": (
        "\n\n而家嘅情況：\n"
        "- 而家係 {when}。\n"
        "- 同你傾偈嘅係：{name}。\n"
        "- 而家嘅鬧鐘：\n{alarms}\n\n"
        "如果被問幾點 / 今日星期幾，根據上面嘅資料答。"
        "如果有鬧鐘喺早上7點之前，輕輕提醒佢早啲訓覺。"
    ),
}

SEGMENTED_RESPONSE_INSTRUCTIONS = """
Output format: reply with ONE JSON object exactly matching:
{"segments":[{"lang":"vi|en|zh|yue","text":"..."}]}

Rule: each segment's "lang" picks the TTS voice. If your reply mixes languages,
put each language in its OWN segment. Foreign words/phrases inside an
explanation MUST be a separate segment — otherwise they get the wrong accent.

NO PINYIN / NO JYUTPING — VERY IMPORTANT:
The TTS engine CANNOT pronounce romanized Chinese (Wǒ hěn lèi, Nǐ hǎo) — it
comes out silent or garbled. So:
- NEVER include Pinyin in any segment. NEVER include Jyutping.
- For Chinese terms, give ONLY the Han characters in a "zh" segment, then
  explain in Vietnamese (or the user's language).
- If you really must hint at pronunciation, describe it briefly in the
  user's language (e.g. "đọc gần như 'ủa hân lây' trong tiếng Việt") —
  but it's usually better to skip pronunciation entirely.

SINO-VIETNAMESE HINT:
Vietnamese speakers often pronounce Chinese words using their Hán-Việt reading.
When the transcript contains "thái dương / tài dương" → user means 太阳 (sun);
"nỉ hảo / ni hao" → 你好; "tạ tạ" → 谢谢; etc. Treat these as the Chinese term
and put the Han characters in a "zh" segment.

Examples:
- VI explains EN: {"segments":[
    {"lang":"vi","text":"Câu này nè:"},
    {"lang":"en","text":"What's your name?"},
    {"lang":"vi","text":"nghĩa là Bạn tên gì."}
  ]}
- VI teaches ZH (NO PINYIN): {"segments":[
    {"lang":"vi","text":"Tiếng Hoa nói:"},
    {"lang":"zh","text":"我很累。"},
    {"lang":"vi","text":"nghĩa là tôi rất mệt đó bạn."}
  ]}
- User asks "thái dương tiếng Việt là gì": {"segments":[
    {"lang":"zh","text":"太阳"},
    {"lang":"vi","text":"trong tiếng Việt là mặt trời đó bạn."}
  ]}
- Pure VI: {"segments":[{"lang":"vi","text":"Trời mưa vì hơi nước ngưng tụ đó bạn."}]}
"""


def build_chat_system_prompt(
    base_prompt: str,
    now: datetime,
    active_alarms_text: str,
    user_name: str = "bạn",
    language: str = "vi",
) -> str:
    """Inject datetime + danh sách báo thức vào system prompt theo `language`.
    Đặt rule JSON segments LÊN ĐẦU để mô hình không bỏ sót giữa biển hướng dẫn."""
    lang = language if language in CONTEXT_TEMPLATES else "vi"
    weekday_name = WEEKDAY_DISPLAY[lang][now.weekday()]
    when = now.strftime("%H:%M, ") + weekday_name + now.strftime(", %d/%m/%Y")
    block = CONTEXT_TEMPLATES[lang].format(
        when=when,
        name=user_name,
        alarms=active_alarms_text,
    )
    return SEGMENTED_RESPONSE_INSTRUCTIONS + "\n" + base_prompt + block
