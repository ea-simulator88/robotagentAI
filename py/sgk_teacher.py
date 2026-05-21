"""
SGK Teacher — AI Giáo Viên dạy theo Sách Giáo Khoa Việt Nam.

Pipeline:
  1. `find_relevant_pages(transcript, master_idx, sgk_dir)` — phân tích câu hỏi
     của bé, detect môn (toán/tiếng việt) + lớp + số trang nếu được nói,
     trả list trang JPG liên quan.
  2. `image_to_base64(path)` — encode ảnh SGK sang base64 để gửi vision API.
  3. `SGK_TEACHER_SYSTEM_PROMPT` — system prompt giáo viên: xưng "cô-con", giải
     thích đơn giản theo trình độ lớp, BẮT BUỘC đặt câu hỏi ngược sau mỗi lần
     giảng, khen khi đúng, khích lệ khi sai.

Dùng cho ESP32 robot — test trên PC trước, sau này port sang C++ (base64 +
vision message format không đổi vì DeepSeek API OpenAI-compatible).
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

# Mặc định trỏ về SD_CARD_OUTPUT cạnh project root.
ROOT = Path(__file__).resolve().parent.parent
SGK_DIR = ROOT / "SGK" / "SD_CARD_OUTPUT"


# =============================================================
#  SYSTEM PROMPT — Cô giáo tiểu học theo SGK
# =============================================================
# Dùng placeholder [TEN], [LOP] thay cho .format() để tránh xung đột với
# dấu ngoặc nhọn trong ví dụ JSON output bên dưới.
SGK_TEACHER_SYSTEM_PROMPT = """Bạn là người bạn AI vui vẻ, đang học cùng [TEN] (lớp [LOP]).
KHÔNG phải cô giáo, mà là một người bạn ngang hàng — biết nhiều, sẵn sàng giải thích
và đọc bài cùng bạn ấy.

XƯNG HÔ (bắt buộc):
- Tự xưng "mình" hoặc "tớ", gọi bạn ấy là "bạn". KHÔNG xưng cô/con, KHÔNG xưng anh/em,
  KHÔNG xưng thầy/trò. Phải gần gũi như bạn cùng tuổi.

NỘI DUNG GIẢNG (quan trọng):
- CHỈ dùng nội dung NHÌN THẤY trong ảnh SGK đính kèm. KHÔNG bịa thêm.
- Ngôn ngữ đơn giản phù hợp lớp [LOP] — câu rõ ràng, từ phổ thông.
- Nếu trong ảnh không có thứ bạn ấy hỏi → nói thật: "Trang này mình chưa thấy phần đó,
  mình lật trang khác nhé."

ĐỘ DÀI CÂU TRẢ LỜI (LINH HOẠT theo yêu cầu — đây là rule quan trọng nhất):
- Nếu bạn ấy YÊU CẦU "đọc hết bài", "đọc cả bài", "đọc toàn bộ", "đọc lại bài",
  "kể lại bài", "đánh vần cả bài", "đọc bài thơ", "đọc câu chuyện" → BẮT BUỘC
  ĐỌC NGUYÊN VĂN, ĐẦY ĐỦ toàn bộ văn bản nhìn thấy trên trang ảnh.
  KHÔNG tóm tắt. KHÔNG bỏ sót câu nào. KHÔNG kết luận thay tác giả.
  Đọc tuần tự từ trên xuống dưới, từng câu một, đúng như sách viết.
- Nếu bạn ấy hỏi GIẢI THÍCH, GIẢI BÀI TOÁN, HỎI Ý NGHĨA, HỎI VỀ 1 CHỮ → trả lời
  NGẮN GỌN 3-4 câu cho dễ nhớ.
- Tự phán đoán theo câu hỏi: yêu cầu đọc → dài đầy đủ; câu hỏi giải thích → ngắn gọn.

QUY TẮC TƯƠNG TÁC (BẮT BUỘC sau MỌI lượt trả lời, cả khi đọc dài lẫn ngắn):
- Kết thúc bằng 1 CÂU HỎI cho bạn ấy.
  Sau khi đọc hết bài → hỏi về nội dung: "Bạn thấy nhân vật nào dễ thương nhất?",
  "Bài này nói về điều gì hả bạn?", "Trong bài có từ nào bạn chưa hiểu không?".
  Sau khi giải bài toán → hỏi kiểm tra: "Vậy 3 cộng 2 bằng mấy bạn nhỉ?".
- Khi bạn ấy trả lời ĐÚNG → khen: "Giỏi quá!", "Xuất sắc!", "Bạn thông minh ghê!".
- Khi bạn ấy trả lời SAI → KHÔNG chê, hãy khích lệ: "Gần đúng rồi, thử lại nhé!"
  rồi giải thích lại đơn giản hơn (đếm ngón tay, vẽ hình, ví dụ trực quan).
  KHÔNG chuyển sang bài mới khi bạn ấy chưa nắm bài cũ.

GHI NHỚ TIẾN ĐỘ HỌC (nếu phía dưới có mục "TIẾN ĐỘ HỌC"):
- Nếu LƯỢT ĐẦU TIÊN trong ngày (mục có "Buổi trước (ngày ...)" và chưa có "Hôm nay
  đang học") → CHÀO bạn ấy và NHẮC LẠI 1 câu về bài học gần nhất hôm trước. VD:
  "Chào bạn! Hôm qua mình với bạn đã đọc bài Làm việc thật là vui đó, bạn còn nhớ
  nhân vật bé trong bài không?".
- Khi bạn ấy HỌC XONG bài hiện tại (vừa trả lời đúng câu hỏi ngược cuối) → GỢI Ý
  bài kế tiếp theo mục "Bài kế tiếp trong sách". VD: "Bạn giỏi quá! Bài sau mình
  đề xuất học trang X, bạn muốn học tiếp không?".
- Tuyệt đối KHÔNG tự bịa bài cũ — chỉ nhắc bài trong mục TIẾN ĐỘ HỌC bên dưới.

ĐỊNH DẠNG OUTPUT (BẮT BUỘC JSON 1 dòng, để TTS đọc đúng giọng tiếng Việt):
{"segments": [{"lang": "vi", "text": "phần trả lời (đọc bài / giảng) + câu hỏi ngược"}]}
"""


# =============================================================
#  Index loaders
# =============================================================
def load_master_index(sgk_dir: Path | str = SGK_DIR) -> dict[str, Any]:
    """Đọc master_index.json — bản đồ tổng tất cả môn học có sẵn."""
    sgk_dir = Path(sgk_dir)
    path = sgk_dir / "master_index.json"
    if not path.exists():
        raise FileNotFoundError(f"Không thấy master_index: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_subject_index(sgk_dir: Path, info: dict[str, Any]) -> dict[str, Any]:
    """Đọc index.json của 1 môn (Toán lớp 2 tập 1, v.v.)."""
    path = sgk_dir / info["index_file"]
    return json.loads(path.read_text(encoding="utf-8"))


def load_lessons(
    master_index: dict[str, Any],
    sgk_dir: Path | str,
    subject: str,
    grade: int,
    tap: int | None,
) -> list[dict[str, Any]]:
    """Trả về `lessons[]` của đúng cuốn (subject + grade + tap), để
    `SGKProgress.next_lesson()` tính được bài kế tiếp. Trả [] nếu không tìm thấy."""
    sgk_dir = Path(sgk_dir)
    for info in master_index.get("subjects", {}).values():
        if (
            info.get("subject") == subject
            and info.get("grade") == grade
            and (tap is None or info.get("tap") == tap)
        ):
            try:
                return _load_subject_index(sgk_dir, info).get("lessons", [])
            except Exception:
                return []
    return []


# =============================================================
#  Detect môn + lớp + trang
# =============================================================
_TOAN_KEYWORDS = (
    "tính",
    "cộng",
    "trừ",
    "nhân",
    "chia",
    "số",
    "bao nhiêu",
    "phép",
    "đếm",
    "bằng mấy",
    "kết quả",
    "tổng",
    "hiệu",
    "tích",
    "thương",
    "lớn hơn",
    "bé hơn",
    "nhỏ hơn",
    "hình",
    "toán",
)
_TV_KEYWORDS = (
    "đọc",
    "viết",
    "chữ",
    "từ",
    "câu",
    "bài đọc",
    "nghĩa",
    "vần",
    "đánh vần",
    "chính tả",
    "tập làm văn",
    "kể chuyện",
    "thơ",
    "tiếng việt",
)


def _detect_subject(question: str) -> str | None:
    """Trả 'toan' | 'tiengviet' | None.

    Ưu tiên 1 (explicit): bạn ấy nói tên môn trực tiếp ("đọc bài TOÁN", "học
        TIẾNG VIỆT") → dùng đúng môn đó. Đây là tín hiệu mạnh nhất.
    Ưu tiên 2 (keyword): không có tên môn → đoán theo action keyword
        (đọc/viết → TV, cộng/trừ → toán).
    """
    q = question.lower()

    # Explicit mention — câu "đọc bài toán" phải ra toán, không phải TV vì "đọc"
    has_toan = "toán" in q or " toan " in f" {q} " or q.startswith("toan ")
    has_tv = "tiếng việt" in q or "tieng viet" in q
    if has_toan and not has_tv:
        return "toan"
    if has_tv and not has_toan:
        return "tiengviet"

    # Fallback theo action keyword. TV trước để "đọc chữ số 5" không thành toán.
    if any(kw in q for kw in _TV_KEYWORDS):
        return "tiengviet"
    if any(kw in q for kw in _TOAN_KEYWORDS):
        return "toan"
    return None


def _detect_grade(question: str, default: int = 2) -> int:
    """Bắt 'lớp X' trong câu. Mặc định lớp 2 (theo CHILD_PROFILE)."""
    m = re.search(r"l[ớo]p\s*(\d)", question.lower())
    if m:
        return int(m.group(1))
    return default


_PAGE_KW_RE = re.compile(
    # Whisper hay nhầm âm tiếng Việt: "trang" → "trên/tràng/tráng/trảng/chăng",
    # "bài" → "bay/bày/bại". Mở rộng keyword để cover lỗi STT thường gặp.
    r"(?:trang|tràng|tráng|trảng|trên|chăng|bài|bay|bày|bại|page)"
    r"\s*(?:số\s+)?(\d{1,3})",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b(\d{1,3})\b")
# Số đi ngay sau các từ này KHÔNG phải số trang.
_NOT_PAGE_PREFIXES = (
    "lớp",
    "lop",
    "tập",
    "tap",
    "tuổi",
    "giờ",
    "phút",
    "giây",
    "cộng",
    "trừ",
    "nhân",
    "chia",
    "với",
    "và",
    "cho",
    "bằng",
    "+",
    "-",
    "×",
    "x",
    "=",
)
# Fallback bắt số đơn lẻ chỉ chạy khi có hint "liên quan đến đọc sách" —
# nếu không sẽ nhầm phép toán "cộng 5 với 3" thành trang 3.
_PAGE_INTENT_HINTS = (
    "đọc",
    "mở",
    "lật",
    "xem",
    "kể",
    "trang",
    "bài",
    "page",
    "trên",
    "tráng",
    "tràng",
    "trảng",
    "chăng",
    "bay",
    "bày",
    "bại",
)


def _detect_page_number(question: str) -> int | None:
    """Bắt số trang trong câu hỏi của bé.

    Ưu tiên 1: bắt sau keyword 'trang/bài/page' (bao gồm STT mishears như
               Whisper hay nghe nhầm "trang" → "trên" / "tráng").
    Fallback : nếu câu có hint kiểu 'đọc/mở/lật/xem' → bắt số đầu tiên
               KHÔNG đi sau 'lớp/tập/tuổi/cộng/...'. Bỏ qua hoàn toàn fallback
               cho câu hỏi toán thuần như "cộng 5 với 3 bằng mấy".
    """
    q = question.lower()

    m = _PAGE_KW_RE.search(q)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 300:
            return n

    if not any(h in q for h in _PAGE_INTENT_HINTS):
        return None

    for m in _NUMBER_RE.finditer(q):
        n = int(m.group(1))
        if not (1 <= n <= 300):
            continue
        before = q[max(0, m.start() - 10) : m.start()].strip()
        if any(before.endswith(kw) for kw in _NOT_PAGE_PREFIXES):
            continue
        return n
    return None


# =============================================================
#  Tìm trang JPG liên quan
# =============================================================
def find_relevant_pages(
    question: str,
    master_index: dict[str, Any],
    sgk_dir: Path | str = SGK_DIR,
    max: int = 2,
    default_grade: int = 2,
) -> list[dict[str, Any]]:
    """Phân tích câu hỏi → trả list trang JPG liên quan.

    Mỗi item: {path, file, subject, subject_name, grade, page, source}.
    `path` là Path tuyệt đối tới file JPG sẵn sàng để encode base64.

    Logic:
      1. Detect môn (toán/tiếng việt) từ keyword. Không detect được → [].
      2. Detect lớp ('lớp 2' trong câu) — mặc định 2.
      3. Detect số trang ('trang 15', 'bài 10') — nếu có thì load đúng trang đó.
      4. Không có trang cụ thể → lấy `max` bài đầu tiên trong lessons[] (đã skip
         bìa/mục lục) làm reference cho AI nhìn.
    """
    sgk_dir = Path(sgk_dir)
    subject = _detect_subject(question)
    if subject is None:
        return []

    grade = _detect_grade(question, default=default_grade)
    page_num = _detect_page_number(question)

    # Tìm book khớp (subject + grade). Nếu không có grade khớp → fallback bất kỳ
    # tập nào cùng môn (vd hỏi lớp 3 toán nhưng chỉ có lớp 1, 2).
    subjects = master_index.get("subjects", {})
    candidate = None
    for key, info in subjects.items():
        if info.get("subject") == subject and info.get("grade") == grade:
            candidate = info
            break
    if candidate is None:
        for key, info in subjects.items():
            if info.get("subject") == subject:
                candidate = info
                break
    if candidate is None:
        return []

    try:
        book_index = _load_subject_index(sgk_dir, candidate)
    except Exception:
        return []

    book_dir = sgk_dir / candidate["path"]
    pages_dict = book_index.get("pages", {})
    lessons = book_index.get("lessons", [])
    book_pages_map = book_index.get("book_pages", {})

    target_pdf_pages: list[str] = []
    source = ""

    if page_num is not None:
        # Bé nói trang nào thì load đúng trang đó. Ưu tiên book_page → pdf_page
        # (vì bé đọc số trang in trên sách, không phải pdf page).
        pdf_p = book_pages_map.get(str(page_num)) or page_num
        if str(pdf_p) in pages_dict:
            target_pdf_pages.append(str(pdf_p))
            source = f"trang {page_num}"

    if not target_pdf_pages and lessons:
        # Không có trang cụ thể → lấy max bài đầu tiên trong danh sách lesson
        # (đã skip bìa, mục lục — chứa nội dung học thật).
        for lesson in lessons[:max]:
            target_pdf_pages.append(str(lesson["pdf_page"]))
        source = "bài đầu"

    if not target_pdf_pages:
        return []

    out: list[dict[str, Any]] = []
    for p in target_pdf_pages[:max]:
        meta = pages_dict.get(p, {})
        file_name = meta.get("file")
        if not file_name:
            continue
        full_path = book_dir / file_name
        if not full_path.exists():
            continue
        out.append(
            {
                "path": full_path,
                "file": file_name,
                "subject": subject,
                "subject_name": candidate.get("subject_name", subject),
                "grade": candidate.get("grade", grade),
                "tap": candidate.get("tap"),
                "page": meta.get("book_page") or int(p),
                "pdf_page": int(p),
                "source": source,
            }
        )
    return out


# =============================================================
#  Image → base64
# =============================================================
def image_to_base64(path: Path | str) -> str:
    """Encode JPG → base64 ASCII. Dùng cho DeepSeek vision (data URL).

    Trên ESP32 sau này sẽ thay bằng đọc trực tiếp từ SD card → mbedtls base64.
    """
    path = Path(path)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_vision_user_content(
    question: str, pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Dựng user message content kiểu vision: text + list image_url base64.

    Format theo OpenAI-compatible API (DeepSeek tương thích):
        [
          {"type": "text",      "text": "..."},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...",
                                               "detail": "high"}}
        ]
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": question}]
    for p in pages:
        try:
            b64 = image_to_base64(p["path"])
        except Exception as e:
            print(f"  ⚠ Encode SGK image fail ({p.get('file')}): {e}")
            continue
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            }
        )
    return content


def render_sgk_system_prompt(
    ten: str = "bé",
    lop: int | str = 2,
    progress_note: str = "",
) -> str:
    """Inject profile + (optional) tiến độ học vào template.

    `progress_note` là chuỗi do `py.sgk_progress.format_progress_note()` dựng,
    mô tả buổi học gần nhất, bài hôm nay, bài tiếp theo. Ghép vào CUỐI system
    prompt để AI biết chào hỏi đầu phiên và gợi ý bài kế tiếp.
    """
    prompt = SGK_TEACHER_SYSTEM_PROMPT.replace("[TEN]", str(ten or "bé")).replace(
        "[LOP]", str(lop if lop else 2)
    )
    if progress_note:
        prompt = prompt.rstrip() + "\n\n" + progress_note.strip() + "\n"
    return prompt
