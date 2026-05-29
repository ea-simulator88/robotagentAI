"""
SGKProgress — nhớ trang nào bạn ấy đã học, ngày nào, để AI Teacher chào hỏi và
gợi ý bài kế tiếp.

Persist JSON. Mỗi lượt SGK pipeline trả lời thành công → ghi 1 entry:
    {"date": "2026-05-21", "subject": "tiengviet", "grade": 2, "tap": 1, "page": 30}

Context inject vào system prompt mỗi lượt (qua `format_progress_note`):
- Bài học gần nhất ngày HÔM QUA (hoặc trước đó) → để AI chào "hôm qua bạn học bài X"
- Bài hôm nay đang học → để AI biết tiếp tục
- Bài kế tiếp trong sách → để AI gợi ý "bài sau là trang Y"
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# Pending switch tự reset sau khoảng này nếu bé không xác nhận lại.
# 10 phút đủ để bé cân nhắc nhưng không kẹt mãi nếu bé bỏ giữa chừng.
PENDING_SWITCH_TTL_SECONDS = 10 * 60


class SGKProgress:
    def __init__(self, path: Path, max_entries: int = 200) -> None:
        self.path = path
        self.max_entries = max_entries
        self.entries: list[dict] = []
        # State "đang đợi bé xác nhận chuyển sách". Lưu cả original transcript
        # để replay request gốc sau khi bé OK.
        self.pending_switch: dict[str, Any] | None = None
        self._load()

    # ---- persistence ----
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if isinstance(data.get("learned"), list):
                    self.entries = [
                        e for e in data["learned"] if isinstance(e, dict)
                    ]
                ps = data.get("pending_switch")
                if isinstance(ps, dict):
                    self.pending_switch = ps
        except Exception as e:
            print(f"[SGKProgress] Load fail: {e} — bắt đầu rỗng")
            self.entries = []
            self.pending_switch = None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"learned": self.entries}
        if self.pending_switch:
            payload["pending_switch"] = self.pending_switch
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- mutation ----
    def record(
        self, subject: str, grade: int, tap: int | None, page: int,
        page_label: str | None = None,
    ) -> None:
        """Ghi nhận bạn ấy vừa học trang `page` của môn này. Dedupe theo (date,
        subject, grade, tap, page) → 1 ngày học cùng trang nhiều lần chỉ tính 1."""
        now = datetime.now()
        today = now.date().isoformat()
        for e in self.entries:
            if (
                e.get("date") == today
                and e.get("subject") == subject
                and e.get("grade") == grade
                and e.get("tap") == tap
                and e.get("page") == page
            ):
                return
        self.entries.append({
            "date": today,
            "ts": now.isoformat(timespec="seconds"),
            "subject": subject,
            "grade": grade,
            "tap": tap,
            "page": page,
            "page_label": page_label or f"trang {page}",
        })
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
        self._save()

    def clear(self) -> None:
        self.entries = []
        self.pending_switch = None
        self._save()

    # ---- switch confirmation state machine ----
    def most_recent_book(self) -> dict[str, Any] | None:
        """Cuốn đang học gần nhất — (subject, grade, tap) từ entry mới nhất."""
        if not self.entries:
            return None
        e = self.entries[-1]
        return {
            "subject": e.get("subject"),
            "grade": e.get("grade"),
            "tap": e.get("tap"),
        }

    def most_recent_entry(self) -> dict[str, Any] | None:
        return self.entries[-1] if self.entries else None

    def set_pending_switch(
        self,
        from_book: dict[str, Any],
        to_book: dict[str, Any],
        original_transcript: str = "",
        book_key: str | None = None,
        lesson_number: int | None = None,
        page_number: int | None = None,
    ) -> None:
        self.pending_switch = {
            "from": from_book,
            "to": to_book,
            "original_transcript": original_transcript,
            "book_key": book_key,
            "lesson_number": lesson_number,
            "page_number": page_number,
            "asked_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save()

    def clear_pending_switch(self) -> None:
        if self.pending_switch is not None:
            self.pending_switch = None
            self._save()

    def pending_expired(self, ttl_seconds: int = PENDING_SWITCH_TTL_SECONDS) -> bool:
        """True nếu pending đã quá hạn (bé bỏ ngang, không xác nhận)."""
        if not self.pending_switch:
            return True
        asked = self.pending_switch.get("asked_at")
        if not asked:
            return True
        try:
            asked_dt = datetime.fromisoformat(asked)
        except ValueError:
            return True
        return (datetime.now() - asked_dt).total_seconds() > ttl_seconds

    def active_pending(self) -> dict[str, Any] | None:
        """Trả pending nếu còn hiệu lực; tự xoá nếu hết hạn."""
        if not self.pending_switch:
            return None
        if self.pending_expired():
            self.clear_pending_switch()
            return None
        return self.pending_switch

    def is_confirming_pending(self, transcript: str) -> bool:
        """Bé có đang xác nhận pending switch không?

        Match nếu transcript chứa 'lớp X' đúng với target grade của pending —
        đủ explicit, không nhầm với câu chuyện khác. KHÔNG match riêng chữ
        'ừ' không kèm số lớp vì dễ false-positive (bé 'ừ' khi đáp câu khác).
        """
        pending = self.active_pending()
        if pending is None:
            return False
        to_grade = (pending.get("to") or {}).get("grade")
        if to_grade is None:
            return False
        m = re.search(r"l[ớo]p\s*(\d)", transcript.lower())
        return bool(m and int(m.group(1)) == to_grade)

    # ---- accessors ----
    def __len__(self) -> int:
        return len(self.entries)

    def today_entries(self, subject: str | None = None) -> list[dict]:
        today = datetime.now().date().isoformat()
        return [
            e for e in self.entries
            if e.get("date") == today
            and (subject is None or e.get("subject") == subject)
        ]

    def latest_before_today(self, subject: str | None = None) -> dict | None:
        """Bài học gần nhất NGÀY TRƯỚC hôm nay. Để AI chào 'hôm qua bạn học...'.
        Trả về None nếu chưa có session nào ngày trước."""
        today = datetime.now().date().isoformat()
        for e in reversed(self.entries):
            if e.get("date") == today:
                continue
            if subject is not None and e.get("subject") != subject:
                continue
            return e
        return None

    def highest_page_in_book(
        self, subject: str, grade: int, tap: int | None,
    ) -> int:
        """Số trang LỚN NHẤT đã học trong đúng cuốn này. Dùng để tính bài kế tiếp."""
        last = 0
        for e in self.entries:
            if (
                e.get("subject") == subject
                and e.get("grade") == grade
                and e.get("tap") == tap
            ):
                p = e.get("page") or 0
                if p > last:
                    last = p
        return last

    def next_lesson(
        self,
        subject: str,
        grade: int,
        tap: int | None,
        lessons: list[dict],
    ) -> dict | None:
        """Bài kế tiếp trong `lessons` (book_index['lessons']) — trang đầu tiên
        có `book_page` lớn hơn trang cao nhất đã học. Nếu chưa học gì → bài đầu.
        Trả None nếu đã học hết sách."""
        if not lessons:
            return None
        highest = self.highest_page_in_book(subject, grade, tap)
        if highest == 0:
            return lessons[0]
        for lesson in lessons:
            book_p = lesson.get("book_page") or lesson.get("pdf_page", 0)
            if book_p > highest:
                return lesson
        return None


# =============================================================
#  Helpers cho switch flow (dùng từ test.py)
# =============================================================
def books_equal(a: dict | None, b: dict | None) -> bool:
    """So sánh (subject, grade, tap). None ≠ None — caller phải tự check None."""
    if a is None or b is None:
        return False
    return (
        a.get("subject") == b.get("subject")
        and a.get("grade") == b.get("grade")
        and a.get("tap") == b.get("tap")
    )


def book_label(book: dict | None) -> str:
    """Hiển thị tên cuốn cho bé: 'Tiếng Việt lớp 2 tập 1'."""
    if not book:
        return "?"
    subj = book.get("subject")
    subj_name = "Tiếng Việt" if subj == "tiengviet" else "Toán" if subj == "toan" else (subj or "?")
    return f"{subj_name} lớp {book.get('grade', '?')} tập {book.get('tap', '?')}"


def build_switch_confirmation_text(
    current_book: dict,
    requested_book: dict,
    recent_entry: dict | None,
    next_lesson: dict | None,
) -> str:
    """Câu nhắc bé đang học cuốn cũ + hỏi xác nhận chuyển. Dùng giọng bạn bè
    "mình/tớ - bạn" giống SGK Teacher persona."""
    cur_label = book_label(current_book)
    req_label = book_label(requested_book)

    parts: list[str] = []
    if recent_entry:
        page_label = recent_entry.get("page_label") or f"trang {recent_entry.get('page')}"
        parts.append(
            f"Khoan nhé bạn! Hôm trước ngày {recent_entry.get('date')} "
            f"mình với bạn đang học {cur_label} {page_label} mà."
        )
    else:
        parts.append(f"Khoan nhé bạn! Bạn đang học {cur_label} đó.")

    if next_lesson:
        nxt_p = next_lesson.get("book_page") or next_lesson.get("pdf_page")
        title = next_lesson.get("title") or f"trang {nxt_p}"
        parts.append(
            f"Mình đề xuất học tiếp {title} của {cur_label} cho liền mạch nha."
        )
    else:
        parts.append(f"Mình nên học tiếp {cur_label} cho liền mạch nha.")

    parts.append(
        f"Nếu bạn thật sự muốn chuyển sang {req_label} thì nói lại 1 lần nữa "
        f"'mình muốn học lớp {requested_book.get('grade')}' nha. "
        f"Bạn muốn học tiếp {cur_label} hay chuyển hẳn sang {req_label}?"
    )
    return " ".join(parts)


def build_no_lesson_text(toc_info: dict[str, Any]) -> str:
    """Khi vision tra mục lục mà không thấy bài N → nói thật + hỏi lại có phải
    bé muốn nói TRANG N không. Tránh AI bịa nội dung bài không có thật."""
    n = toc_info.get("lesson_number")
    subj_name = toc_info.get("subject_name", "?")
    grade = toc_info.get("grade", "?")
    tap = toc_info.get("tap", "?")
    return (
        f"Mình tra mục lục {subj_name} lớp {grade} tập {tap} mà không thấy bài "
        f"{n} bạn ơi. Có phải bạn muốn học TRANG {n} không? "
        f"Hoặc bạn nói lại số bài khác cho mình nha."
    )


def build_ambiguous_subject_text(lesson_or_page_label: str = "") -> str:
    """Khi bé yêu cầu SGK rõ ràng (có 'bài/trang/lớp/tập/giảng') nhưng KHÔNG
    nói môn nào, và bé cũng chưa từng học gì → hỏi lại chọn môn. Tránh đoán bậy."""
    suffix = f" ({lesson_or_page_label})" if lesson_or_page_label else ""
    return (
        f"Bạn muốn học môn nào nhỉ{suffix} — Toán hay Tiếng Việt? "
        f"Bạn nói rõ giúp mình với nha."
    )


# =============================================================
#  Format progress note để inject vào system prompt
# =============================================================
def format_progress_note(
    progress: SGKProgress,
    current_pages: list[dict[str, Any]] | None = None,
    next_lesson: dict[str, Any] | None = None,
) -> str:
    """Dựng đoạn text ngắn về tiến độ học, ghép vào cuối SGK system prompt.

    Bao gồm:
      - Bài gần nhất NGÀY TRƯỚC (nếu có) → để AI chào hỏi đầu phiên
      - Bài đang học HÔM NAY (current_pages) → AI biết context
      - Bài kế tiếp trong sách (next_lesson) → để AI gợi ý sau khi học xong

    Nếu không có dữ liệu gì → trả chuỗi rỗng (không inject section dư thừa).
    """
    lines: list[str] = []

    prev = progress.latest_before_today()
    if prev:
        subject_name = "Tiếng Việt" if prev.get("subject") == "tiengviet" else "Toán"
        prev_label = prev.get("page_label") or f"trang {prev.get('page')}"
        lines.append(
            f"- Buổi trước (ngày {prev.get('date')}): bạn ấy đã học "
            f"{subject_name} lớp {prev.get('grade')} tập {prev.get('tap')} "
            f"{prev_label}."
            " Hãy chào bạn ấy và nhắc lại 1 câu về bài cũ trước khi vào bài mới."
        )

    if current_pages:
        head = current_pages[0]
        pages_str = ", ".join(
            f"trang {p.get('page')}" for p in current_pages if p.get("page")
        )
        lines.append(
            f"- Hôm nay đang học: {head.get('subject_name', '?')} lớp "
            f"{head.get('grade', '?')} tập {head.get('tap', '?')} — {pages_str}."
        )

    if next_lesson:
        next_p = next_lesson.get("book_page") or next_lesson.get("pdf_page")
        title = next_lesson.get("title") or f"trang {next_p}"
        lines.append(
            f"- Bài kế tiếp trong sách: {title}."
            " Sau khi học xong bài hôm nay → gợi ý bạn ấy học bài này tiếp."
        )

    if not lines:
        return ""
    return "TIẾN ĐỘ HỌC (dùng để chào hỏi + gợi ý):\n" + "\n".join(lines)
