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
from datetime import datetime
from pathlib import Path
from typing import Any


class SGKProgress:
    def __init__(self, path: Path, max_entries: int = 200) -> None:
        self.path = path
        self.max_entries = max_entries
        self.entries: list[dict] = []
        self._load()

    # ---- persistence ----
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("learned"), list):
                self.entries = [e for e in data["learned"] if isinstance(e, dict)]
        except Exception as e:
            print(f"[SGKProgress] Load fail: {e} — bắt đầu rỗng")
            self.entries = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"learned": self.entries}
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
        self._save()

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
