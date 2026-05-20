"""
ConversationMemory — nhớ lịch sử hội thoại qua các turn.

Lưu danh sách messages dạng [{"role": "user"|"assistant", "content": "..."}, ...]
ra JSON. Tự load lại khi khởi tạo → bot nhớ cả qua các lần restart Python.

Chính sách:
  - Đĩa: lưu TOÀN BỘ hội thoại, KHÔNG bao giờ xóa cũ (trừ khi gọi clear()).
  - LLM: chỉ gửi `max_turns` cặp gần nhất (mặc định 30) để khớp context window.
Nếu hội thoại dài hàng trăm turn — file vẫn có đủ, có thể đọc lại sau.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class ConversationMemory:
    def __init__(self, path: Path, max_turns: int = 30) -> None:
        """
        path      : nơi lưu JSON (tests/_tmp/chat_history.json hoặc /history.json trên SD)
        max_turns : số CẶP user+assistant gửi cho LLM mỗi lần. Đĩa lưu hết, không trim.
        """
        self.path = path
        self.max_turns = max_turns
        self.messages: list[dict] = []
        self._load()

    # ---- persistence ----
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.messages = [
                    m
                    for m in data
                    if isinstance(m, dict)
                    and m.get("role") in ("user", "assistant")
                    and isinstance(m.get("content"), str)
                ]
            elif isinstance(data, dict) and "messages" in data:
                self.messages = data["messages"]
        except Exception as e:
            print(f"[Memory] Load fail: {e} — bắt đầu rỗng")
            self.messages = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- mutation ----
    def add_user(self, content: str) -> None:
        if not content:
            return
        self.messages.append(
            {
                "role": "user",
                "content": content,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self._save()

    def add_assistant(self, content: str) -> None:
        if not content:
            return
        self.messages.append(
            {
                "role": "assistant",
                "content": content,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self._save()

    def clear(self) -> None:
        self.messages = []
        self._save()

    # ---- accessors ----
    def for_llm(self, max_turns: int | None = None) -> list[dict]:
        """Trả về cleaning list messages cho API: chỉ role+content, cắt về `max_turns`
        cặp gần nhất. Đĩa không bị ảnh hưởng."""
        limit = (max_turns or self.max_turns) * 2
        tail = self.messages[-limit:] if limit > 0 else self.messages
        return [{"role": m["role"], "content": m["content"]} for m in tail]

    def __len__(self) -> int:
        return len(self.messages)

    def summary(self) -> str:
        """Tóm tắt ngắn để in lên console."""
        n_user = sum(1 for m in self.messages if m["role"] == "user")
        n_ai = sum(1 for m in self.messages if m["role"] == "assistant")
        n_to_llm = min(len(self.messages), self.max_turns * 2)
        return (
            f"{n_user} câu user + {n_ai} câu AI trên đĩa "
            f"(gửi LLM {n_to_llm} messages = {self.max_turns} cặp gần nhất)"
        )
