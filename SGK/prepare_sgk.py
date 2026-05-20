"""
prepare_sgk.py — Chuyển PDF SGK sang JPG tối ưu cho AI Vision
Chạy 1 lần trên PC, output copy vào SD card

Cách dùng:
  python prepare_sgk.py --pdf "Lớp_2_Toán_tập_1.pdf" --subject toan --grade 2 --tap 1
  python prepare_sgk.py --pdf "Lớp_2_Tiếng_Việt_tập_1.pdf" --subject tiengviet --grade 2 --tap 1
  python prepare_sgk.py --all  (convert tất cả PDF trong folder hiện tại)
"""

import fitz  # PyMuPDF
import json
import os
import re
import argparse
from pathlib import Path
from PIL import Image
import io

# ── Cấu hình tối ưu cho ESP32 Vision ──────────────────────────────────────────
DPI         = 120          # Đủ rõ cho Vision AI, không quá nặng
JPEG_QUALITY= 82           # 80-85 là sweet spot: chất lượng tốt, file nhỏ
MAX_WIDTH   = 900          # px — resize nếu quá rộng
MAX_HEIGHT  = 1260         # px — resize nếu quá cao
TARGET_KB   = 150          # KB target mỗi trang (thông tin, không hard limit)

# ── Từ khóa nhận diện loại trang ──────────────────────────────────────────────
PAGE_TYPE_KEYWORDS = {
    "muc_luc":    ["mục lục", "table of contents", "nội dung"],
    "loi_noi_dau":["lời nói đầu", "lời giới thiệu", "hướng dẫn sử dụng"],
    "on_tap":     ["ôn tập", "ôn tập học kì", "tổng kết"],
    "thuc_hanh":  ["thực hành", "trải nghiệm"],
    "bai_hoc":    [],  # default
}

# Từ điển nhận diện chủ đề bài học từ tên file / subject
SUBJECT_META = {
    "toan": {
        "ten_day_du": "Toán",
        "keywords": ["số", "cộng", "trừ", "nhân", "chia", "hình", "đo", "bảng",
                     "phép", "tính", "lít", "cm", "kg", "giờ", "ngày", "tháng",
                     "ôn tập", "luyện tập", "thực hành"],
    },
    "tiengviet": {
        "ten_day_du": "Tiếng Việt",
        "keywords": ["đọc", "viết", "chính tả", "tập đọc", "kể chuyện", "luyện từ",
                     "câu", "nghe", "nói", "chữ hoa", "bài thơ", "văn bản"],
    },
    "tiengviet_tap1": {"ten_day_du": "Tiếng Việt tập 1"},
    "tiengviet_tap2": {"ten_day_du": "Tiếng Việt tập 2"},
    "toan_tap1":      {"ten_day_du": "Toán tập 1"},
    "toan_tap2":      {"ten_day_du": "Toán tập 2"},
}


def detect_subject_from_filename(filename: str) -> tuple[str, int, int]:
    """
    Trả về (subject_key, grade, tap) từ tên file.
    Ví dụ: 'Lớp_2_Toán_tập_1.pdf' → ('toan', 2, 1)
    """
    name = filename.lower()
    # Grade
    grade_match = re.search(r'lớp[_\s]*(\d)', name) or re.search(r'l[oô]p[_\s]*(\d)', name)
    grade = int(grade_match.group(1)) if grade_match else 0
    # Tập
    tap_match = re.search(r'tập[_\s]*(\d)', name) or re.search(r'tap[_\s]*(\d)', name)
    tap = int(tap_match.group(1)) if tap_match else 1
    # Subject
    if 'toán' in name or 'toan' in name:
        subject = 'toan'
    elif 'tiếng việt' in name or 'tieng viet' in name or 'tiengviet' in name:
        subject = 'tiengviet'
    elif 'tiếng anh' in name or 'english' in name:
        subject = 'english'
    elif 'khoa học' in name or 'khoa hoc' in name:
        subject = 'khoahoc'
    else:
        subject = 'khac'
    return subject, grade, tap


def extract_text_from_page(page) -> str:
    """Extract text từ page PyMuPDF, fallback empty string nếu scan."""
    try:
        text = page.get_text("text").strip()
        return text if len(text) > 5 else ""
    except:
        return ""


def detect_page_type(text: str, page_num: int) -> str:
    """Nhận diện loại trang dựa trên text extract được."""
    text_lower = text.lower()
    for page_type, keywords in PAGE_TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return page_type
    return "bai_hoc"


def get_page_number_from_text(text: str) -> int | None:
    """
    Cố gắng lấy số trang in trong sách (khác với index PDF).
    SGK thường có số trang ở cuối trang.
    """
    # Tìm số đơn lẻ ở cuối text (thường là số trang)
    matches = re.findall(r'\b(\d{1,3})\b', text[-50:] if len(text) > 50 else text)
    if matches:
        try:
            num = int(matches[-1])
            if 1 <= num <= 200:  # range hợp lý cho SGK
                return num
        except:
            pass
    return None


def convert_pdf_to_jpg(pdf_path: str, output_dir: str, subject: str, grade: int, tap: int):
    """
    Convert PDF → JPG tối ưu + tạo index.json thông minh.
    """
    pdf_path = Path(pdf_path)
    subject_key = f"{subject}_l{grade}_t{tap}" if tap > 0 else f"{subject}_l{grade}"
    out_dir = Path(output_dir) / "sgk" / subject_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"📚 Convert: {pdf_path.name}")
    print(f"   Subject: {subject} | Lớp: {grade} | Tập: {tap}")
    print(f"   Output:  {out_dir}")
    print(f"{'='*60}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    print(f"   Tổng số trang PDF: {total_pages}")

    # ── Index structure ────────────────────────────────────────────────────────
    index = {
        "subject":      subject,
        "subject_name": SUBJECT_META.get(subject, {}).get("ten_day_du", subject),
        "grade":        grade,
        "tap":          tap,
        "total_pages":  total_pages,
        "pdf_source":   pdf_path.name,
        "pages":        {},        # pdf_page_num → metadata
        "book_pages":   {},        # book_page_num → pdf_page_num (để lookup nhanh)
        "lessons":      [],        # list các bài học với trang bắt đầu
    }

    total_size = 0
    sizes = []

    for page_num in range(total_pages):
        page = doc[page_num]
        pdf_page = page_num + 1  # 1-based

        # ── Render page ────────────────────────────────────────────────────────
        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to PIL for resize + quality control
        img_data = pix.tobytes("jpeg")
        img = Image.open(io.BytesIO(img_data))

        # Resize nếu quá lớn
        w, h = img.size
        if w > MAX_WIDTH or h > MAX_HEIGHT:
            ratio = min(MAX_WIDTH / w, MAX_HEIGHT / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Save JPEG
        filename = f"page_{pdf_page:03d}.jpg"
        filepath = out_dir / filename
        img.save(str(filepath), "JPEG", quality=JPEG_QUALITY, optimize=True)

        file_size = filepath.stat().st_size
        total_size += file_size
        sizes.append(file_size)

        # ── Extract metadata ───────────────────────────────────────────────────
        text = extract_text_from_page(page)
        page_type = detect_page_type(text, pdf_page)
        book_page = get_page_number_from_text(text)

        page_meta = {
            "file":       filename,
            "pdf_page":   pdf_page,
            "book_page":  book_page,
            "type":       page_type,
            "size_kb":    round(file_size / 1024, 1),
            "text_preview": text[:120].replace('\n', ' ') if text else "",
        }
        index["pages"][str(pdf_page)] = page_meta

        # Map book page → pdf page để tìm nhanh
        if book_page:
            index["book_pages"][str(book_page)] = pdf_page

        # Progress
        if pdf_page % 20 == 0 or pdf_page == total_pages:
            avg_kb = total_size / pdf_page / 1024
            print(f"   [{pdf_page:3d}/{total_pages}] avg {avg_kb:.0f}KB/trang, "
                  f"total {total_size/1024/1024:.1f}MB")

    doc.close()

    # ── Tạo lesson index từ mục lục (trang 4-7 thường là mục lục) ─────────────
    print(f"\n   📋 Đang tạo lesson index...")
    index["lessons"] = build_lesson_index(out_dir, index, subject, grade)

    # ── Save index.json ────────────────────────────────────────────────────────
    index_path = out_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    avg_kb = total_size / total_pages / 1024
    min_kb = min(sizes) / 1024
    max_kb = max(sizes) / 1024
    print(f"\n   ✅ Hoàn thành!")
    print(f"   📁 {total_pages} files JPG")
    print(f"   📊 Kích thước: avg={avg_kb:.0f}KB, min={min_kb:.0f}KB, max={max_kb:.0f}KB")
    print(f"   💾 Tổng: {total_size/1024/1024:.1f}MB")
    print(f"   📄 Index: {index_path}")
    print(f"   🎯 Bài học detected: {len(index['lessons'])}")

    return index


def build_lesson_index(out_dir: Path, index: dict, subject: str, grade: int) -> list:
    """
    Dùng DeepSeek Vision để đọc trang mục lục và tạo lesson mapping.
    Fallback: tạo index đơn giản theo trang.
    """
    lessons = []

    # Cố gắng đọc text từ các trang đầu (mục lục)
    # Với SGK scan, text rỗng → tạo index cơ bản
    muc_luc_pages = []
    for pdf_page_str, meta in index["pages"].items():
        if meta["type"] == "muc_luc" or int(pdf_page_str) <= 8:
            muc_luc_pages.append(int(pdf_page_str))

    # Parse book_pages để tạo lesson list
    book_pages = index.get("book_pages", {})

    if len(book_pages) > 10:
        # Có đủ book page mapping → tạo lesson index từ đó
        sorted_book_pages = sorted([(int(k), v) for k, v in book_pages.items()])
        for book_p, pdf_p in sorted_book_pages:
            # Chỉ lấy các trang nội dung (bỏ qua trang đầu sách)
            if book_p >= 6:
                lessons.append({
                    "book_page":  book_p,
                    "pdf_page":   pdf_p,
                    "file":       f"page_{pdf_p:03d}.jpg",
                    "title":      f"Trang {book_p}",
                    "subject":    subject,
                    "grade":      grade,
                })
    else:
        # Fallback: index theo pdf page (bỏ ~5 trang đầu là bìa/mục lục)
        start_content = 7  # Thường trang 7-8 trở đi là nội dung
        for pdf_page_str, meta in index["pages"].items():
            pdf_p = int(pdf_page_str)
            if pdf_p >= start_content and meta["type"] == "bai_hoc":
                lessons.append({
                    "book_page":  meta.get("book_page") or pdf_p,
                    "pdf_page":   pdf_p,
                    "file":       meta["file"],
                    "title":      f"Trang {meta.get('book_page') or pdf_p}",
                    "subject":    subject,
                    "grade":      grade,
                    "preview":    meta.get("text_preview", ""),
                })

    return lessons


def create_master_index(output_dir: str):
    """Tạo master index.json tổng hợp tất cả SGK trong SD card."""
    out_dir = Path(output_dir)
    sgk_dir = out_dir / "sgk"

    master = {
        "version": "1.0",
        "description": "SGK Index cho Robot AI",
        "subjects": {}
    }

    if not sgk_dir.exists():
        return

    for subject_dir in sorted(sgk_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        index_file = subject_dir / "index.json"
        if not index_file.exists():
            continue
        with open(index_file, encoding="utf-8") as f:
            idx = json.load(f)

        key = subject_dir.name
        master["subjects"][key] = {
            "subject":      idx["subject"],
            "subject_name": idx.get("subject_name", idx["subject"]),
            "grade":        idx["grade"],
            "tap":          idx["tap"],
            "total_pages":  idx["total_pages"],
            "lesson_count": len(idx.get("lessons", [])),
            "path":         f"sgk/{key}",
            "index_file":   f"sgk/{key}/index.json",
        }

    master_path = out_dir / "master_index.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, ensure_ascii=False, indent=2)

    print(f"\n📚 Master index: {master_path}")
    print(f"   Tổng SGK: {len(master['subjects'])} cuốn")
    for key, info in master["subjects"].items():
        print(f"   • {info['subject_name']} lớp {info['grade']} "
              f"tập {info['tap']}: {info['total_pages']} trang")


def main():
    parser = argparse.ArgumentParser(description="Convert SGK PDF → JPG cho AI Vision")
    parser.add_argument("--pdf",     help="Path đến file PDF")
    parser.add_argument("--subject", help="Môn học: toan/tiengviet/english/khoahoc")
    parser.add_argument("--grade",   type=int, help="Lớp: 1-5")
    parser.add_argument("--tap",     type=int, default=1, help="Tập: 1 hoặc 2")
    parser.add_argument("--output",  default="./SD_CARD", help="Thư mục output")
    parser.add_argument("--all",     action="store_true",
                        help="Convert tất cả PDF trong folder hiện tại")
    args = parser.parse_args()

    output_dir = args.output

    if args.all:
        # Auto-detect và convert tất cả PDF
        pdf_files = list(Path(".").glob("*.pdf")) + list(Path(".").glob("**/*.pdf"))
        if not pdf_files:
            print("❌ Không tìm thấy file PDF nào!")
            return
        print(f"🔍 Tìm thấy {len(pdf_files)} file PDF:")
        for f in pdf_files:
            print(f"   • {f.name}")
        print()
        for pdf_file in pdf_files:
            subject, grade, tap = detect_subject_from_filename(pdf_file.name)
            if grade == 0:
                print(f"⚠️  Bỏ qua {pdf_file.name} — không nhận diện được lớp")
                continue
            convert_pdf_to_jpg(str(pdf_file), output_dir, subject, grade, tap)

    elif args.pdf:
        subject = args.subject
        grade   = args.grade
        tap     = args.tap or 1

        # Auto-detect nếu không truyền vào
        if not subject or not grade:
            auto_subject, auto_grade, auto_tap = detect_subject_from_filename(args.pdf)
            subject = subject or auto_subject
            grade   = grade   or auto_grade
            tap     = tap     or auto_tap
            print(f"🔍 Auto-detect: subject={subject}, grade={grade}, tap={tap}")

        convert_pdf_to_jpg(args.pdf, output_dir, subject, grade, tap)

    else:
        parser.print_help()
        return

    # Tạo master index sau khi convert xong
    create_master_index(output_dir)
    print(f"\n✅ Xong! Copy thư mục '{output_dir}' vào SD card là dùng được.")


if __name__ == "__main__":
    main()
