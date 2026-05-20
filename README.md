# 🤖 AI Voice Assistant ESP32-S3

## Tổng quan

Bot học tập cho trẻ em, hỗ trợ 4 ngôn ngữ, chạy trên ESP32-S3.
**Con nói → Bot trả lời bằng giọng nói thông minh.**

---

## 🛠 Hardware đã mua

| Linh kiện | Chi tiết |
|---|---|
| Board | Waveshare ESP32-S3-LCD-1.47B-M (LX7 dual core 240 MHz, 8 MB PSRAM, 16 MB Flash) |
| Màn hình | LCD ST7789 172×320 IPS |
| Mic | INMP441 I2S (国产版, đã hàn pin) |
| Audio amp | MAX98357 I2S (BGA封装, đã hàn) |
| Loa | 8 Ω · 1 W · 30 mm |
| Dây | Jumper Đực–Cái 21 cm + Cái–Cái 20 cm |
| Pin | LiPo 3.7 V (mua tại VN) |
| MicroSD | 8 GB Class 10 |

---

## 🔌 Sơ đồ kết nối GPIO

| Module | Chân |
|---|---|
| **INMP441 mic** | SCK → GPIO4 · SD → GPIO6 · WS → GPIO5 · L/R → GND · VDD → 3.3 V |
| **MAX98357 amp** | BCLK → GPIO15 · DIN → GPIO7 · LRC → GPIO16 |
| **MicroSD (SPI)** | CS → GPIO34 · MOSI → GPIO35 · MISO → GPIO37 · SCLK → GPIO36 |
| **RGB LED** | GPIO38 (NeoPixel WS2812 onboard) |
| **Button** | GPIO0 (BOOT, INPUT_PULLUP) |

> Chân LCD (ST7789) định nghĩa trong `platformio.ini` qua TFT_eSPI build flags: MOSI=45, SCLK=40, CS=42, DC=41, RST=39, BL=48.

---

## 🔑 API Keys

Lưu trong [include/config.h](include/config.h) — file này **đã gitignored**.
Mẫu trống ở [config.example.h](config.example.h).

| Key | Trạng thái |
|---|---|
| `GROQ_API_KEY` | ✅ |
| `DEEPSEEK_API_KEY` | ✅ |
| `BRAVE_API_KEY` | ✅ |
| Edge TTS | ✅ (không cần key) |

---

## 🌐 4 ngôn ngữ

| Ngôn ngữ | Whisper code | Edge TTS voice |
|---|---|---|
| 🇻🇳 Tiếng Việt | `vi` | `vi-VN-NamMinhNeural` (nam) |
| 🇬🇧 Tiếng Anh | `en` | `en-US-AriaNeural` (nữ) |
| 🇨🇳 Phổ Thông | `zh` | `zh-CN-XiaoxiaoNeural` (nữ) |
| 🇭🇰 Quảng Đông | `yue` → `zh` | `zh-HK-HiuMaanNeural` (nữ) |

---

## 🧱 Kiến trúc hệ thống

```
Con nói 🎤
   ↓
INMP441 thu âm
   ↓
ESP32 gửi audio → Backend VPS
   ↓
Groq Whisper STT → text
   ↓
Auto-detect ngôn ngữ (vi / en / zh / yue)
   ↓
AlarmManager check lệnh báo thức ─── yes ──→ confirm + thực thi
   ↓ no
DeepSeek V4 xử lý
   (kèm datetime + profile + alarm list)
   ↓
Backend trả về text
   ↓
Edge TTS → audio MP3
   ↓
ESP32 phát loa + hiệu ứng LCD + RGB LED
```

---

## 🐍 Backend (Python FastAPI — VPS ~100 k/tháng)

| File | Vai trò |
|---|---|
| `main.py` | API endpoints |
| [alarm_manager.py](tests/alarm_manager.py) | Báo thức NLP 4 ngôn ngữ ✅ |
| `profile_manager.py` | Quản lý profile người dùng |
| `stt.py` | Groq Whisper |
| `llm.py` | DeepSeek V4 + datetime + web search |
| `tts.py` | Edge TTS |
| `search.py` | Brave Search ($5 credit/tháng free) |

> Hiện code đang ở thư mục [tests/](tests/) như Python prototype. Khi hoàn thiện sẽ tách sang `backend/` để deploy FastAPI.

---

## 📁 Cấu trúc thư mục

```
Robot_ESP32-S3/
├── src/
│   └── main.cpp                # Firmware chính ESP32-S3
├── include/
│   ├── config.h                # Cấu hình/API key thật, file chính
│   └── config.example.h        # Mẫu config
├── lib/                        # Thư viện local PlatformIO nếu có
├── test/                       # Folder test mặc định của PlatformIO
├── py/
│   ├── common.py               # Module Python dùng chung
│   ├── alarm_manager.py        # Logic báo thức
│   ├── alarm_parser.py         # Parse intent báo thức
│   └── memory.py               # Lưu hội thoại
├── tests/
│   ├── test.py                 # File test tổng hợp, không phải firmware chính
│   └── _tmp/                   # File sinh ra khi test: mp3, wav, json
├── platformio.ini              # Cấu hình build/flash ESP32
├── requirements.txt            # Dependency Python test/backend prototype
├── config.example.h            # Mẫu config
└── README.md
```

---

## ✨ Tính năng

- ✅ Voice Q&A 4 ngôn ngữ
- ✅ Auto-detect ngôn ngữ
- ✅ Nhớ lịch sử chat (MicroSD)
- ✅ Profile người dùng (tên, sở thích, lớp học)
- ✅ Báo thức thông minh NLP 4 ngôn ngữ
- ✅ Thời gian thực (datetime inject vào mọi LLM call)
- ✅ Web search (Brave)
- ✅ RGB LED báo trạng thái
- ✅ LCD hiển thị biểu cảm

---

## 🗣 Phong cách giao tiếp

- **Bot xưng:** mình / tớ (không xưng "thầy/cô" hay "chị/em")
- **Gọi người dùng:** bạn / tên riêng
- Giải thích đơn giản, ví dụ gần gũi, ≤ 3 câu mỗi lượt
- Như bạn bè cùng tuổi, không như giáo viên

---

## 📊 Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Hardware đã đặt mua (Taobao + Shopee) | ✅ |
| PlatformIO setup | ✅ |
| Firmware ESP32 C++ ([src/main.cpp](src/main.cpp)) | ✅ |
| Python pipeline test (STT + LLM + TTS) | ✅ |
| Alarm manager 4 ngôn ngữ | ✅ |
| Backend FastAPI hoàn thiện | ⏳ |
| Hardware về (12–15 ngày) | ⏳ |
| Deploy VPS | ⏳ |
| File STL vỏ robot One Piece | ⏳ |

---

## 📋 Việc cần làm tiếp

1. Hoàn thiện backend FastAPI (local test trước)
2. ~~Fix lỗi `pyaudioop`~~ — đã xong (dùng numpy + ffmpeg thay pydub)
3. Fix lỗi TTS phát âm sai ngôn ngữ (xem **🐛 Known Bugs** bên dưới)
4. Test `alarm_manager` đầy đủ với mọi edge case
5. Khi hardware về → flash firmware → test end-to-end
6. Deploy backend lên VPS
7. Design file STL vỏ One Piece (đo kích thước sau khi ráp xong)
8. GitHub commit khi code ổn định

---

## 🐛 Known Bugs

### TTS phát âm sai khi câu trả lời hỗn hợp ngôn ngữ

**Mô tả:** Khi hỏi bằng tiếng Việt nhưng AI trả lời kèm câu ví dụ tiếng Anh (VD: dạy ngoại ngữ), TTS dùng giọng `vi-VN-NamMinhNeural` để đọc cả phần tiếng Anh → phát âm rất sai.

**Ví dụ thực tế** (từ terminal test):
```
User  : Có thể dạy tôi một câu giao tiếp tiếng Anh đơn giản không?
Detected: vi → voice = vi-VN-NamMinhNeural
AI    : Dạy bạn câu này nè: "What's your name?" – nghĩa là "Bạn tên gì?".
TTS   : lang=vi voice=vi-VN-NamMinhNeural  ← ĐỌC CẢ "What's your name?" BẰNG GIỌNG VIỆT ❌
```

**Root cause:** `tts.py` dùng `conv_lang` (ngôn ngữ cuộc hội thoại) cho toàn bộ text, không detect ngôn ngữ từng đoạn.

**Hướng fix:** Tách text thành segments theo dấu ngoặc kép, detect ngôn ngữ từng segment (dấu Việt → `vi`, latin thuần → `en`, CJK → `zh`/`yue`), synthesize riêng rồi concat bằng ffmpeg.

---

## 🧰 Stack công nghệ

| Lớp | Công nghệ |
|---|---|
| Firmware | C++ Arduino / PlatformIO |
| Backend | Python FastAPI |
| STT | Groq Whisper (free) |
| AI | DeepSeek V4-Flash (gần free) |
| TTS | Edge TTS (free, không giới hạn) |
| Search | Brave Search ($5 credit/tháng) |
| Storage | MicroSD + JSON files |

---

## 🚀 Cách chạy (giai đoạn prototype)

```powershell
# 1. Cài Python deps (sau khi đã có .env và include/config.h)
pip install -r requirements.txt

# 2. Test từng module
python tests/test_stt.py                       # mic → Groq Whisper
python tests/test_llm.py                       # DeepSeek 4 ngôn ngữ
python tests/test_tts.py --all                 # Edge TTS 4 giọng
python tests/test_voices.py                    # so sánh giọng vi-VN & zh-HK

# 3. Pipeline end-to-end
python tests/test_pipeline.py                  # dùng profile.language
python tests/test_pipeline.py --lang auto      # tự nhận dạng ngôn ngữ

# 4. Test báo thức
python tests/test_alarm.py --suite manager     # CRUD offline
python tests/test_alarm.py --suite parser      # NLP 4 ngôn ngữ
python tests/test_alarm.py --suite live --in 5 # nghe chime thật sau 5s

# 5. Firmware ESP32 (khi có hardware)
pio run -t upload
pio device monitor
```

---

## 💰 Roadmap thương mại

| Giai đoạn | Mục tiêu |
|---|---|
| **1** | Personal use, test UX với gia đình |
| **2** | Backend + App phụ huynh |
| **3** | Freemium model (free: 30 câu/ngày · premium: 100k/tháng) |
| **4** | Bán hardware + subscription |
