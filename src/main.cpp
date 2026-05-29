/**
 * AI Voice Assistant - ESP32-S3 (Multilingual)
 * Board   : Waveshare ESP32-S3 LCD 1.47" (ST7789 172x320)
 * Mic     : INMP441 (I2S)
 * Speaker : MAX98357 (I2S) + loa 8Ω
 * Storage : MicroSD (profile + history)
 * Network : WiFi (WiFiManager portal)
 * STT     : Groq Whisper API
 * LLM     : DeepSeek V4 API
 * TTS     : ElevenLabs multilingual_v2
 * Langs   : Tiếng Việt, English, 普通话, 廣東話
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiManager.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>
#include <SPI.h>
#include <SD_MMC.h>
#include <FS.h>
#include <TFT_eSPI.h>
#include <Adafruit_NeoPixel.h>
#include <driver/i2s.h>
#include <Audio.h>
#include <WebSocketsClient.h>

#include "config.h"   // Khóa API + endpoints

// ========================= CẤU HÌNH CHÂN =========================
// LCD ST7789 đã định nghĩa trong platformio.ini qua TFT_eSPI build flags

// INMP441 I2S Microphone (RX) — dùng I2S port 1 (Audio lib cần port 0)
#define I2S_MIC_PORT        I2S_NUM_1
#define I2S_MIC_SCK         4
#define I2S_MIC_WS          5
#define I2S_MIC_SD          6

// MAX98357 I2S Audio (TX) — dùng ESP32-audioI2S
#define I2S_SPK_BCLK        8
#define I2S_SPK_LRC         7
#define I2S_SPK_DIN         9

// Onboard TF card (SDMMC, Waveshare ESP32-S3-LCD-1.47B)
#define SD_CLK              14
#define SD_CMD              15
#define SD_D0               16
#define SD_D1               18
#define SD_D2               17
#define SD_D3               21

// RGB LED (NeoPixel onboard)
#define RGB_LED_PIN         38
#define RGB_LED_COUNT       1

// Nút BOOT để kích hoạt ghi âm
#define BUTTON_PIN          0

// ========================= CẤU HÌNH AUDIO =========================
#define SAMPLE_RATE         16000
#define SAMPLE_BITS         16
#define RECORD_SECONDS_MAX  15        // Giữ câu hỏi ngắn để STT/upload nhanh hơn
#define RECORD_BUFFER_LEN   (SAMPLE_RATE * RECORD_SECONDS_MAX * (SAMPLE_BITS / 8))
#define TTS_IDLE_TIMEOUT_MS 10000
#define DMA_BUF_COUNT       8
#define DMA_BUF_LEN         1024
#define ALWAYS_LISTEN       0         // 0 = đứng yên, chỉ nghe khi bấm nút BOOT
#define MIC_CHUNK_SAMPLES   512
#define VOICE_START_LEVEL   900
#define VOICE_STOP_LEVEL    700
#define VOICE_START_CHUNKS  3
#define VOICE_MIN_MS        500       // Nói ít nhất 0.5s trước khi cho phép dừng
#define VOICE_SILENCE_MS    600       // Im lặng 0.6s thì dừng record
#define CONVO_TIMEOUT_MS    15000     // 15s không nói gì thì thoát conversation mode
#define EDGE_TTS_VOLUME     "+60%"    // Edge TTS prosody gain. Giảm xuống +30% nếu bị rè.

// Wake phrase mac dinh. Sua o day neu sau nay ban muon doi ten goi robot.
#define WAKE_PHRASE         "hey jet"

// ========================= NGÔN NGỮ =========================
enum Language { LANG_VI = 0, LANG_EN, LANG_ZH, LANG_YUE, LANG_COUNT };

struct LanguageCfg {
    const char* whisperCode;     // mã ngôn ngữ Whisper
    const char* displayName;     // hiển thị LCD (Latin)
    const char* voiceId;         // ElevenLabs voice_id
    const char* systemPrompt;    // hướng dẫn LLM
    const char* switchConfirm;   // câu xác nhận khi đổi ngôn ngữ
    uint16_t    accentColor;     // màu chủ đề LCD
};

// Voice ID mặc định của ElevenLabs (đa ngôn ngữ với eleven_multilingual_v2).
// Người dùng có thể đổi voice_id cho từng ngôn ngữ ở đây.
//   Bella   : EXAVITQu4vr4xnSDxMaL
//   Rachel  : 21m00Tcm4TlvDq8ikWAM
//   Antoni  : ErXwobaYiN019PkySvjV
//   Domi    : AZnzlk1XvdvUeBnXmlld
static const LanguageCfg LANGS[LANG_COUNT] = {
    { "vi", "TIENG VIET",  "EXAVITQu4vr4xnSDxMaL",
      "Bạn là người bạn AI vui vẻ của trẻ em 6-12 tuổi. "
      "Trả lời ngắn gọn, tự nhiên bằng tiếng Việt có dấu đầy đủ. "
      "Tuyệt đối không viết tiếng Việt không dấu.",
      "Đã chuyển sang tiếng Việt.",
      TFT_GREEN },
    { "en", "ENGLISH",     "21m00Tcm4TlvDq8ikWAM",
      "You are a friendly voice assistant. Reply briefly in English.",
      "Switched to English.",
      TFT_CYAN },
    { "zh", "MANDARIN",    "EXAVITQu4vr4xnSDxMaL",
      "你是一位友善的语音助手。请用普通话简短回答。",
      "已切换到普通话。",
      TFT_YELLOW },
    { "yue","CANTONESE",   "EXAVITQu4vr4xnSDxMaL",
      "你係一個友善嘅語音助手。請用廣東話簡短回答。",
      "已經轉咗去廣東話。",
      TFT_MAGENTA },
};

// ========================= TRẠNG THÁI =========================
enum AssistantState {
    STATE_BOOT,
    STATE_WIFI_CONNECTING,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_TRANSCRIBING,
    STATE_THINKING,
    STATE_SPEAKING,
    STATE_ERROR
};

// ========================= ĐỐI TƯỢNG TOÀN CỤC =========================
TFT_eSPI            tft = TFT_eSPI();
Adafruit_NeoPixel   rgb(RGB_LED_COUNT, RGB_LED_PIN, NEO_GRB + NEO_KHZ800);
Audio*              audio = nullptr;  // Speaker khởi tạo trong setup để tránh treo trước Serial

AssistantState      currentState  = STATE_BOOT;
Language            currentLang   = LANG_VI;
String              userName      = "Bạn";
String              lastUserText;
String              lastAssistantText;
bool                inConversation = false;  // true = đang trong cuộc hội thoại liên tục

uint8_t*            recordBuffer  = nullptr;
int32_t*            micRawBuffer  = nullptr;
size_t              recordedBytes = 0;
bool                sdMounted     = false;

// Persistent HTTPS clients — giữ TCP + TLS handshake xuyên giữa các lượt
// để khỏi mỗi lần gọi API lại tốn 1.5–3 s bắt tay mới trên mbedTLS.
WiFiClientSecure    gGroqClient;
WiFiClientSecure    gDeepSeekClient;
bool                gGroqClientInit     = false;
bool                gDeepSeekClientInit = false;
HTTPClient          gDeepSeekHttp;

#define MAX_TTS_SEGMENTS 8
String              ttsSegmentTexts[MAX_TTS_SEGMENTS];
Language            ttsSegmentLangs[MAX_TTS_SEGMENTS];
uint8_t             ttsSegmentCount = 0;
uint8_t             ttsSegmentIndex = 0;

// ========================= KHAI BÁO HÀM =========================
void   setState(AssistantState s);
void   updateDisplay(const String& title, const String& body, uint16_t color);
void   drawRobotFace(AssistantState state, uint16_t color);
void   setLED(uint8_t r, uint8_t g, uint8_t b);
bool   initSDCard();
void   loadProfile();
void   saveConversation(const String& user, const String& bot);
bool   initMicI2S();
bool   waitForSpeechStart();
size_t recordAudio();
size_t readMicChunk(int16_t* out, size_t maxSamples, uint32_t* avgAbs);
String sendToGroqWhisper();
String askDeepSeek(const String& userMessage);
String buildSegmentedSystemPrompt();
Language langFromWhisperName(const String& name, const String& transcript, Language fallback);
bool   looksLikeCantonese(const String& text);
String normalizeForEcho(const String& text);
bool   isEchoReply(const String& reply, const String& userMessage);
Language langFromCode(const String& code, Language fallback);
void   resetTtsSegments();
void   prepareSingleTtsSegment(const String& text, Language lang);
void   prepareTtsSegments(const String& answer, Language fallbackLang);
String flattenTtsSegments();
bool   startNextTtsSegment();
bool   speakViaEdgeTTS(const String& text, Language voiceLang);
bool   speakViaElevenLabsSD(const String& text, Language voiceLang);
bool   speakViaGoogleTtsDirect(const String& text, Language voiceLang);
bool   isJunkTranscript(const String& text);
int    detectLanguageSwitch(const String& text);
bool   extractWakeCommand(String& text);
String buildWavHeader(uint32_t pcmBytes);

// ========================= VIETNAMESE ASCII HELPER =========================
// Bỏ dấu tiếng Việt (UTF-8) → ASCII cho LCD fonts không hỗ trợ Unicode
String stripVietnamese(const String& input) {
    // Map 2-byte UTF-8 Vietnamese chars → ASCII
    struct VietMap { const char* utf8; char ascii; };
    static const VietMap map[] = {
        // a
        {"\xC3\xA0",'a'},{"\xC3\xA1",'a'},{"\xC3\xA3",'a'},{"\xE1\xBA\xA1",'a'},
        {"\xC4\x83",'a'},{"\xE1\xBA\xAF",'a'},{"\xE1\xBA\xB1",'a'},{"\xE1\xBA\xB3",'a'},{"\xE1\xBA\xB5",'a'},{"\xE1\xBA\xB7",'a'},
        {"\xC3\xA2",'a'},{"\xE1\xBA\xA5",'a'},{"\xE1\xBA\xA7",'a'},{"\xE1\xBA\xA9",'a'},{"\xE1\xBA\xAB",'a'},{"\xE1\xBA\xAD",'a'},
        // e
        {"\xC3\xA8",'e'},{"\xC3\xA9",'e'},{"\xE1\xBA\xBB",'e'},{"\xE1\xBA\xBD",'e'},{"\xE1\xBA\xB9",'e'},
        {"\xC3\xAA",'e'},{"\xE1\xBB\x81",'e'},{"\xE1\xBA\xBF",'e'},{"\xE1\xBB\x83",'e'},{"\xE1\xBB\x85",'e'},{"\xE1\xBB\x87",'e'},
        // i
        {"\xC3\xAC",'i'},{"\xC3\xAD",'i'},{"\xE1\xBB\x89",'i'},{"\xC4\xA9",'i'},{"\xE1\xBB\x8B",'i'},
        // o
        {"\xC3\xB2",'o'},{"\xC3\xB3",'o'},{"\xE1\xBB\x8F",'o'},{"\xC3\xB5",'o'},{"\xE1\xBB\x8D",'o'},
        {"\xC3\xB4",'o'},{"\xE1\xBB\x91",'o'},{"\xE1\xBB\x93",'o'},{"\xE1\xBB\x95",'o'},{"\xE1\xBB\x97",'o'},{"\xE1\xBB\x99",'o'},
        {"\xC6\xA1",'o'},{"\xE1\xBB\x9B",'o'},{"\xE1\xBB\x9D",'o'},{"\xE1\xBB\x9F",'o'},{"\xE1\xBB\xA1",'o'},{"\xE1\xBB\xA3",'o'},
        // u
        {"\xC3\xB9",'u'},{"\xC3\xBA",'u'},{"\xE1\xBB\xA7",'u'},{"\xC5\xA9",'u'},{"\xE1\xBB\xA5",'u'},
        {"\xC6\xB0",'u'},{"\xE1\xBB\xA9",'u'},{"\xE1\xBB\xAB",'u'},{"\xE1\xBB\xAD",'u'},{"\xE1\xBB\xAF",'u'},{"\xE1\xBB\xB1",'u'},
        // y
        {"\xE1\xBB\xB3",'y'},{"\xC3\xBD",'y'},{"\xE1\xBB\xB7",'y'},{"\xE1\xBB\xB9",'y'},{"\xE1\xBB\xB5",'y'},
        // d
        {"\xC4\x91",'d'},
        // UPPER
        {"\xC3\x80",'A'},{"\xC3\x81",'A'},{"\xC3\x83",'A'},{"\xC4\x82",'A'},{"\xC3\x82",'A'},
        {"\xC3\x88",'E'},{"\xC3\x89",'E'},{"\xC3\x8A",'E'},
        {"\xC3\x8C",'I'},{"\xC3\x8D",'I'},
        {"\xC3\x92",'O'},{"\xC3\x93",'O'},{"\xC3\x95",'O'},{"\xC3\x94",'O'},{"\xC6\xA0",'O'},
        {"\xC3\x99",'U'},{"\xC3\x9A",'U'},{"\xC6\xAF",'U'},
        {"\xC3\x9D",'Y'},
        {"\xC4\x90",'D'},
    };
    String out;
    out.reserve(input.length());
    const char* p = input.c_str();
    while (*p) {
        bool found = false;
        for (const auto& m : map) {
            size_t len = strlen(m.utf8);
            if (strncmp(p, m.utf8, len) == 0) {
                out += m.ascii;
                p += len;
                found = true;
                break;
            }
        }
        if (!found) {
            if ((uint8_t)*p >= 0x80) {
                // Skip unknown multi-byte UTF-8 char
                if ((*p & 0xE0) == 0xC0) p += 2;
                else if ((*p & 0xF0) == 0xE0) p += 3;
                else if ((*p & 0xF8) == 0xF0) p += 4;
                else p++;
                out += '?';
            } else {
                out += *p++;
            }
        }
    }
    return out;
}

// ========================= LCD HELPER =========================
void drawRobotFace(AssistantState state, uint16_t color) {
    int cx = 58;
    int cy = tft.height() / 2 + 6;
    int r  = min(46, max(30, tft.height() / 3));

    uint16_t faceFill = TFT_DARKGREY;
    tft.fillCircle(cx, cy, r, faceFill);
    tft.drawCircle(cx, cy, r, color);
    tft.drawCircle(cx, cy, r - 1, color);

    int eyeY = cy - 12;
    int leftX = cx - 16;
    int rightX = cx + 16;
    uint16_t eyeColor = TFT_WHITE;

    if (state == STATE_THINKING) {
        tft.fillCircle(leftX, eyeY, 5, eyeColor);
        tft.fillCircle(rightX, eyeY, 5, eyeColor);
        tft.fillCircle(leftX + 2, eyeY + 1, 2, TFT_BLACK);
        tft.fillCircle(rightX + 2, eyeY + 1, 2, TFT_BLACK);
        tft.drawString("...", cx - 12, cy + 8);
    } else if (state == STATE_LISTENING || state == STATE_TRANSCRIBING) {
        tft.fillCircle(leftX, eyeY, 6, eyeColor);
        tft.fillCircle(rightX, eyeY, 6, eyeColor);
        tft.fillCircle(leftX, eyeY, 2, TFT_BLACK);
        tft.fillCircle(rightX, eyeY, 2, TFT_BLACK);
        tft.drawCircle(cx, cy + 15, 8, eyeColor);
    } else if (state == STATE_SPEAKING) {
        tft.fillRoundRect(leftX - 7, eyeY - 4, 14, 8, 3, eyeColor);
        tft.fillRoundRect(rightX - 7, eyeY - 4, 14, 8, 3, eyeColor);
        tft.fillRoundRect(cx - 18, cy + 10, 36, 10, 4, eyeColor);
        tft.drawFastVLine(cx - 6, cy + 10, 10, TFT_BLACK);
        tft.drawFastVLine(cx + 6, cy + 10, 10, TFT_BLACK);
    } else if (state == STATE_ERROR) {
        tft.drawLine(leftX - 5, eyeY - 5, leftX + 5, eyeY + 5, TFT_RED);
        tft.drawLine(leftX + 5, eyeY - 5, leftX - 5, eyeY + 5, TFT_RED);
        tft.drawLine(rightX - 5, eyeY - 5, rightX + 5, eyeY + 5, TFT_RED);
        tft.drawLine(rightX + 5, eyeY - 5, rightX - 5, eyeY + 5, TFT_RED);
        tft.drawLine(cx - 14, cy + 18, cx + 14, cy + 18, TFT_RED);
    } else {
        tft.fillCircle(leftX, eyeY, 5, eyeColor);
        tft.fillCircle(rightX, eyeY, 5, eyeColor);
        tft.drawLine(cx - 18, cy + 12, cx - 8, cy + 20, eyeColor);
        tft.drawLine(cx - 8, cy + 20, cx + 8, cy + 20, eyeColor);
        tft.drawLine(cx + 8, cy + 20, cx + 18, cy + 12, eyeColor);
    }
}

void updateDisplay(const String& title, const String& body, uint16_t color) {
    String safeTitle = stripVietnamese(title);
    String safeBody  = stripVietnamese(body);

    tft.fillScreen(TFT_BLACK);
    drawRobotFace(currentState, color);

    int panelX = 116;
    int panelW = tft.width() - panelX - 8;
    int y = 12;

    tft.setTextDatum(TL_DATUM);
    tft.setTextSize(2);
    tft.setTextColor(color, TFT_BLACK);
    tft.drawString(safeTitle, panelX, y);

    // Thanh ngôn ngữ hiện hành
    tft.setTextSize(1);
    tft.setTextColor(LANGS[currentLang].accentColor, TFT_BLACK);
    String safeLangName = stripVietnamese(LANGS[currentLang].displayName);
    tft.drawString(safeLangName, panelX, y + 24);

    tft.setTextColor(TFT_WHITE, TFT_BLACK);

    int x = panelX;
    y = 52;
    int lineH = 12;
    int maxChars = max(8, panelW / 6);
    String word, line;

    for (size_t i = 0; i <= safeBody.length(); i++) {
        char c = i < safeBody.length() ? safeBody[i] : ' ';
        if (c == ' ' || c == '\n' || i == safeBody.length()) {
            if ((line.length() + word.length() + 1) > (size_t)maxChars || c == '\n') {
                tft.drawString(line, x, y);
                y += lineH;
                line = word;
            } else {
                if (line.length()) line += " ";
                line += word;
            }
            word = "";
            if (c == '\n') { tft.drawString(line, x, y); y += lineH; line = ""; }
        } else {
            word += c;
        }
        if (y > tft.height() - lineH) break;
    }
    if (line.length()) tft.drawString(line, x, y);
}

// ========================= LED HELPER =========================
void setLED(uint8_t r, uint8_t g, uint8_t b) {
    rgb.setPixelColor(0, rgb.Color(r, g, b));
    rgb.show();
}

// ========================= STATE MACHINE =========================
void setState(AssistantState s) {
    currentState = s;
    switch (s) {
        case STATE_BOOT:
            setLED(40, 40, 40);
            updateDisplay("BOOTING", "Vui long doi...", TFT_CYAN);
            break;
        case STATE_WIFI_CONNECTING:
            setLED(0, 0, 80);
            updateDisplay("WIFI", "Dang ket noi...", TFT_BLUE);
            break;
        case STATE_IDLE:
            setLED(0, 60, 0);
            updateDisplay("READY",
                String("Chao ") + userName + "!\nNoi de hoi...",
                TFT_GREEN);
            break;
        case STATE_LISTENING:
            setLED(80, 0, 0);
            updateDisplay("LISTENING", "Hay noi ro rang...", TFT_RED);
            break;
        case STATE_TRANSCRIBING:
            setLED(80, 80, 0);
            updateDisplay("STT", "Groq Whisper...", TFT_YELLOW);
            break;
        case STATE_THINKING:
            setLED(80, 0, 80);
            updateDisplay("THINKING",
                String("U: ") + lastUserText + "\n\nDeepSeek...",
                TFT_MAGENTA);
            break;
        case STATE_SPEAKING:
            setLED(0, 80, 80);
            updateDisplay("SPEAKING", lastAssistantText, TFT_CYAN);
            break;
        case STATE_ERROR:
            setLED(80, 0, 0);
            updateDisplay("ERROR", "Xem log Serial.", TFT_RED);
            break;
    }
}

// ========================= MICROSD =========================
bool initSDCard() {
    Serial.println("[SD] SD_MMC.setPins..."); Serial.flush();
    if (!SD_MMC.setPins(SD_CLK, SD_CMD, SD_D0, SD_D1, SD_D2, SD_D3)) {
        Serial.println("[SD] setPins fail");
        sdMounted = false;
        return false;
    }
    Serial.println("[SD] SD_MMC.begin..."); Serial.flush();
    delay(10);
    if (!SD_MMC.begin("/sdcard", false, false, 20000, 5)) {
        Serial.println("[SD] Mount fail");
        sdMounted = false;
        return false;
    }
    sdMounted = true;
    Serial.printf("[SD] OK, %llu MB\n", SD_MMC.cardSize() / (1024ULL * 1024ULL));
    return true;
}

void loadProfile() {
    if (!SD_MMC.exists("/profile.json")) {
        Serial.println("[Profile] Khong co, dung mac dinh");
        return;
    }
    File f = SD_MMC.open("/profile.json", FILE_READ);
    if (!f) return;
    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, f);
    f.close();
    if (err) {
        Serial.printf("[Profile] Parse err: %s\n", err.c_str());
        return;
    }
    if (doc["name"].is<const char*>()) userName = doc["name"].as<String>();
    if (doc["language"].is<const char*>()) {
        String l = doc["language"].as<String>();
        for (int i = 0; i < LANG_COUNT; i++) {
            if (l.equalsIgnoreCase(LANGS[i].whisperCode)) { currentLang = (Language)i; break; }
        }
    }
    Serial.printf("[Profile] %s / %s\n",
                  userName.c_str(), LANGS[currentLang].displayName);
}

void saveConversation(const String& user, const String& bot) {
    if (!sdMounted) return;
    File f = SD_MMC.open("/history.log", FILE_APPEND);
    if (!f) return;
    f.printf("[%lu][%s] U: %s\n[%lu][%s] A: %s\n",
             millis(), LANGS[currentLang].whisperCode, user.c_str(),
             millis(), LANGS[currentLang].whisperCode, bot.c_str());
    f.close();
}

// ========================= I2S MIC =========================
bool initMicI2S() {
    i2s_config_t cfg = {};
    cfg.mode               = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
    cfg.sample_rate        = SAMPLE_RATE;
    cfg.bits_per_sample    = I2S_BITS_PER_SAMPLE_32BIT;
    cfg.channel_format     = I2S_CHANNEL_FMT_ONLY_LEFT;
    cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
    cfg.intr_alloc_flags   = ESP_INTR_FLAG_LEVEL1;
    cfg.dma_buf_count      = DMA_BUF_COUNT;
    cfg.dma_buf_len        = DMA_BUF_LEN;
    cfg.use_apll           = false;

    i2s_pin_config_t pins = {};
    pins.bck_io_num   = I2S_MIC_SCK;
    pins.ws_io_num    = I2S_MIC_WS;
    pins.data_in_num  = I2S_MIC_SD;
    pins.data_out_num = I2S_PIN_NO_CHANGE;

    if (i2s_driver_install(I2S_MIC_PORT, &cfg, 0, nullptr) != ESP_OK) return false;
    if (i2s_set_pin(I2S_MIC_PORT, &pins) != ESP_OK) return false;
    i2s_zero_dma_buffer(I2S_MIC_PORT);
    return true;
}

size_t readMicChunk(int16_t* out, size_t maxSamples, uint32_t* avgAbs) {
    if (!out || maxSamples == 0) return 0;
    if (!micRawBuffer) return 0;
    if (maxSamples > MIC_CHUNK_SAMPLES) maxSamples = MIC_CHUNK_SAMPLES;

    size_t bytesRead = 0;
    esp_err_t err = i2s_read(I2S_MIC_PORT, micRawBuffer, maxSamples * sizeof(int32_t),
                             &bytesRead, pdMS_TO_TICKS(120));
    if (err != ESP_OK || bytesRead == 0) {
        if (avgAbs) *avgAbs = 0;
        return 0;
    }

    size_t samples = bytesRead / sizeof(int32_t);
    uint64_t sumAbs = 0;
    for (size_t i = 0; i < samples; i++) {
        int32_t s = micRawBuffer[i] >> 14;
        if (s > 32767)  s = 32767;
        if (s < -32768) s = -32768;
        int16_t pcm = (int16_t)s;
        out[i] = pcm;
        sumAbs += abs((int)pcm);
    }

    if (avgAbs) *avgAbs = samples ? (uint32_t)(sumAbs / samples) : 0;
    return samples;
}

bool waitForSpeechStart() {
#if ALWAYS_LISTEN
    int16_t pcm[MIC_CHUNK_SAMPLES];
    uint8_t loudChunks = 0;

    while (currentState == STATE_IDLE) {
        if (digitalRead(BUTTON_PIN) == LOW) return true;

        uint32_t level = 0;
        size_t samples = readMicChunk(pcm, MIC_CHUNK_SAMPLES, &level);
        if (samples == 0) {
            delay(10);
            continue;
        }

        if (level >= VOICE_START_LEVEL) {
            loudChunks++;
            if (loudChunks >= VOICE_START_CHUNKS) {
                Serial.printf("[VAD] Speech start, level=%u\n", (unsigned)level);
                return true;
            }
        } else if (loudChunks > 0) {
            loudChunks--;
        }
    }
#endif
    return false;
}

size_t recordAudio() {
    recordedBytes = 0;
    int16_t pcm[MIC_CHUNK_SAMPLES];
    unsigned long start = millis();
    unsigned long lastVoice = start;

    while (recordedBytes < RECORD_BUFFER_LEN - MIC_CHUNK_SAMPLES * sizeof(int16_t) &&
           (millis() - start) < RECORD_SECONDS_MAX * 1000UL) {
        uint32_t level = 0;
        size_t samples = readMicChunk(pcm, MIC_CHUNK_SAMPLES, &level);
        if (samples == 0) continue;

        memcpy(recordBuffer + recordedBytes, pcm, samples * sizeof(int16_t));
        recordedBytes += samples * sizeof(int16_t);

        if (level >= VOICE_STOP_LEVEL) {
            lastVoice = millis();
        }

        if ((millis() - start) > VOICE_MIN_MS &&
            (millis() - lastVoice) > VOICE_SILENCE_MS) {
            break;
        }
    }

    Serial.printf("[REC] %u bytes (%.2fs)\n",
                  (unsigned)recordedBytes,
                  recordedBytes / (float)(SAMPLE_RATE * 2));
    return recordedBytes;
}

// ========================= WAV HEADER =========================
String buildWavHeader(uint32_t pcmBytes) {
    uint8_t h[44];
    uint32_t fileSize = pcmBytes + 36;
    uint32_t byteRate = SAMPLE_RATE * (SAMPLE_BITS / 8);
    uint16_t blockAlign = (SAMPLE_BITS / 8);

    memcpy(h, "RIFF", 4);
    h[4] = fileSize & 0xff; h[5] = (fileSize >> 8) & 0xff;
    h[6] = (fileSize >> 16) & 0xff; h[7] = (fileSize >> 24) & 0xff;
    memcpy(h + 8, "WAVEfmt ", 8);
    h[16] = 16; h[17] = 0; h[18] = 0; h[19] = 0;
    h[20] = 1;  h[21] = 0;
    h[22] = 1;  h[23] = 0;
    h[24] = SAMPLE_RATE & 0xff; h[25] = (SAMPLE_RATE >> 8) & 0xff;
    h[26] = (SAMPLE_RATE >> 16) & 0xff; h[27] = (SAMPLE_RATE >> 24) & 0xff;
    h[28] = byteRate & 0xff; h[29] = (byteRate >> 8) & 0xff;
    h[30] = (byteRate >> 16) & 0xff; h[31] = (byteRate >> 24) & 0xff;
    h[32] = blockAlign & 0xff; h[33] = 0;
    h[34] = SAMPLE_BITS; h[35] = 0;
    memcpy(h + 36, "data", 4);
    h[40] = pcmBytes & 0xff; h[41] = (pcmBytes >> 8) & 0xff;
    h[42] = (pcmBytes >> 16) & 0xff; h[43] = (pcmBytes >> 24) & 0xff;
    return String((char*)h, 44);
}

// ========================= GROQ WHISPER =========================
String sendToGroqWhisper() {
    if (recordedBytes == 0) return "";

    if (!gGroqClientInit) {
        gGroqClient.setInsecure();
        gGroqClient.setTimeout(30000);
        gGroqClientInit = true;
    }

    bool reused = gGroqClient.connected();
    if (!reused) {
        unsigned long ths = millis();
        if (!gGroqClient.connect("api.groq.com", 443)) {
            Serial.println("[GROQ] Connect fail");
            return "";
        }
        Serial.printf("[GROQ] TLS handshake %lu ms\n", millis() - ths);
    } else {
        Serial.println("[GROQ] Reuse connection");
    }

    String boundary = "----ESP32Boundary7MA4YWxkTrZu0gW";
    String head =
        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        "whisper-large-v3\r\n"
        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n"
        "verbose_json\r\n"
        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n"
        "The audio may be Vietnamese, English, Mandarin Chinese, or Cantonese. "
        "Transcribe exactly what the speaker says.\r\n"
        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"voice.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n";
    String wavHeader = buildWavHeader(recordedBytes);
    String tail = "\r\n--" + boundary + "--\r\n";

    size_t contentLength = head.length() + wavHeader.length() + recordedBytes + tail.length();

    gGroqClient.printf("POST /openai/v1/audio/transcriptions HTTP/1.1\r\n");
    gGroqClient.printf("Host: api.groq.com\r\n");
    gGroqClient.printf("Authorization: Bearer %s\r\n", GROQ_API_KEY);
    gGroqClient.printf("Content-Type: multipart/form-data; boundary=%s\r\n", boundary.c_str());
    gGroqClient.printf("Content-Length: %u\r\n", (unsigned)contentLength);
    // KHÔNG gửi "Connection: close" → server giữ TCP/TLS sống, lượt sau khỏi handshake.
    gGroqClient.printf("\r\n");

    gGroqClient.print(head);
    gGroqClient.print(wavHeader);
    const size_t CHUNK = 4096;
    for (size_t i = 0; i < recordedBytes; i += CHUNK) {
        size_t n = min(CHUNK, recordedBytes - i);
        gGroqClient.write(recordBuffer + i, n);
    }
    gGroqClient.print(tail);

    // Parse headers — tìm Content-Length để biết đọc đến đâu thì dừng (giữ
    // connection clean cho lượt sau). Nếu server đột nhiên đóng (server timeout
    // giữa các lượt) thì gọi connect() lại ở lần sau.
    int  contentLen = -1;
    bool keepAlive  = true;   // HTTP/1.1 mặc định keep-alive
    unsigned long tHdr = millis();
    while (gGroqClient.connected() && millis() - tHdr < 15000) {
        String line = gGroqClient.readStringUntil('\n');
        if (line.length() == 0 || line == "\r") break;
        String low = line;
        low.toLowerCase();
        if (low.startsWith("content-length:")) {
            String num = line.substring(15);
            num.trim();
            contentLen = num.toInt();
        } else if (low.startsWith("connection:")) {
            String v = low.substring(11);
            v.trim();
            if (v == "close") keepAlive = false;
        } else if (low.startsWith("transfer-encoding:") &&
                   low.indexOf("chunked") >= 0) {
            contentLen = -1;    // chunked → fallback read-until-close
            keepAlive  = false;
        }
    }

    String body;
    if (contentLen > 0) {
        body.reserve(contentLen + 1);
        unsigned long lastData = millis();
        while ((int)body.length() < contentLen) {
            int avail = gGroqClient.available();
            if (avail > 0) {
                int toRead = min(avail, contentLen - (int)body.length());
                for (int k = 0; k < toRead; k++) body += (char)gGroqClient.read();
                lastData = millis();
            } else if (millis() - lastData > 5000) {
                Serial.println("[GROQ] Body read timeout");
                gGroqClient.stop();
                keepAlive = false;
                break;
            } else {
                delay(1);
            }
        }
    } else {
        // Không biết độ dài → đọc tới khi socket đóng, không thể reuse.
        unsigned long lastData = millis();
        while ((gGroqClient.connected() || gGroqClient.available()) &&
               millis() - lastData < 8000) {
            if (gGroqClient.available()) {
                body += (char)gGroqClient.read();
                lastData = millis();
            } else {
                delay(1);
            }
        }
        gGroqClient.stop();
        keepAlive = false;
    }

    if (!keepAlive) gGroqClient.stop();

    int braceStart = body.indexOf('{');
    if (braceStart < 0) return "";
    body = body.substring(braceStart);
    int braceEnd = body.lastIndexOf('}');
    if (braceEnd > 0) body = body.substring(0, braceEnd + 1);

    JsonDocument doc;
    if (deserializeJson(doc, body)) return "";
    if (!doc["text"].is<const char*>()) return "";
    String text = doc["text"].as<String>();
    text.trim();

    String detectedName = doc["language"].is<const char*>() ? doc["language"].as<String>() : "";
    Language detectedLang = langFromWhisperName(detectedName, text, currentLang);
    if (detectedLang != currentLang) {
        currentLang = detectedLang;
        Serial.printf("[LANG] Auto detect -> %s (%s)\n",
                      LANGS[currentLang].displayName,
                      detectedName.c_str());
    } else if (detectedName.length()) {
        Serial.printf("[LANG] Auto detect kept -> %s (%s)\n",
                      LANGS[currentLang].displayName,
                      detectedName.c_str());
    }
    return text;
}

// ========================= SEGMENTED CHAT / TTS =========================
String buildSegmentedSystemPrompt() {
    return String(LANGS[currentLang].systemPrompt) +
        "\n\nIMPORTANT OUTPUT FORMAT:\n"
        "Return ONLY one JSON object. No markdown, no extra text.\n"
        "The object must be {\"segments\":[{\"lang\":\"vi|en|zh|yue\",\"text\":\"...\"}]}.\n"
        "Never repeat the user's sentence as the whole answer. Answer the question or ask a short clarification.\n"
        "For every segment with lang=vi, write natural Vietnamese with full diacritics. "
        "Never write unaccented Vietnamese like 'Toi muon uong nuoc'.\n"
        "Use the user's language for explanations, but put every pronunciation sample "
        "in its own item with the real language of that sample.\n"
        "Examples:\n"
        "{\"segments\":[{\"lang\":\"vi\",\"text\":\"Câu tiếng Anh là:\"},"
        "{\"lang\":\"en\",\"text\":\"I want to drink water.\"},"
        "{\"lang\":\"vi\",\"text\":\"Nghĩa là: Tôi muốn uống nước.\"}]}\n"
        "If the user asks in Vietnamese for English or Chinese, explain in Vietnamese "
        "and tag English samples as en, Mandarin samples as zh, Cantonese samples as yue.\n"
        "If the user asks in Cantonese for English or Mandarin, explain in Cantonese "
        "and tag English samples as en, Mandarin samples as zh.\n";
}

bool looksLikeCantonese(const String& text) {
    const char* markers[] = {
        "嘅", "咁", "嚟", "喺", "唔", "啲", "咗", "佢", "嗰", "冇",
        "咩", "嘢", "點解", "啦", "㗎", "喎", "噃", "畀"
    };
    for (const char* marker : markers) {
        if (text.indexOf(marker) >= 0) return true;
    }
    return false;
}

Language langFromWhisperName(const String& name, const String& transcript, Language fallback) {
    String n = name;
    n.toLowerCase();
    n.trim();
    if (n == "vi" || n == "vietnamese") return LANG_VI;
    if (n == "en" || n == "english") return LANG_EN;
    if (n == "zh" || n == "chinese" || n == "mandarin") {
        return looksLikeCantonese(transcript) ? LANG_YUE : LANG_ZH;
    }
    if (n == "yue" || n == "cantonese") return LANG_YUE;
    return fallback;
}

String normalizeForEcho(const String& text) {
    String out;
    out.reserve(text.length());
    String t = text;
    t.trim();
    t.toLowerCase();
    for (size_t i = 0; i < t.length(); i++) {
        char c = t[i];
        if (c == '.' || c == ',' || c == '?' || c == '!' ||
            c == ':' || c == ';' || c == '"' || c == '\'' ||
            c == '-' || c == '(' || c == ')' || c == '[' || c == ']') {
            continue;
        }
        out += c;
    }
    out.trim();
    return out;
}

bool isEchoReply(const String& reply, const String& userMessage) {
    String a = normalizeForEcho(reply);
    String b = normalizeForEcho(userMessage);
    if (a.isEmpty() || b.isEmpty()) return false;
    if (a == b) return true;
    return ((a.length() <= b.length() + 8 && b.indexOf(a) >= 0) ||
            (b.length() <= a.length() + 8 && a.indexOf(b) >= 0));
}

Language langFromCode(const String& code, Language fallback) {
    String c = code;
    c.toLowerCase();
    c.trim();
    if (c == "vi") return LANG_VI;
    if (c == "en") return LANG_EN;
    if (c == "zh") return LANG_ZH;
    if (c == "yue") return LANG_YUE;
    return fallback;
}

void resetTtsSegments() {
    for (uint8_t i = 0; i < MAX_TTS_SEGMENTS; i++) {
        ttsSegmentTexts[i] = "";
        ttsSegmentLangs[i] = currentLang;
    }
    ttsSegmentCount = 0;
    ttsSegmentIndex = 0;
}

void prepareSingleTtsSegment(const String& text, Language lang) {
    resetTtsSegments();
    String clean = text;
    clean.trim();
    if (clean.isEmpty()) return;
    ttsSegmentTexts[0] = clean;
    ttsSegmentLangs[0] = lang;
    ttsSegmentCount = 1;
}

void prepareTtsSegments(const String& answer, Language fallbackLang) {
    resetTtsSegments();

    String json = answer;
    json.trim();
    int objStart = json.indexOf('{');
    int objEnd = json.lastIndexOf('}');
    int arrStart = json.indexOf('[');
    int arrEnd = json.lastIndexOf(']');
    if (objStart >= 0 && objEnd > objStart) {
        json = json.substring(objStart, objEnd + 1);
    } else if (arrStart >= 0 && arrEnd > arrStart) {
        json = json.substring(arrStart, arrEnd + 1);
    }

    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, json);
    if (err) {
        prepareSingleTtsSegment(answer, fallbackLang);
        return;
    }

    JsonArray arr;
    if (doc.is<JsonArray>()) {
        arr = doc.as<JsonArray>();
    } else if (doc["segments"].is<JsonArray>()) {
        arr = doc["segments"].as<JsonArray>();
    } else {
        prepareSingleTtsSegment(answer, fallbackLang);
        return;
    }
    for (JsonVariant itemVariant : arr) {
        if (ttsSegmentCount >= MAX_TTS_SEGMENTS) break;
        JsonObject item = itemVariant.as<JsonObject>();
        if (item.isNull()) continue;
        const char* text = item["text"] | "";
        const char* lang = item["lang"] | "";
        String segmentText = String(text);
        segmentText.trim();
        if (segmentText.isEmpty()) continue;

        ttsSegmentTexts[ttsSegmentCount] = segmentText;
        ttsSegmentLangs[ttsSegmentCount] = langFromCode(String(lang), fallbackLang);
        ttsSegmentCount++;
    }

    if (ttsSegmentCount == 0) {
        prepareSingleTtsSegment(answer, fallbackLang);
    }
}

String flattenTtsSegments() {
    String out;
    for (uint8_t i = 0; i < ttsSegmentCount; i++) {
        if (ttsSegmentTexts[i].isEmpty()) continue;
        if (out.length()) out += " ";
        out += ttsSegmentTexts[i];
    }
    return out;
}

bool startNextTtsSegment() {
    while (ttsSegmentIndex < ttsSegmentCount) {
        String text = ttsSegmentTexts[ttsSegmentIndex];
        Language lang = ttsSegmentLangs[ttsSegmentIndex];
        ttsSegmentIndex++;
        text.trim();
        if (text.isEmpty()) continue;
        Serial.printf("[TTS] Segment %u/%u lang=%s\n",
                      (unsigned)ttsSegmentIndex,
                      (unsigned)ttsSegmentCount,
                      LANGS[lang].whisperCode);
        // Edge TTS: giọng tự nhiên, miễn phí (cần SD)
        if (sdMounted && speakViaEdgeTTS(text, lang)) return true;
        // Fallback: Google TTS (không cần SD, giọng robot hơn)
        if (speakViaGoogleTtsDirect(text, lang)) return true;
    }
    return false;
}

// ========================= DEEPSEEK =========================
String askDeepSeek(const String& userMessage) {
    JsonDocument req;
    req["model"]       = "deepseek-chat";
    req["temperature"] = 0.4;
    req["max_tokens"]  = 120;
    req["stream"]      = false;
    JsonObject responseFormat = req["response_format"].to<JsonObject>();
    responseFormat["type"] = "json_object";
    JsonArray msgs = req["messages"].to<JsonArray>();
    JsonObject sys = msgs.add<JsonObject>();
    sys["role"]    = "system";
    sys["content"] = buildSegmentedSystemPrompt();
    JsonObject usr = msgs.add<JsonObject>();
    usr["role"]    = "user";
    usr["content"] = userMessage;

    String payload;
    serializeJson(req, payload);
    Serial.printf("[DeepSeek] POST %u bytes\n", (unsigned)payload.length());

    if (!gDeepSeekClientInit) {
        gDeepSeekClient.setInsecure();
        gDeepSeekClient.setTimeout(60000);
        gDeepSeekHttp.setReuse(true);  // giữ TCP/TLS sống giữa các lượt
        gDeepSeekClientInit = true;
    }

    String reply;
    for (uint8_t attempt = 1; attempt <= 2 && reply.isEmpty(); attempt++) {
        bool reused = gDeepSeekClient.connected();
        unsigned long ths = millis();

        if (!gDeepSeekHttp.begin(gDeepSeekClient, DEEPSEEK_URL)) {
            Serial.println("[DeepSeek] begin fail");
            return "";
        }
        gDeepSeekHttp.useHTTP10(true);  // tránh lỗi body rỗng với chunked response
        gDeepSeekHttp.addHeader("Content-Type", "application/json");
        gDeepSeekHttp.addHeader("Accept", "application/json");
        gDeepSeekHttp.addHeader("Accept-Encoding", "identity");
        gDeepSeekHttp.addHeader("Authorization", String("Bearer ") + DEEPSEEK_API_KEY);
        gDeepSeekHttp.setTimeout(60000);

        int code = gDeepSeekHttp.POST(payload);
        unsigned long dt = millis() - ths;
        Serial.printf("[DeepSeek] attempt=%u HTTP %d (%lu ms, %s)\n",
                      attempt, code, dt, reused ? "reuse" : "new TLS");
        String resp = gDeepSeekHttp.getString();
        Serial.printf("[DeepSeek] body len=%u\n", (unsigned)resp.length());
        if (resp.length()) Serial.printf("[DeepSeek] body: %.220s\n", resp.c_str());

        if (code == HTTP_CODE_OK && resp.length()) {
            JsonDocument doc;
            DeserializationError err = deserializeJson(doc, resp);
            if (!err) {
                reply = doc["choices"][0]["message"]["content"].as<String>();
                reply.trim();
            } else {
                Serial.printf("[DeepSeek] JSON err: %s\n", err.c_str());
            }
        } else {
            Serial.printf("[DeepSeek] err: %s\n", gDeepSeekHttp.errorToString(code).c_str());
        }
        gDeepSeekHttp.end();  // với setReuse(true) → KHÔNG đóng underlying client

        if (reply.isEmpty() && attempt < 2) {
            Serial.println("[DeepSeek] empty/invalid reply, retry...");
            // Fail có thể do server đã đóng kết nối → ép drop để lần 2 handshake lại.
            gDeepSeekClient.stop();
            delay(350);
        }
    }
    return reply;
}

// ========================= EDGE TTS (Microsoft) =========================
// Giọng tự nhiên, miễn phí, qua WebSocket đến speech.platform.bing.com
// Lưu MP3 vào SD rồi phát qua ESP32-audioI2S.

static File    edgeTtsFile;
static bool    edgeTtsDone;
static bool    edgeTtsConnected;
static int     edgeTtsBytes;

static const char* EDGE_VOICES[] = {
    "vi-VN-NamMinhNeural",     // LANG_VI - same as tests/test.py / py/common.py
    "en-US-AriaNeural",        // LANG_EN - same as tests/test.py / py/common.py
    "zh-CN-XiaoxiaoNeural",    // LANG_ZH
    "zh-HK-HiuMaanNeural",     // LANG_YUE - same as tests/test.py / py/common.py
};

void edgeTtsEvent(WStype_t type, uint8_t* payload, size_t length) {
    switch (type) {
        case WStype_CONNECTED:
            Serial.println("[EDGE] WS connected");
            edgeTtsConnected = true;
            break;
        case WStype_DISCONNECTED:
            Serial.println("[EDGE] WS disconnected");
            edgeTtsDone = true;
            break;
        case WStype_TEXT: {
            String msg((char*)payload, length);
            if (msg.indexOf("turn.end") >= 0) {
                Serial.println("[EDGE] turn.end");
                edgeTtsDone = true;
            }
            break;
        }
        case WStype_BIN: {
            if (length < 2) break;
            uint16_t hdrLen = ((uint16_t)payload[0] << 8) | payload[1];
            size_t audioStart = 2 + hdrLen;
            if (audioStart >= length) break;
            size_t audioLen = length - audioStart;
            if (edgeTtsFile && audioLen > 0) {
                edgeTtsFile.write(payload + audioStart, audioLen);
                edgeTtsBytes += audioLen;
            }
            break;
        }
        default: break;
    }
}

bool speakViaEdgeTTS(const String& text, Language voiceLang) {
    if (!audio || !sdMounted) return false;

    String clean = text;
    clean.trim();
    if (clean.isEmpty()) return false;

    // XML escape
    clean.replace("&", "&amp;");
    clean.replace("<", "&lt;");
    clean.replace(">", "&gt;");
    clean.replace("\"", "&quot;");

    const char* voice = EDGE_VOICES[voiceLang < LANG_COUNT ? voiceLang : 0];
    Serial.printf("[EDGE] voice=%s len=%u\n", voice, (unsigned)clean.length());

    // Mở file trên SD
    if (SD_MMC.exists("/tts.mp3")) SD_MMC.remove("/tts.mp3");
    edgeTtsFile = SD_MMC.open("/tts.mp3", FILE_WRITE);
    if (!edgeTtsFile) {
        Serial.println("[EDGE] SD open fail");
        return false;
    }

    edgeTtsDone = false;
    edgeTtsConnected = false;
    edgeTtsBytes = 0;

    WebSocketsClient ws;
    ws.onEvent(edgeTtsEvent);
    ws.setExtraHeaders("Origin: chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold");
    ws.beginSSL("speech.platform.bing.com", 443,
        "/consumer/speech/synthesize/readaloud/token/6A5AA1D4EAFF4E9FB37E23D68491D6F4");

    // Chờ kết nối (max 10s)
    unsigned long t0 = millis();
    while (!edgeTtsConnected && !edgeTtsDone && millis() - t0 < 10000) {
        ws.loop();
        delay(10);
    }
    if (!edgeTtsConnected) {
        Serial.println("[EDGE] Connect timeout");
        edgeTtsFile.close();
        SD_MMC.remove("/tts.mp3");
        return false;
    }

    // Gửi config (output format)
    ws.sendTXT(
        "Content-Type:application/json; charset=utf-8\r\n"
        "Path:speech.config\r\n\r\n"
        "{\"context\":{\"synthesis\":{\"audio\":{"
        "\"metadataoptions\":{\"sentenceBoundaryEnabled\":\"false\","
        "\"wordBoundaryEnabled\":\"false\"},"
        "\"outputFormat\":\"audio-24khz-48kbitrate-mono-mp3\"}}}}"
    );
    delay(50);
    ws.loop();

    // Tạo request ID (32 hex chars)
    String reqId;
    for (int i = 0; i < 32; i++) reqId += String(random(16), HEX);

    // Gửi SSML
    String ssml = String("X-RequestId:") + reqId +
        "\r\nContent-Type:application/ssml+xml\r\nPath:ssml\r\n\r\n"
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>";
    ssml += "<voice name='";
    ssml += voice;
    ssml += "'><prosody pitch='+0Hz' rate='+0%' volume='";
    ssml += EDGE_TTS_VOLUME;
    ssml += "'>";
    ssml += clean;
    ssml += "</prosody></voice></speak>";
    ws.sendTXT(ssml);

    // Chờ nhận audio (max 30s)
    t0 = millis();
    while (!edgeTtsDone && millis() - t0 < 30000) {
        ws.loop();
        delay(1);
    }

    edgeTtsFile.close();
    ws.disconnect();

    Serial.printf("[EDGE] Saved %d bytes to /tts.mp3\n", edgeTtsBytes);
    if (edgeTtsBytes < 100) {
        SD_MMC.remove("/tts.mp3");
        return false;
    }

    bool ok = audio->connecttoFS(SD_MMC, "/tts.mp3");
    Serial.printf("[EDGE] connecttoFS: %d\n", ok);
    return ok;
}

// ========================= ELEVENLABS TTS VIA SD =========================
// Có microSD: tải MP3 từ ElevenLabs, lưu /tts.mp3 rồi phát qua ESP32-audioI2S.
bool speakViaElevenLabsSD(const String& text, Language voiceLang) {
    if (!audio || !sdMounted) return false;

    String clean = text;
    clean.trim();
    if (clean.isEmpty()) return false;

    WiFiClientSecure client;
    client.setInsecure();
    client.setTimeout(60000);
    HTTPClient https;

    String url = String("https://") + ELEVENLABS_HOST +
                 "/v1/text-to-speech/" + LANGS[voiceLang].voiceId;

    if (!https.begin(client, url)) {
        Serial.println("[TTS] ElevenLabs begin fail");
        return false;
    }
    https.addHeader("xi-api-key", ELEVENLABS_API_KEY);
    https.addHeader("Content-Type", "application/json");
    https.addHeader("Accept", "audio/mpeg");
    https.setTimeout(60000);

    JsonDocument req;
    req["text"]     = clean;
    req["model_id"] = ELEVENLABS_MODEL;
    JsonObject vs = req["voice_settings"].to<JsonObject>();
    vs["stability"]         = 0.5;
    vs["similarity_boost"]  = 0.75;
    vs["style"]             = 0.0;
    vs["use_speaker_boost"] = true;

    String payload;
    serializeJson(req, payload);
    Serial.printf("[TTS] ElevenLabs POST len=%u\n", (unsigned)clean.length());

    int code = https.POST(payload);
    if (code != HTTP_CODE_OK) {
        Serial.printf("[TTS] ElevenLabs HTTP %d: %s\n",
                      code, https.errorToString(code).c_str());
        String body = https.getString();
        if (body.length()) Serial.printf("[TTS] body: %.160s\n", body.c_str());
        https.end();
        return false;
    }

    if (SD_MMC.exists("/tts.mp3")) SD_MMC.remove("/tts.mp3");
    File mp3 = SD_MMC.open("/tts.mp3", FILE_WRITE);
    if (!mp3) {
        Serial.println("[TTS] SD open /tts.mp3 fail");
        https.end();
        return false;
    }

    WiFiClient* stream = https.getStreamPtr();
    uint8_t buf[1024];
    int total = https.getSize();
    int written = 0;
    unsigned long lastData = millis();
    while (https.connected() && (total < 0 || written < total)) {
        size_t avail = stream->available();
        if (avail) {
            int n = stream->readBytes(buf, min(avail, sizeof(buf)));
            if (n > 0) {
                mp3.write(buf, n);
                written += n;
                lastData = millis();
            }
        } else if (millis() - lastData > TTS_IDLE_TIMEOUT_MS) {
            break;
        } else {
            delay(1);
        }
    }
    mp3.close();
    https.end();

    Serial.printf("[TTS] ElevenLabs saved %d bytes\n", written);
    if (written < 100) return false;

    bool ok = audio->connecttoFS(SD_MMC, "/tts.mp3");
    Serial.printf("[TTS] connecttoFS returned: %d, isRunning: %d\n", ok, audio->isRunning());
    return ok;
}

// ========================= GOOGLE TTS DIRECT =========================
// Google Translate TTS: miễn phí, giới hạn ~200 ký tự/request.
// Nếu text dài hơn, tách thành câu nhỏ rồi phát từng câu blocking.
bool speakViaGoogleTtsDirect(const String& text, Language voiceLang) {
    String clean = text;
    clean.trim();
    if (clean.isEmpty()) return false;

    const char* langCode = "vi";
    switch (voiceLang) {
        case LANG_VI:  langCode = "vi";    break;
        case LANG_EN:  langCode = "en";    break;
        case LANG_ZH:  langCode = "zh-CN"; break;
        case LANG_YUE: langCode = "yue";   break;
        default:       langCode = "vi";    break;
    }

    if (!audio) return false;

    // Nếu text ngắn, phát trực tiếp (non-blocking, loop() sẽ xử lý)
    if (clean.length() <= 180) {
        Serial.printf("[TTS] Google lang=%s len=%u\n", langCode, (unsigned)clean.length());
        bool ok = audio->connecttospeech(clean.c_str(), langCode);
        Serial.printf("[TTS] connecttospeech: %d\n", ok);
        return ok;
    }

    // Text dài: tách theo dấu câu, phát blocking từng phần
    Serial.printf("[TTS] Google LONG text lang=%s len=%u, splitting...\n",
                  langCode, (unsigned)clean.length());
    int start = 0;
    bool anyOk = false;
    while (start < (int)clean.length()) {
        // Tìm điểm cắt tốt nhất trong 180 ký tự
        int end = start + 180;
        if (end >= (int)clean.length()) {
            end = clean.length();
        } else {
            // Tìm dấu câu gần nhất để cắt
            int bestCut = -1;
            const char* delims = ".!?,;:\n";
            for (int i = end; i > start + 20; i--) {
                if (strchr(delims, clean[i])) {
                    bestCut = i + 1;
                    break;
                }
            }
            // Nếu không tìm thấy dấu câu, tìm khoảng trắng
            if (bestCut < 0) {
                for (int i = end; i > start + 20; i--) {
                    if (clean[i] == ' ') { bestCut = i + 1; break; }
                }
            }
            if (bestCut > start) end = bestCut;
        }

        String chunk = clean.substring(start, end);
        chunk.trim();
        start = end;
        if (chunk.isEmpty()) continue;

        Serial.printf("[TTS] Chunk len=%u: %.40s...\n", (unsigned)chunk.length(), chunk.c_str());
        bool ok = audio->connecttospeech(chunk.c_str(), langCode);
        if (ok) {
            anyOk = true;
            // Chờ phát xong chunk này trước khi phát chunk tiếp
            unsigned long t0 = millis();
            while (audio->isRunning() && millis() - t0 < 15000) {
                audio->loop();
            }
        }
    }
    return anyOk;
}

// ========================= WAKE PHRASE =========================
// Chi phan hoi khi cau noi co WAKE_PHRASE, vi du: "hey Jet hom nay the nao?"
// Neu muon doi ten goi robot sau nay, sua #define WAKE_PHRASE o phan cau hinh audio.
bool extractWakeCommand(String& text) {
    String lower = text;
    lower.toLowerCase();

    int pos = lower.indexOf(WAKE_PHRASE);
    int endPos = pos >= 0 ? pos + strlen(WAKE_PHRASE) : -1;

    if (pos < 0) {
        const char* variants[] = {
            "hay jet", "hey jet", "hay j", "hey j",
            "hay chet", "hey chet", "hey jack", "hay ch", "hey ch",
            "e jet", "eh jet"
        };
        for (size_t i = 0; i < sizeof(variants) / sizeof(variants[0]); i++) {
            int vpos = lower.indexOf(variants[i]);
            if (vpos < 0) continue;

            pos = vpos;
            endPos = vpos + strlen(variants[i]);
            if (strcmp(variants[i], "hay ch") == 0 || strcmp(variants[i], "hey ch") == 0) {
                int firstSpace = lower.indexOf(' ', vpos);
                int secondSpace = firstSpace >= 0 ? lower.indexOf(' ', firstSpace + 1) : -1;
                endPos = secondSpace >= 0 ? secondSpace : text.length();
            }
            break;
        }
    }

    if (pos < 0) {
        int heyPos = lower.indexOf("hey");
        int jetPos = heyPos >= 0 ? lower.indexOf("jet", heyPos + 3) : -1;
        if (heyPos >= 0 && jetPos >= 0 && (jetPos - heyPos) <= 8) {
            pos = heyPos;
            endPos = jetPos + 3;
        }
    }

    if (pos < 0) return false;

    text = text.substring(endPos);
    text.trim();

    while (text.startsWith(",") || text.startsWith(".") ||
           text.startsWith(":") || text.startsWith(";") ||
           text.startsWith("-") || text.startsWith("!")) {
        text = text.substring(1);
        text.trim();
    }

    return true;
}

// ========================= STT FILTER =========================
bool isJunkTranscript(const String& text) {
    String t = text;
    t.toLowerCase();
    t.trim();

    if (t.length() < 2) return true;

    // Whisper hay sinh các câu này khi mic thu im lặng/nhiễu nền.
    if (t.indexOf("subscribe") >= 0) return true;
    if (t.indexOf("youtube") >= 0) return true;
    if (t.indexOf("video") >= 0) return true;
    if (t.indexOf("channel") >= 0) return true;
    if (t.indexOf("like") >= 0 && t.indexOf("share") >= 0) return true;
    if (t.indexOf("la la school") >= 0) return true;
    if (t.indexOf("dang ky") >= 0 || t.indexOf("đăng ký") >= 0) return true;
    if (t.indexOf("kenh") >= 0 || t.indexOf("kênh") >= 0) return true;
    if (t.indexOf("youtube channel") >= 0) return true;
    if (t.indexOf("khong bo lo") >= 0 || t.indexOf("không bỏ lỡ") >= 0) return true;
    if (t.indexOf("nhung video hap dan") >= 0 || t.indexOf("những video hấp dẫn") >= 0) return true;
    if (t.indexOf("cam on cac ban da xem") >= 0 || t.indexOf("cảm ơn các bạn đã xem") >= 0) return true;
    if (t.indexOf("hen gap lai") >= 0 || t.indexOf("hẹn gặp lại") >= 0) return true;
    if (t.indexOf("nho like") >= 0 || t.indexOf("nhớ like") >= 0) return true;

    return false;
}

// ========================= LANGUAGE SWITCH DETECTION =========================
// Trả về Language mới hoặc -1 nếu không phải lệnh đổi ngôn ngữ.
int detectLanguageSwitch(const String& text) {
    String t = text;
    t.toLowerCase();

    bool wantsSwitch =
        t.indexOf("chuyen sang") >= 0 || t.indexOf("chuyển sang") >= 0 ||
        t.indexOf("doi sang")    >= 0 || t.indexOf("đổi sang") >= 0 ||
        t.indexOf("switch to")   >= 0 || t.indexOf("change to") >= 0 ||
        t.indexOf("speak ")      >= 0 ||
        t.indexOf("切换")        >= 0 || t.indexOf("切換") >= 0 ||
        t.indexOf("转去")        >= 0 || t.indexOf("轉去") >= 0 ||
        t.indexOf("讲")          >= 0 || t.indexOf("講")   >= 0;

    // Không yêu cầu chính xác phải có chuyển; bắt nhãn ngôn ngữ là đủ.
    int target = -1;
    if (t.indexOf("tieng viet") >= 0 || t.indexOf("tiếng việt") >= 0 ||
        t.indexOf("vietnamese") >= 0 || t.indexOf("越南") >= 0)
        target = LANG_VI;
    else if (t.indexOf("tieng anh") >= 0 || t.indexOf("tiếng anh") >= 0 ||
             t.indexOf("english")   >= 0 || t.indexOf("英文") >= 0 ||
             t.indexOf("英語")      >= 0 || t.indexOf("英语") >= 0)
        target = LANG_EN;
    else if (t.indexOf("pho thong") >= 0 || t.indexOf("phổ thông") >= 0 ||
             t.indexOf("mandarin")  >= 0 || t.indexOf("普通话") >= 0 ||
             t.indexOf("普通話")    >= 0 || t.indexOf("国语") >= 0 ||
             t.indexOf("國語")      >= 0)
        target = LANG_ZH;
    else if (t.indexOf("quang dong") >= 0 || t.indexOf("quảng đông") >= 0 ||
             t.indexOf("cantonese")  >= 0 || t.indexOf("廣東") >= 0 ||
             t.indexOf("广东")       >= 0 || t.indexOf("粵語") >= 0 ||
             t.indexOf("粤语")       >= 0)
        target = LANG_YUE;

    if (target < 0) return -1;
    if (target == currentLang && !wantsSwitch) return -1;  // chỉ nhắc tên thôi
    return target;
}

// ========================= SETUP =========================
void setup() {
    Serial.begin(115200);
    delay(3000);  // Đợi USB-CDC reconnect sau reset để Serial Monitor bắt kịp
    Serial.println("\n=== AI Voice Assistant (Multilingual) ===");

    pinMode(BUTTON_PIN, INPUT_PULLUP);

    if (!psramFound()) {
        Serial.println("[FATAL] Khong thay PSRAM");
        while (true) delay(1000);
    }
    Serial.printf("[PSRAM] %u bytes free\n", (unsigned)ESP.getFreePsram());
    recordBuffer = (uint8_t*)ps_malloc(RECORD_BUFFER_LEN);
    if (!recordBuffer) {
        Serial.println("[FATAL] ps_malloc fail");
        while (true) delay(1000);
    }
    micRawBuffer = (int32_t*)heap_caps_malloc(MIC_CHUNK_SAMPLES * sizeof(int32_t), MALLOC_CAP_DMA);
    if (!micRawBuffer) {
        Serial.println("[FATAL] mic buffer malloc fail");
        while (true) delay(1000);
    }

    Serial.println("[TFT] init start...");
    Serial.flush();
    tft.init();
    Serial.println("[TFT] init done");
    Serial.flush();
    tft.setRotation(1);  // Landscape: chữ và mặt robot nằm ngang
    pinMode(TFT_BL, OUTPUT);
    digitalWrite(TFT_BL, HIGH);
    tft.fillScreen(TFT_BLACK);
    Serial.println("[TFT] display ready");
    Serial.flush();

    Serial.println("[RGB] init..."); Serial.flush();
    rgb.begin();
    rgb.setBrightness(50);
    rgb.show();
    Serial.println("[RGB] done"); Serial.flush();

    setState(STATE_BOOT);
    delay(500);

    updateDisplay("SD INIT", "Dang doc the nho...", TFT_YELLOW);
    if (!initSDCard()) {
        Serial.println("[WARN] Khong co SD, fallback Google TTS");
    } else {
        loadProfile();
    }

    updateDisplay("MIC INIT", "Dang khoi dong mic...", TFT_YELLOW);
    Serial.println("[MIC] init..."); Serial.flush();
    if (!initMicI2S()) {
        Serial.println("[FATAL] Mic init fail");
        setState(STATE_ERROR);
        while (true) delay(1000);
    }
    Serial.println("[MIC] done"); Serial.flush();

    updateDisplay("AUDIO INIT", "Dang khoi dong loa...", TFT_CYAN);
    Serial.println("[AUDIO] init..."); Serial.flush();
    audio = new Audio(false, 3, I2S_NUM_0);  // Speaker dùng I2S port 0 (Audio lib yêu cầu), mic dùng port 1
    if (!audio) {
        Serial.println("[FATAL] Audio malloc fail");
        setState(STATE_ERROR);
        while (true) delay(1000);
    }
    audio->setPinout(I2S_SPK_BCLK, I2S_SPK_LRC, I2S_SPK_DIN);
    audio->setVolume(21);  // MAX volume (0-21)
    Serial.println("[AUDIO] done"); Serial.flush();

    setState(STATE_WIFI_CONNECTING);
    WiFiManager wm;
    wm.setConfigPortalTimeout(180);
    if (!wm.autoConnect("ESP32-Voice-Setup")) {
        Serial.println("[WiFi] Timeout");
        setState(STATE_ERROR);
        ESP.restart();
    }
    Serial.printf("[WiFi] OK %s\n", WiFi.localIP().toString().c_str());

    // Phát câu chào khi khởi động xong
    Serial.println("[BOOT] Playing greeting..."); Serial.flush();
    updateDisplay("READY", "Xin chao! Hay noi gi di...", TFT_GREEN);
    if (audio->connecttospeech("Xin chào! Mình là Jet, sẵn sàng nghe bạn.", "vi")) {
        unsigned long t0 = millis();
        while (audio->isRunning() && millis() - t0 < 10000) {
            audio->loop();
        }
    }
    Serial.println("[BOOT] Ready"); Serial.flush();

    setState(STATE_IDLE);
}

// ========================= LOOP =========================
void loop() {
    // --- Đang phát TTS ---
    if (currentState == STATE_SPEAKING) {
        if (audio) audio->loop();
        if (!audio || !audio->isRunning()) {
            if (startNextTtsSegment()) {
                return;  // còn segment tiếp theo
            }
            saveConversation(lastUserText, lastAssistantText);
            resetTtsSegments();
            // Trả lời xong thì đứng yên, không tự nghe tiếp để tránh mic bắt tiếng nền/YouTube.
            inConversation = false;
            Serial.println("[CONVO] TTS done, idle.");
            setState(STATE_IDLE);
        }
        return;
    }

    // --- Chờ người dùng nói ---
    if (currentState == STATE_IDLE) {
        bool triggered = false;

        if (digitalRead(BUTTON_PIN) == LOW) {
            delay(40);
            if (digitalRead(BUTTON_PIN) == LOW) {
                triggered = true;
                inConversation = false;
            }
        }

        if (!triggered) {
            if (inConversation) {
                // Đang trong conversation mode: chờ có tiếng nói hoặc timeout
                unsigned long waitStart = millis();
                int16_t pcm[MIC_CHUNK_SAMPLES];
                uint8_t loudChunks = 0;
                while (millis() - waitStart < CONVO_TIMEOUT_MS) {
                    if (digitalRead(BUTTON_PIN) == LOW) { triggered = true; break; }
                    uint32_t level = 0;
                    size_t samples = readMicChunk(pcm, MIC_CHUNK_SAMPLES, &level);
                    if (samples == 0) { delay(5); continue; }
                    if (level >= VOICE_START_LEVEL) {
                        loudChunks++;
                        if (loudChunks >= VOICE_START_CHUNKS) {
                            triggered = true;
                            Serial.printf("[VAD] Speech in convo, level=%u\n", (unsigned)level);
                            break;
                        }
                    } else if (loudChunks > 0) {
                        loudChunks--;
                    }
                }
                if (!triggered) {
                    // Timeout — thoát conversation mode
                    Serial.println("[CONVO] Timeout, exiting conversation");
                    inConversation = false;
                    updateDisplay("READY", "San sang nghe...", TFT_GREEN);
                    return;
                }
            } else {
                // Ngoài conversation: chờ wake word hoặc nút bấm
                triggered = waitForSpeechStart();
            }
        }

        if (!triggered) return;

        unsigned long turnStartMs = millis();
        setState(STATE_LISTENING);
        size_t n = recordAudio();
        unsigned long recDoneMs = millis();
        if (n < SAMPLE_RATE) {
            updateDisplay("TOO SHORT", "Noi lau hon mot chut.", TFT_ORANGE);
            delay(1000);
            setState(STATE_IDLE);
            return;
        }

        setState(STATE_TRANSCRIBING);
        String transcript = sendToGroqWhisper();
        unsigned long sttDoneMs = millis();
        transcript.trim();
        if (transcript.isEmpty()) {
            updateDisplay("NO INPUT", "Thu lai.", TFT_ORANGE);
            delay(1000);
            setState(STATE_IDLE);
            return;
        }
        Serial.printf("[STT] %s\n", transcript.c_str());

        if (isJunkTranscript(transcript)) {
            Serial.printf("[STT] Ignored junk: %s\n", transcript.c_str());
            setState(STATE_IDLE);
            return;
        }

        lastUserText = transcript;
        inConversation = false;  // Push-to-talk: trả lời xong quay về trạng thái tĩnh
        Serial.printf("[CMD] %s\n", transcript.c_str());

        // Phát hiện lệnh đổi ngôn ngữ
        int newLang = detectLanguageSwitch(transcript);
        if (newLang >= 0) {
            currentLang = (Language)newLang;
            Serial.printf("[LANG] Switch -> %s\n", LANGS[currentLang].displayName);
            lastAssistantText = LANGS[currentLang].switchConfirm;
            prepareSingleTtsSegment(lastAssistantText, currentLang);
            if (startNextTtsSegment()) {
                setState(STATE_SPEAKING);
            } else {
                updateDisplay("LANG OK", LANGS[currentLang].displayName,
                              LANGS[currentLang].accentColor);
                delay(1500);
                setState(STATE_IDLE);
            }
            return;
        }

        setState(STATE_THINKING);
        String reply = askDeepSeek(transcript);
        unsigned long llmDoneMs = millis();
        reply.trim();
        if (reply.isEmpty() || isEchoReply(reply, transcript)) {
            reply = "Mạng AI đang chậm nên mình chưa trả lời được. Bạn hỏi lại ngắn hơn một chút nhé.";
            prepareSingleTtsSegment(reply, currentLang);
        } else {
            prepareTtsSegments(reply, currentLang);
        }
        lastAssistantText = flattenTtsSegments();
        if (lastAssistantText.isEmpty()) lastAssistantText = reply;
        if (isEchoReply(lastAssistantText, transcript)) {
            reply = "Mạng AI đang chậm nên mình chưa trả lời được. Bạn hỏi lại ngắn hơn một chút nhé.";
            prepareSingleTtsSegment(reply, currentLang);
            lastAssistantText = reply;
        }
        Serial.printf("[LLM raw] %s\n", reply.c_str());
        Serial.printf("[LLM text] %s\n", lastAssistantText.c_str());
        Serial.printf("[TIME] rec=%lums stt=%lums llm=%lums total_to_text=%lums\n",
                      recDoneMs - turnStartMs,
                      sttDoneMs - recDoneMs,
                      llmDoneMs - sttDoneMs,
                      llmDoneMs - turnStartMs);

        if (startNextTtsSegment()) {
            setState(STATE_SPEAKING);
        } else {
            // TTS fail: thử phát bằng Google TTS trực tiếp
            Serial.println("[TTS] ElevenLabs fail, trying Google...");
            if (speakViaGoogleTtsDirect(lastAssistantText, currentLang)) {
                setState(STATE_SPEAKING);
            } else {
                updateDisplay("TTS FAIL", lastAssistantText, TFT_ORANGE);
                delay(2500);
                saveConversation(lastUserText, lastAssistantText);
                setState(STATE_IDLE);
            }
        }
    }

    delay(5);
}

// ========================= CALLBACKS AUDIO LIB =========================
void audio_info(const char* info)    { Serial.printf("[AUDIO] %s\n", info); }
void audio_eof_mp3(const char* info) { Serial.printf("[AUDIO] EOF: %s\n", info); }
