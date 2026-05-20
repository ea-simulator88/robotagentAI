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
#include <SD.h>
#include <FS.h>
#include <TFT_eSPI.h>
#include <Adafruit_NeoPixel.h>
#include <driver/i2s.h>
#include <Audio.h>

#include "config.h"   // Khóa API + endpoints

// ========================= CẤU HÌNH CHÂN =========================
// LCD ST7789 đã định nghĩa trong platformio.ini qua TFT_eSPI build flags

// INMP441 I2S Microphone (RX)
#define I2S_MIC_PORT        I2S_NUM_0
#define I2S_MIC_SCK         4
#define I2S_MIC_WS          5
#define I2S_MIC_SD          6

// MAX98357 I2S Audio (TX) — dùng ESP32-audioI2S
#define I2S_SPK_BCLK        15
#define I2S_SPK_LRC         16
#define I2S_SPK_DIN         7

// MicroSD (SPI riêng)
#define SD_CS               34
#define SD_MOSI             35
#define SD_MISO             37
#define SD_SCLK             36

// RGB LED (NeoPixel onboard)
#define RGB_LED_PIN         38
#define RGB_LED_COUNT       1

// Nút BOOT để kích hoạt ghi âm
#define BUTTON_PIN          0

// ========================= CẤU HÌNH AUDIO =========================
#define SAMPLE_RATE         16000
#define SAMPLE_BITS         16
#define RECORD_SECONDS_MAX  10
#define RECORD_BUFFER_LEN   (SAMPLE_RATE * RECORD_SECONDS_MAX * (SAMPLE_BITS / 8))
#define TTS_IDLE_TIMEOUT_MS 10000
#define DMA_BUF_COUNT       8
#define DMA_BUF_LEN         1024
#define ALWAYS_LISTEN       1
#define MIC_CHUNK_SAMPLES   512
#define VOICE_START_LEVEL   900
#define VOICE_STOP_LEVEL    450
#define VOICE_START_CHUNKS  3
#define VOICE_MIN_MS        900
#define VOICE_SILENCE_MS    1000

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
      "Bạn là trợ lý ảo thân thiện. Trả lời ngắn gọn bằng tiếng Việt.",
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
SPIClass            sdSPI(HSPI);
Audio               audio;

AssistantState      currentState  = STATE_BOOT;
Language            currentLang   = LANG_VI;
String              userName      = "Bạn";
String              lastUserText;
String              lastAssistantText;

uint8_t*            recordBuffer  = nullptr;
int32_t*            micRawBuffer  = nullptr;
size_t              recordedBytes = 0;

#define MAX_TTS_SEGMENTS 8
String              ttsSegmentTexts[MAX_TTS_SEGMENTS];
Language            ttsSegmentLangs[MAX_TTS_SEGMENTS];
uint8_t             ttsSegmentCount = 0;
uint8_t             ttsSegmentIndex = 0;

// ========================= KHAI BÁO HÀM =========================
void   setState(AssistantState s);
void   updateDisplay(const String& title, const String& body, uint16_t color);
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
Language langFromCode(const String& code, Language fallback);
void   resetTtsSegments();
void   prepareSingleTtsSegment(const String& text, Language lang);
void   prepareTtsSegments(const String& answer, Language fallbackLang);
String flattenTtsSegments();
bool   startNextTtsSegment();
bool   speakViaElevenLabs(const String& text, Language voiceLang);
int    detectLanguageSwitch(const String& text);
bool   extractWakeCommand(String& text);
String buildWavHeader(uint32_t pcmBytes);

// ========================= LCD HELPER =========================
void updateDisplay(const String& title, const String& body, uint16_t color) {
    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(color, TFT_BLACK);
    tft.setTextDatum(TC_DATUM);
    tft.setTextSize(2);
    tft.drawString(title, tft.width() / 2, 8);

    // Thanh ngôn ngữ hiện hành
    tft.setTextSize(1);
    tft.setTextColor(LANGS[currentLang].accentColor, TFT_BLACK);
    tft.drawString(LANGS[currentLang].displayName, tft.width() / 2, 32);

    tft.setTextDatum(TL_DATUM);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);

    int x = 4, y = 56;
    int lineH = 12;
    int maxChars = (tft.width() - 8) / 6;
    String word, line;

    for (size_t i = 0; i <= body.length(); i++) {
        char c = i < body.length() ? body[i] : ' ';
        if (c == ' ' || c == '\n' || i == body.length()) {
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
                String("Chao ") + userName + "!\nNoi: Hey Jet...",
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
    sdSPI.begin(SD_SCLK, SD_MISO, SD_MOSI, SD_CS);
    if (!SD.begin(SD_CS, sdSPI, 20000000)) {
        Serial.println("[SD] Mount fail");
        return false;
    }
    Serial.printf("[SD] OK, %llu MB\n", SD.cardSize() / (1024ULL * 1024ULL));
    return true;
}

void loadProfile() {
    if (!SD.exists("/profile.json")) {
        Serial.println("[Profile] Khong co, dung mac dinh");
        return;
    }
    File f = SD.open("/profile.json", FILE_READ);
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
    File f = SD.open("/history.log", FILE_APPEND);
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

    WiFiClientSecure client;
    client.setInsecure();
    client.setTimeout(30000);

    if (!client.connect("api.groq.com", 443)) {
        Serial.println("[GROQ] Connect fail");
        return "";
    }

    const char* langCode = LANGS[currentLang].whisperCode;
    // Whisper không hỗ trợ "yue" trực tiếp → dùng "zh" cho Quảng Đông
    if (strcmp(langCode, "yue") == 0) langCode = "zh";

    String boundary = "----ESP32Boundary7MA4YWxkTrZu0gW";
    String head =
        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        "whisper-large-v3\r\n"
        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"language\"\r\n\r\n" +
        String(langCode) + "\r\n"
        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n"
        "json\r\n"
        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"voice.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n";
    String wavHeader = buildWavHeader(recordedBytes);
    String tail = "\r\n--" + boundary + "--\r\n";

    size_t contentLength = head.length() + wavHeader.length() + recordedBytes + tail.length();

    client.printf("POST /openai/v1/audio/transcriptions HTTP/1.1\r\n");
    client.printf("Host: api.groq.com\r\n");
    client.printf("Authorization: Bearer %s\r\n", GROQ_API_KEY);
    client.printf("Content-Type: multipart/form-data; boundary=%s\r\n", boundary.c_str());
    client.printf("Content-Length: %u\r\n", (unsigned)contentLength);
    client.printf("Connection: close\r\n\r\n");

    client.print(head);
    client.print(wavHeader);
    const size_t CHUNK = 2048;
    for (size_t i = 0; i < recordedBytes; i += CHUNK) {
        size_t n = min(CHUNK, recordedBytes - i);
        client.write(recordBuffer + i, n);
    }
    client.print(tail);

    String line;
    while (client.connected()) {
        line = client.readStringUntil('\n');
        if (line == "\r" || line.length() == 0) break;
    }
    String body;
    while (client.available()) body += (char)client.read();
    client.stop();

    int braceStart = body.indexOf('{');
    if (braceStart < 0) return "";
    body = body.substring(braceStart);
    int braceEnd = body.lastIndexOf('}');
    if (braceEnd > 0) body = body.substring(0, braceEnd + 1);

    JsonDocument doc;
    if (deserializeJson(doc, body)) return "";
    if (!doc["text"].is<const char*>()) return "";
    return doc["text"].as<String>();
}

// ========================= SEGMENTED CHAT / TTS =========================
String buildSegmentedSystemPrompt() {
    return String(LANGS[currentLang].systemPrompt) +
        "\n\nIMPORTANT OUTPUT FORMAT:\n"
        "Return ONLY one JSON object. No markdown, no extra text.\n"
        "The object must be {\"segments\":[{\"lang\":\"vi|en|zh|yue\",\"text\":\"...\"}]}.\n"
        "Use the user's language for explanations, but put every pronunciation sample "
        "in its own item with the real language of that sample.\n"
        "Examples:\n"
        "{\"segments\":[{\"lang\":\"vi\",\"text\":\"Cau tieng Anh la:\"},"
        "{\"lang\":\"en\",\"text\":\"I want to drink water.\"},"
        "{\"lang\":\"vi\",\"text\":\"Nghia la: Toi muon uong nuoc.\"}]}\n"
        "If the user asks in Vietnamese for English or Chinese, explain in Vietnamese "
        "and tag English samples as en, Mandarin samples as zh, Cantonese samples as yue.\n"
        "If the user asks in Cantonese for English or Mandarin, explain in Cantonese "
        "and tag English samples as en, Mandarin samples as zh.\n";
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
        if (speakViaElevenLabs(text, lang)) return true;
    }
    return false;
}

// ========================= DEEPSEEK =========================
String askDeepSeek(const String& userMessage) {
    WiFiClientSecure client;
    client.setInsecure();
    client.setTimeout(45000);
    HTTPClient https;

    if (!https.begin(client, DEEPSEEK_URL)) return "";
    https.addHeader("Content-Type", "application/json");
    https.addHeader("Authorization", String("Bearer ") + DEEPSEEK_API_KEY);
    https.setTimeout(45000);

    JsonDocument req;
    req["model"]       = "deepseek-chat";
    req["temperature"] = 0.7;
    req["max_tokens"]  = 700;
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

    int code = https.POST(payload);
    String reply;
    if (code == HTTP_CODE_OK) {
        String resp = https.getString();
        JsonDocument doc;
        if (!deserializeJson(doc, resp)) {
            reply = doc["choices"][0]["message"]["content"].as<String>();
        }
    } else {
        Serial.printf("[DeepSeek] HTTP %d\n", code);
    }
    https.end();
    return reply;
}

// ========================= ELEVENLABS TTS =========================
// POST /v1/text-to-speech/{voice_id} → trả MP3 binary
bool speakViaElevenLabs(const String& text, Language voiceLang) {
    if (!SD.begin(SD_CS, sdSPI, 20000000)) {
        Serial.println("[TTS] SD chua san sang");
    }

    WiFiClientSecure client;
    client.setInsecure();
    client.setTimeout(45000);
    HTTPClient https;

    String url = String("https://") + ELEVENLABS_HOST +
                 "/v1/text-to-speech/" + LANGS[voiceLang].voiceId;

    if (!https.begin(client, url)) return false;
    https.addHeader("xi-api-key", ELEVENLABS_API_KEY);
    https.addHeader("Content-Type", "application/json");
    https.addHeader("Accept", "audio/mpeg");
    https.setTimeout(45000);

    JsonDocument req;
    req["text"]       = text;
    req["model_id"]   = ELEVENLABS_MODEL;
    JsonObject vs = req["voice_settings"].to<JsonObject>();
    vs["stability"]        = 0.5;
    vs["similarity_boost"] = 0.75;
    vs["style"]            = 0.0;
    vs["use_speaker_boost"]= true;

    String payload;
    serializeJson(req, payload);

    int code = https.POST(payload);
    if (code != HTTP_CODE_OK) {
        Serial.printf("[TTS] HTTP %d: %s\n", code, https.errorToString(code).c_str());
        https.end();
        return false;
    }

    // Lưu MP3 ra SD rồi phát qua ESP32-audioI2S
    if (SD.exists("/tts.mp3")) SD.remove("/tts.mp3");
    File mp3 = SD.open("/tts.mp3", FILE_WRITE);
    if (!mp3) {
        Serial.println("[TTS] Mo file fail");
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
    Serial.printf("[TTS] Saved %d bytes\n", written);
    if (written < 100) return false;

    audio.connecttoFS(SD, "/tts.mp3");
    return true;
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
    delay(200);
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

    tft.init();
    tft.setRotation(0);
    tft.fillScreen(TFT_BLACK);
    pinMode(TFT_BL, OUTPUT);
    digitalWrite(TFT_BL, HIGH);

    rgb.begin();
    rgb.setBrightness(50);
    rgb.show();

    setState(STATE_BOOT);
    delay(500);

    if (!initSDCard()) {
        Serial.println("[WARN] Khong co SD");
    } else {
        loadProfile();
    }

    if (!initMicI2S()) {
        Serial.println("[FATAL] Mic init fail");
        setState(STATE_ERROR);
        while (true) delay(1000);
    }

    audio.setPinout(I2S_SPK_BCLK, I2S_SPK_LRC, I2S_SPK_DIN);
    audio.setVolume(18);

    setState(STATE_WIFI_CONNECTING);
    WiFiManager wm;
    wm.setConfigPortalTimeout(180);
    if (!wm.autoConnect("ESP32-Voice-Setup")) {
        Serial.println("[WiFi] Timeout");
        setState(STATE_ERROR);
        ESP.restart();
    }
    Serial.printf("[WiFi] OK %s\n", WiFi.localIP().toString().c_str());

    setState(STATE_IDLE);
}

// ========================= LOOP =========================
void loop() {
    if (currentState == STATE_SPEAKING) {
        audio.loop();
        if (!audio.isRunning()) {
            if (startNextTtsSegment()) {
                return;
            }
            saveConversation(lastUserText, lastAssistantText);
            resetTtsSegments();
            setState(STATE_IDLE);
        }
        return;
    }

    if (currentState == STATE_IDLE &&
        (digitalRead(BUTTON_PIN) == LOW || waitForSpeechStart())) {
        if (digitalRead(BUTTON_PIN) == LOW) {
            delay(40);
            if (digitalRead(BUTTON_PIN) != LOW) return;
        }

        setState(STATE_LISTENING);
        size_t n = recordAudio();
        if (n < SAMPLE_RATE) {
            updateDisplay("TOO SHORT", "Noi lau hon mot chut.", TFT_ORANGE);
            delay(1500);
            setState(STATE_IDLE);
            return;
        }

        setState(STATE_TRANSCRIBING);
        String transcript = sendToGroqWhisper();
        transcript.trim();
        if (transcript.isEmpty()) {
            updateDisplay("NO INPUT", "Thu lai.", TFT_ORANGE);
            delay(1500);
            setState(STATE_IDLE);
            return;
        }
        Serial.printf("[STT] %s\n", transcript.c_str());

        if (!extractWakeCommand(transcript)) {
            Serial.printf("[WAKE] Ignored, missing phrase: %s\n", WAKE_PHRASE);
            setState(STATE_IDLE);
            return;
        }

        if (transcript.isEmpty()) {
            updateDisplay("WAKE OK", "Ban muon hoi gi?", TFT_CYAN);
            delay(1200);
            setState(STATE_IDLE);
            return;
        }

        lastUserText = transcript;
        Serial.printf("[WAKE] Command: %s\n", transcript.c_str());

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
        reply.trim();
        if (reply.isEmpty()) reply = "Xin loi, chua tra loi duoc.";
        prepareTtsSegments(reply, currentLang);
        lastAssistantText = flattenTtsSegments();
        if (lastAssistantText.isEmpty()) lastAssistantText = reply;
        Serial.printf("[LLM raw] %s\n", reply.c_str());
        Serial.printf("[LLM text] %s\n", lastAssistantText.c_str());

        if (startNextTtsSegment()) {
            setState(STATE_SPEAKING);
        } else {
            updateDisplay("TTS FAIL", lastAssistantText, TFT_ORANGE);
            delay(2500);
            saveConversation(lastUserText, lastAssistantText);
            setState(STATE_IDLE);
        }
    }

    delay(10);
}

// ========================= CALLBACKS AUDIO LIB =========================
void audio_info(const char* info)    { Serial.printf("[AUDIO] %s\n", info); }
void audio_eof_mp3(const char* info) { Serial.printf("[AUDIO] EOF: %s\n", info); }
