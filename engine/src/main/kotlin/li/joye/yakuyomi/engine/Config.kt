package li.joye.yakuyomi.engine

/**
 * 引擎參數（CLAUDE.md §5 Config：第一層 schema + 預設值）。
 *
 * 參考使用者的 m-i-t config_deepseek.json，但**預設值對齊本專案實際使用的模型**：
 *   偵測 = comic-text-detector（非 m-i-t 的 default 偵測器）
 *   OCR  = 48px CTC（非 48px 自回歸）
 *   去字 = Koharu lama-manga.onnx（固定 512；非 lama_large）
 * 因此部分數值與該 config 的 default/48px/lama_large 不同，差異處以註解標出。
 *
 * 設定粒度：§11 v1 全域。標〔設定〕者＝預期在設定頁開放調整（頻繁調，如直/橫排、provider/key/語言）；
 * 未標者＝預設，留可控空間（仍在此結構內），需要時可快速加進設定 UI。
 */
data class EngineConfig(
    val detector: DetectorConfig = DetectorConfig(),
    val ocr: OcrConfig = OcrConfig(),
    val translator: TranslatorConfig = TranslatorConfig(),
    val inpainter: InpainterConfig = InpainterConfig(),
    val render: RenderConfig = RenderConfig(),
)

data class DetectorConfig(
    val inputSize: Int = 1024,        // comic-text-detector 標準（config.detection_size=1536 是 default 偵測器）
    val textThreshold: Float = 0.3f,  // SegDetectorRepresenter thresh
    val boxThreshold: Float = 0.6f,   // 〔設定〕ctd.py 外部過濾（config.box_threshold=0.7 是 default 偵測器）
    val unclipRatio: Float = 1.5f,    // 〔設定〕ctd unclip（config.unclip_ratio=2.3 是 default 偵測器）
    val minSide: Float = 3f,
)

data class OcrConfig(
    val textHeight: Int = 48,         // 48px CTC
    val minTextLength: Int = 0,       // config.ocr.min_text_length
)

data class TranslatorConfig(
    val provider: String = "deepseek",                                  // 〔設定〕config.translator
    val targetLang: String = "CHT",                                     // 〔設定〕config.target_lang
    val model: String = "deepseek-chat",                                // 〔設定〕
    val apiBase: String = "https://api.deepseek.com/chat/completions",  // 〔設定〕custom_openai 用
    val toLangName: String = "Traditional Chinese (Taiwan, 台灣慣用的繁體中文用語)",
    val temperature: Double = 0.3,
    val concurrency: Int = 4,         // 〔設定〕跨頁並發上限（§10 Semaphore 旋鈕；Pipeline 用）
    val filterText: String? = null,   // 〔設定〕config.filter_text：regex 命中譯文則濾掉該區（例 ".*badtext.*"）
)

data class InpainterConfig(
    val tileSize: Int = 512,          // Koharu lama-manga.onnx 固定 512（改了會對不上模型）
    val windowRatio: Float = 1.7f,    // Koharu BALLOON_WINDOW_RATIO
    val maskDilate: Float = 7f,       // ~ config.kernel_size / mask_dilation_offset
)

data class RenderConfig(
    val orientation: TextOrientation = TextOrientation.AUTO, // 〔設定〕對應 config.render.direction=auto（CJK→直排）
    val fontBorder: Boolean = true,                              // 〔設定〕config.render.disable_font_border=false
    val fontSizeMax: Int = 60,
    val fontSizeMin: Int = 9,
    // 排版幾何（純文字框法，對齊 parity/typeset_parity.py；不常動，留可控空間）
    val expandW: Float = 1.3f,   // 文字框放大倍率（寬）給呼吸空間
    val expandH: Float = 1.5f,   // 文字框放大倍率（直欄高 / 橫排列高）
    val colTrim: Int = 2,        // 直排每欄少放幾字（縮短欄長、減少凸出；欄變多→字級自動縮）
    // 文字顏色：auto＝取去字後背景亮度判黑/白字（最穩、白底黑字/黑底白字）；mono＝一律黑字白邊
    val colorMode: String = "auto",
    val bgDark: Int = 110,       // auto：去字後背景平均亮度 < 此值＝暗底 → 白字
)
