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
    val segThreshold: Float = 0.3f,   // seg 文字筆畫遮罩二值門檻（去字用；parity/seg_validate.py 驗證 0.3 對齊、覆蓋好）
)

data class OcrConfig(
    val textHeight: Int = 48,         // 48px CTC
    val minTextLength: Int = 0,       // config.ocr.min_text_length
    val ignoreBubble: Int = 0,        // 〔設定〕config.ocr.ignore_bubble：1–50 開啟，跳過彩色/非氣泡 SFX 類文字（預設 0＝關）
    val minProb: Float = 0.5f,        // config.ocr.prob：OCR 平均信心 < 此值就丟（剃除低信心誤讀；m-i-t 預設 0.5）
    val useXnnpack: Boolean = false,  // ★預設關：XNNPACK 會把 48px CTC OCR 模型算錯（真機實證吐空），改純 CPU 才正確
)

// 預設 few-shot（日→繁中）：示範 <|i|> 逐行格式。改語言對時連同 toLangName/fromLangName 一起換成對應譯文。
private const val DEFAULT_SAMPLE_SOURCE =
    "<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n<|2|>きみ… 大丈夫⁉\n<|3|>なんだこいつ 空気読めて ないのか…？"
private const val DEFAULT_SAMPLE_TARGET =
    "<|1|>好尷尬…我不想引人注目…我想消失…\n<|2|>你…沒事吧⁉\n<|3|>這傢伙是看不懂氣氛嗎…？"

/**
 * 翻譯設定。**語言對可任意**（不寫死日→繁中，只是預設）：
 *  - [toLangName] = 目標語言（LLM 直接照這個翻）。
 *  - [fromLangName] = 來源語言標註（進 prompt 措辭；實際來源其實由 OCR 模型決定＝BYOM 換模型就換來源）。
 *  - [sampleSource]/[sampleTarget] = few-shot 範例（同時示範格式與語言對）。
 * 換語言對：設 toLangName + fromLangName + 對應的 few-shot（三者要一致，否則 few-shot 會把輸出帶偏）。
 */
data class TranslatorConfig(
    val provider: String = "deepseek",                                  // 〔設定〕config.translator
    val targetLang: String = "CHT",                                     // 〔設定〕config.target_lang
    val model: String = "deepseek-chat",                                // 〔設定〕
    val apiBase: String = "https://api.deepseek.com/chat/completions",  // 〔設定〕custom_openai 用
    val toLangName: String = "Traditional Chinese (Taiwan, 台灣慣用的繁體中文用語)",  // 〔設定〕目標語言
    val fromLangName: String = "Japanese",          // 〔設定〕來源語言標註（空白＝讓 LLM 自己判）
    val sampleSource: String = DEFAULT_SAMPLE_SOURCE, // 〔設定〕few-shot 原文（空白＝不放範例）
    val sampleTarget: String = DEFAULT_SAMPLE_TARGET, // 〔設定〕few-shot 譯文（要跟 toLangName 同語言）
    val temperature: Double = 0.3,
    // 跨頁批次翻譯（對映 m-i-t --batch-size / --batch-concurrent；§2 翻譯批次策略、§10 並發旋鈕）
    val batchSize: Int = 8,              // 〔設定〕批次頁數：concurrent 模式＝同時並發的頁數上限；merged 模式＝每 prompt 併幾頁
    val batchConcurrent: Boolean = true, // 〔設定〕true＝逐頁分開請求、批內並發（防 truncation/幻覺，推薦）；false＝併大 prompt（有風險）
    val filterText: String? = null,   // 〔設定〕config.filter_text：regex 命中譯文則濾掉該區（例 ".*badtext.*"）
)

data class InpainterConfig(
    // 〔設定〕去字方法（真機實測拍板預設＝lama 整頁：見下）：
    //   lama+wholeImage＝整頁縮 512 跑一次 LaMa（~6s 去字、~18s/頁）。【預設】從不醜爆（最多輕微暈開），跨整個庫最安全。
    //   boxfill        ＝取氣泡底色填字區（瞬間、~12s/頁、平/單色泡泡最乾淨）。但多彩/壓在畫面上的字會塗錯色塊＝失敗得很醜，故不當預設。
    //   lama+逐區       ＝每區各跑一次 LaMa（小泡較銳利，但 N 區＝N× 整頁算力 ⇒ ~81s/頁；大/彩色泡泡仍會糊）。
    // ※ 不開 concurrency／獨立 session：去字是純 CPU、核數固定，平行切核不增總算力（實測並發≈序列）。逐區慢是「做 N 倍的事」，平行救不了。
    val method: String = "boxfill",   // 真機 A/B 拍板：boxfill+seg 最快(~11.5s/頁)又乾淨(細筆畫+就近取色)、品質追平最貴的逐格
    val wholeImage: Boolean = true,   // lama：true＝整頁一次（快）/ false＝逐區（小泡銳利、慢）
    val tileSize: Int = 512,          // Koharu lama-manga.onnx 固定 512（改了對不上模型）
    val windowRatio: Float = 1.7f,    // Koharu BALLOON_WINDOW_RATIO（lama 逐區裁窗）
    val maskDilate: Float = 7f,       // ~ config.kernel_size / mask_dilation_offset
    val intraThreads: Int = 4,        // LaMa session intra-op 執行緒（整頁/逐區都用滿 4 核）
)

data class RenderConfig(
    val orientation: TextOrientation = TextOrientation.AUTO, // 〔設定〕對應 config.render.direction=auto（CJK→直排）
    val fontBorder: Boolean = true,                              // 〔設定〕config.render.disable_font_border=false
    val fontSizeMax: Int = 60,
    val fontSizeMin: Int = 9,
    // 排版幾何（純文字框法，對齊 parity/typeset_parity.py；不常動，留可控空間）
    val expandW: Float = 1.3f,   // 文字框放大倍率（寬）給呼吸空間
    val expandH: Float = 1.5f,   // 文字框放大倍率（直欄高 / 橫排列高）
    val colTrim: Int = 3,        // 直排每欄少放幾字（縮短欄長、減少凸出；欄變多→字級自動縮）
    val fontScale: Float = 0.85f, // 算好字級後整體縮放（<1＝更小、更 fit 格子、留邊距）
    // 文字顏色：auto＝取去字後背景亮度判黑/白字（最穩、白底黑字/黑底白字）；mono＝一律黑字白邊
    val colorMode: String = "auto",
    val bgDark: Int = 110,       // auto：去字後背景平均亮度 < 此值＝暗底 → 白字
)
