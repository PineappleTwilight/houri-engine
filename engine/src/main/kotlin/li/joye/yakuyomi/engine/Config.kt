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
    // seg 文字筆畫遮罩二值門檻（去字用）。★ 0.3 會濾掉漢字旁注音「假名」的弱訊號 → 去字留一排假名殘留。
    // 降到 0.12＝偵測器其實看得到假名、只是 prob 弱（桌面 parity/auto_diag.py dev_furi3 實證）。只影響去字遮罩、不動偵測框/OCR。
    val segThreshold: Float = 0.12f,
    // QNN(Hexagon NPU) EP：true＝偵測走 NPU；fp32 模型自動以 fp16 推論（enable_htp_fp16_precision 預設）。
    // 需 onnxruntime-android-qnn AAR。失敗（無庫/不支援）自動退回 XNNPACK→CPU。實際用的 EP 見 [Detector.ep]。
    val useQnn: Boolean = false,
)

data class OcrConfig(
    val textHeight: Int = 48,         // 48px CTC
    val minTextLength: Int = 0,       // config.ocr.min_text_length
    val ignoreBubble: Int = 0,        // 〔設定〕config.ocr.ignore_bubble：1–50 開啟，跳過彩色/非氣泡 SFX 類文字（預設 0＝關）
    val minProb: Float = 0.5f,        // config.ocr.prob：OCR 平均信心 < 此值就丟（剃除低信心誤讀；m-i-t 預設 0.5）
    val useXnnpack: Boolean = false,  // ★預設關：XNNPACK 會把 48px CTC OCR 模型算錯（真機實證吐空），改純 CPU 才正確
    // 逐行並發 OCR：小圖塊（48px 高、窄）吃不滿 intra-op 4 緒 → 改「每行單緒、N 行並發」把核填滿。
    // concurrent=true → session intra-op 設 1（單行單緒）、靠 Semaphore(concurrency) 並發；false → 單行用滿 NUM_THREADS、序列（現狀）。
    // 純 CPU、ORT 共享 thread pool ⇒ 收益需實測（sandbox 去背比較 OCR 列 A/B）。與「批次 padding」不同：零 padding 浪費。
    val concurrent: Boolean = false,
    val concurrency: Int = 6,         // 同時在飛的行數上限（8 核留 2 核給系統）
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
    // 預設 auto：每區判背景——白且均勻＝對話框→平塗背景白（瞬間、保證無殘留）；否則(臉/髮/壓畫面)→lama逐區重建。
    // 對齊桌面 parity/auto_diag.py（含假名修復、bubble 路由、平塗）。可改 boxfill(全平塗就近取色) / lama(整頁或逐區)。
    val method: String = "auto",
    val wholeImage: Boolean = true,   // lama：true＝整頁一次（快）/ false＝逐區（小泡銳利、慢）
    val tileSize: Int = 512,          // Koharu lama-manga.onnx 固定 512（改了對不上模型）
    val windowRatio: Float = 1.7f,    // Koharu BALLOON_WINDOW_RATIO（lama 逐區裁窗）
    // 去字遮罩膨脹（半徑 = maskDilate/2）。★關鍵：漫畫在臉/頭髮上的字會描一圈白邊；遮罩太薄只蓋黑筆畫、
    // 白邊留在外面 → boxfill 取到白邊抹成白塊、lama 把白邊當 context 延伸成白塊（兩者都像沒去字）。
    // 加厚到半徑~12 吞掉白邊後，lama 周圍 context 全變底圖 → 重建乾淨（桌面 inpaint_dev DIL=12 實證 ≈ MIT）。
    val maskDilate: Float = 24f,      // 半徑 12px：吞掉文字白邊（之前 7=半徑4 會殘白塊）
    // auto 路由：背景亮度 std < autoStdThreshold 且均值 ≥ autoWhiteThreshold ＝對話框→平塗；否則 lama。
    // ★ std 在「未膨脹 textMask 的行框四邊形多邊形內」量（斜框修正：軸對齊 bbox 會把傾斜泡泡的角落雜訊算進來→誤判）。
    // 實測（quad 量測、桌面 auto_diag.py 01.jpg 驗）：真白泡 std 2-3、壓在亮建築/牆面上的字 std 9-21、臉/髮 24+。
    // 真泡泡(2-3)與壓畫面(9+)中間有大空檔 → 門檻落在裡面（讓真泡泡留餘裕只走快速平塗）。
    // ★ 引擎實機量測比桌面 auto_diag 系統性偏低（窄框 bg 像素少、std 估計差）：桌面量 9.6 的窄框，引擎量 <8 → 8 仍漏判 boxfill。
    //   故引擎側用 6（仍遠高於真泡泡 2-3）。sandbox 去背比較已把引擎實測 std 標在每個框上＝日後調此值直接看引擎真值。
    val autoStdThreshold: Float = 6f,    // 12→8→6：壓畫面窄框引擎量 ~7 漏判 boxfill；6 給足餘裕（桌面 auto_diag 01.jpg + 真機去背比較驗證）
    val autoWhiteThreshold: Float = 190f, // 背景亮度均值門檻：對話框是白底
    val bboxPad: Int = 16,                // 去字 allow 用區域 bbox 矩形外擴 px：涵蓋貼 bbox 邊界的假名（行框太緊會漏）
    val intraThreads: Int = 4,        // LaMa session intra-op 執行緒（整頁/逐區都用滿 4 核）
    // QNN(Hexagon NPU) EP：true＝lama 去字走 NPU；fp32 自動 fp16。需 -qnn AAR。失敗退回 XNNPACK→CPU。實際 EP 見 [Inpainter.ep]。
    val useQnn: Boolean = false,
)

data class RenderConfig(
    val orientation: TextOrientation = TextOrientation.AUTO, // 〔設定〕對應 config.render.direction=auto（CJK→直排）
    val fontBorder: Boolean = true,                              // 〔設定〕config.render.disable_font_border=false
    val artStrokeRatio: Float = 0.16f,                           // 壓畫面區(lama/auto忙碌)的白邊寬＝字級×此（比一般 0.10 粗；busy 背景上黑字粗白邊更好讀）
    val fontSizeMax: Int = 60,
    val fontSizeMin: Int = 9,
    // 排版幾何（純文字框法，對齊 parity/typeset_parity.py；不常動，留可控空間）
    val expandW: Float = 1.3f,   // 文字框放大倍率（寬）給呼吸空間
    val expandH: Float = 1.5f,   // 文字框放大倍率（直欄高 / 橫排列高）
    val colTrim: Int = 3,        // 直排每欄少放幾字（縮短欄長、減少凸出；欄變多→字級自動縮）
    val rowTrim: Int = 3,        // 橫排每行少放幾字（colTrim 的橫排對映：行變短、列變多→字級自動縮）
    val fontScale: Float = 0.85f, // 算好字級後整體縮放（<1＝更小、更 fit 格子、留邊距）
    // 文字顏色：auto＝取去字後背景亮度判黑/白字（最穩、白底黑字/黑底白字）；mono＝一律黑字白邊
    val colorMode: String = "auto",
    val bgDark: Int = 110,       // auto：去字後背景平均亮度 < 此值＝暗底 → 白字
)
