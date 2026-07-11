package li.joye.yakuyomi.engine

/**
 * 引擎參數（CLAUDE.md §5 Config：第一層 schema + 預設值）。
 *
 * 參考使用者的 m-i-t config_deepseek.json，但**預設值對齊本專案實際使用的模型**：
 *   偵測 = comic-text-detector（非 m-i-t 的 default 偵測器）
 *   OCR  = 48px CTC（非 48px 自回歸）
 *   去字 = AOT-GAN（m-i-t inpainting.ckpt，NCNN·整頁 768；LaMa 已退役）
 * 因此部分數值與該 config 的 default/48px 不同，差異處以註解標出。
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
    val intraThreads: Int = 6,        // XNNPACK intra-op 緒。同 lama：6 大核甜蜜點（4→6 預期省 ~0.4s；探測確認）
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
    val concurrent: Boolean = true,   // 預設開：真機 8.9s→4.8s(快46%)、零品質風險(每行邏輯不變)、8 核全填滿
    val concurrency: Int = 8,         // 同時在飛的行數上限（＝核數，全核並發；conc=核數為甜蜜點，再高無核可用）
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
    // ⚠️ toLangName / fromLangName 預設被 fork :domain 的 TranslationPreferences.DEFAULT_TARGET_LANG /
    //   DEFAULT_SOURCE_LANG 鏡像（:domain 不能 import 引擎）。改這兩個請同步改那邊，否則 few-shot 判斷會 drift。
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
    // 〔設定〕去字方法（真機 A/B 拍板＝兩門別；LaMa/逐格/auto 逐區路由皆已退役）：
    //   boxfill＝「快速去字」：取字區就近的背景色平塗（瞬間、平/單色泡泡最乾淨；但壓畫面/多彩會塗錯色塊＝粗糙）。不跑去字模型。
    //   aot（預設）＝「AI 去字」：全區都跑 AOT-GAN 重建背景（含乾淨泡泡、無平塗路由），整頁一次（tileSize，見下）。需 [ModelSet.aotInpainterNcnn]（NCNN）或 [aotInpainter]（ORT 備援）。
    //     走 NCNN AOT（整頁固定 tile、Vulkan-capable=GPU/NPU-ready）；去字被翻譯的網路等待蓋住(§8)，故 tile 大小幾乎不加牆鐘。
    //   （auto/auto_aot＝舊的逐區路由「乾淨泡平塗、忙碌區才跑模型」＝已移除：AI 去字會有些地方是 boxfill＝不一致，Inpainter 一律把 auto* 當 aot。）
    //   （lama＝LaMa 去字，已退役：留 method 相容但需另外提供 lama 模型，否則 Yakuyomi.create loud-error。）
    // ※ 不開 concurrency／獨立 session：去字是純 CPU、核數固定，平行切核不增總算力（實測並發≈序列）。
    val method: String = "aot",
    val wholeImage: Boolean = true,   // lama/aot：true＝整頁一次（快）/ false＝逐區（小泡銳利、慢）
    val tileSize: Int = 768,          // 整頁 AOT 去字解析度。AOT 全卷積·任意尺寸；768＝畫質/記憶體/藏在翻譯下的甜蜜點（真機 A/B 拍板：512 忙碌區糊、1024 記憶體 2× 且貼翻譯天花板）。LaMa(退役)才鎖 512。
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
    // LaMa session intra-op 執行緒。真機探測(SD 8 Gen 3, 8 核=6 大+2 小)：4→7.0s、6→5.8s(最快)、8→6.3s。
    // 6＝用滿 6 大核（1×X4+5×A720）、避開 2 個慢的 A520 小核（加進去反拖累）。4→6 去字快 ~17%、整頁/逐區都受惠。
    val intraThreads: Int = 6,        // 4→6（big.LITTLE 甜蜜點＝大核數；非全 8 核，小核拖累）
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
    // 縱中橫（tate-chu-yoko）：直排時把連續短 ASCII 串（2–4 字的數字/字母/!?）併成一格水平並排（年齡「20」、年份「2020」、「!?」不再上下堆疊歪頭讀）。
    // §4 第三層知情偏離：m-i-t/parity 逐字畫、無此邏輯；只影響直排內 ASCII 短串，CJK 不變。預設開、可關回逐字。
    val tateChuYoko: Boolean = true,
)
