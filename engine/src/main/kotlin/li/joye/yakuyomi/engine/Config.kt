package li.joye.yakuyomi.engine

/**
 * Engine parameters (CLAUDE.md §5 Config: first-layer schema + defaults).
 *
 * Referenced from user's m-i-t config_deepseek.json, but **defaults are aligned to the models actually used**:
 *   detection = DBNet (m-i-t default detector, ResNet34+DB head)
 *   OCR = 48px CTC (not 48px autoregressive)
 *   inpainting = AOT-GAN (m-i-t inpainting.ckpt, NCNN whole-page 768; LaMa retired)
 * Some values therefore differ from that config's default/48px, differences annotated.
 *
 * Granularity: §11 v1 global. Marked `[Settings]` = expected to be exposed in settings (frequently tuned, e.g., vertical/horizontal, provider/key/language);
 * unmarked = default, keep tunable headroom (still inside this structure), can be added to settings UI quickly when needed.
 */
data class EngineConfig(
    val detector: DetectorConfig = DetectorConfig(),
    val ocr: OcrConfig = OcrConfig(),
    val translator: TranslatorConfig = TranslatorConfig(),
    val inpainter: InpainterConfig = InpainterConfig(),
    val render: RenderConfig = RenderConfig(),
)

data class DetectorConfig(
    val minSide: Float = 3f,
    // seg stroke mask binarization threshold (for inpainting). 0.3 filters weak furigana signals next to kanji → leaves a row of furigana after inpainting.
    // Lowered to 0.12 = detector actually sees furigana but prob is weak (desktop parity/auto_diag.py dev_furi3). Only affects inpaint mask, not detection box/OCR.
    val segThreshold: Float = 0.12f,
    // DBNet (m-i-t default detector, sole detector in this project): ResNet34+DB head, 1.6-2.5x more correct reads than retired ctd (device-proven)
    // out0=db (2ch, ch0=raw logits, Kotlin adds sigmoid), out1=mask (1ch, half/full-res depends on platform, already sigmoid). DB post-processing see Detector.linesFromProbMap.
    val dbnetInputSize: Int = 1024,       // DBNet sweet spot (proven on device 3 pages x size x OCR: @1024 highest read rate + warm ~0.9s; @960 rough, @1280+ slow and misreads, @768 misses. resize_aspect -> input canvas 768x1024, rectangle avoids square 832-992 crash zone)
    val detectUnsharp: Boolean = false,   // Optional: detection input sharpening (marginal + OOD; default off per demo06 A/B)
    val dbBinThreshold: Float = 0.5f,     // DB binarize: sigmoid(db ch0) > this (m-i-t text_threshold=0.5)
    val dbBoxThreshold: Float = 0.7f,     // DB score filter: component-mean prob < this dropped (m-i-t box_threshold=0.7)
    val dbUnclipRatio: Float = 2.3f,      // DB unclip expansion (m-i-t unclip_ratio=2.3)
)

data class OcrConfig(
    val textHeight: Int = 48,         // 48px CTC
    val minTextLength: Int = 0,       // config.ocr.min_text_length
    val ignoreBubble: Int = 0,        // [Settings] config.ocr.ignore_bubble: 1-50 enabled, skip colored/non-bubble SFX text (default 0=off)
    val minProb: Float = 0.5f,        // config.ocr.prob: drop OCR avg confidence < this (filter low-confidence misreads; m-i-t default 0.5)
    // Expand each side of detection quad by N px before OCR crop (RotRect.expand; only OCR crop, not detection box => inpaint mask via seg strokes unaffected).
    // Root cause: thin detection boxes clip glyphs -> 48px CTC empty read (model_48px_ctc drops 0-char boxes before prob threshold) -> region filtered by Pipeline textRegions -> left untranslated = user sees "missing bubble". Desktop 16 pages: pad=4 reads 345->398 (+15%), box count unchanged,
    // break 9 vs save 350; 006 "sono toori ja" box only 23px wide "tsu" clipped -> pad=0 empty, pad=12 correct p=0.993.
    // Default 4 = device A/B proven (sandbox 6 pages 161 boxes, matched by detection box index): save 2 (including 006), break 0, effectively fix ~14 (e.g., 'to ichi mo...'), cost = minor noise ('!'->'—', one kana, LLM tolerant; same tradeoff as useBicubic). Also ~20% faster OCR (wider box -> shorter CTC sequence).
    // pad=8/12 regress (8: break 2; 12: break 1) => 4 is sweet spot. Desktop m-i-t warp sim gave +15% reads, device only +2 (engine bicubic warp baseline already 98%), so device value cannot be copied from desktop.
    val stripPad: Int = 4,
    val useXnnpack: Boolean = false,  // Default off: XNNPACK miscomputes 48px CTC model (device proven empty), pure CPU is correct
    // Per-line concurrent OCR: small tiles (48px high, narrow) under-utilize intra-op 4 threads -> use "1 thread per line, N lines concurrent" to fill cores.
    // concurrent=true -> session intra-op =1 (one line one thread), via Semaphore(concurrency); false -> one line uses NUM_THREADS, sequential.
    // Pure CPU, ORT shared thread pool => gains need measurement (sandbox A/B). Different from "batch padding": zero padding is wasteful.
    val concurrent: Boolean = true,   // Default on: device 8.9s->4.8s (46% faster), zero quality risk (per-line logic unchanged), 8 cores saturated
    val concurrency: Int = 8,         // Max concurrent lines in flight (=core count, sweet spot; higher gives no gain)
    // Crop scaling interpolation: true=hand-rolled perspective bicubic (saves small kana misreads -> sentence-ending negation no longer inverted), false=Canvas bilinear (old).
    // parity 517 chars/100 lines vs bilinear 486/96; device A/B proven: +6% OCR time, restores tails like "才能がある/言わないから/言いなさい" (eliminates most dangerous "meaning inversion"), occasional extra noise char (LLM tolerant). Net positive -> default on.
    val useBicubic: Boolean = true,
    // OCR strip sharpening (unsharp mask, see Ocr.unsharp): counteracts blur from warping ~30px wide vertical strip upscaled to 48px,
    // restores small kana that were blurred away (added v0.16.9). Default on = proven to restore small kana (no side effect on clean lines with p~1.0);
    // off = revert to no sharpening (strip blurrier, small kana may be missed). Was hard-coded always on, extracted to setting 2026-07-16 (advanced users can turn off).
    val ocrUnsharp: Boolean = true,
)

// Default few-shot (ja->cht): demonstrates <|i|> line format. When changing language pair, update toLangName/fromLangName together with corresponding translation.
private const val DEFAULT_SAMPLE_SOURCE =
    "<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n<|2|>きみ… 大丈夫⁉\n<|3|>なんだこいつ 空気読めて ないのか…？"
private const val DEFAULT_SAMPLE_TARGET =
    "<|1|>好尷尬…我不想引人注目…我想消失…\n<|2|>你…沒事吧⁉\n<|3|>這傢伙是看不懂氣氛嗎…？"

/**
 * Translation settings. **Language pair is arbitrary** (not hard-coded ja->cht, just default):
 *  - [toLangName] = target language (LLM translates directly into this).
 *  - [fromLangName] = source language label (goes into prompt wording; actual source is determined by OCR model = BYOM model determines source).
 *  - [sampleSource]/[sampleTarget] = few-shot example (demonstrates both format and language pair).
 * Change language pair: set toLangName + fromLangName + corresponding few-shot (all three must be consistent, otherwise few-shot will bias output).
 */
data class TranslatorConfig(
    val provider: String = "deepseek",                                  // 〔設定〕config.translator
    val targetLang: String = "CHT",                                     // [Settings] config.target_lang
    // Default matches LlmProviders deepseek entry. Old name deepseek-chat retired 2026-07-24 15:59 UTC (400) 
    // -> changed to corresponding deepseek-v4-flash; existing settings with old name migrated in place by LlmProviders.migrateModel.
    val model: String = "deepseek-v4-flash",                            // [Settings]
    val apiBase: String = "https://api.deepseek.com/chat/completions",  // [Settings] for custom_openai
    // Warning: toLangName / fromLangName defaults are mirrored from fork :domain TranslationPreferences.DEFAULT_TARGET_LANG /
    // DEFAULT_SOURCE_LANG (domain cannot import engine). If you change these, sync there, otherwise few-shot drift.
    val toLangName: String = "Traditional Chinese (Taiwan, Taiwan common Traditional Chinese)",  // [Settings] target language
    val fromLangName: String = "Japanese",          // [Settings] source language label (blank = let LLM decide)
    val sampleSource: String = DEFAULT_SAMPLE_SOURCE, // [Settings] few-shot source (blank = no example)
    val sampleTarget: String = DEFAULT_SAMPLE_TARGET, // [Settings] few-shot target (must match toLangName language)
    // [Settings] **Sampling temperature**. Not every provider/model consumes it: OpenAI reasoning models (o series, gpt-5) reject
    // temperature (400), DeepSeek in thinking mode accepts but ignores => LlmProviders.requestParams decides whether to send and clamps to provider's valid range (see ParamRule).
    val temperature: Double = 0.3,
    // [Settings] **Thinking mode (reasoning)**, default **off**.
    // Why off by default: newer models (DeepSeek v4, Gemini 3...) think by default — for structured "line-by-line translation" tasks extra seconds and tokens give no clear quality gain => off = replicates old deepseek-chat non-thinking behavior (fast and cheap).
    // On = allow thinking: **slower, more tokens, higher cost**.
    // Field shape per-provider (thinking / reasoning_effort / enable_thinking / reasoning), mapping see ParamRule;
    // Not every provider can turn off (OpenAI o series only to minimal, Gemini 3.x only minimal, Groq GPT-OSS cannot be turned off, self-hosted custom/sakura never send field).
    val thinking: Boolean = false,
    // Cross-page batch translation (maps to m-i-t --batch-size / --batch-concurrent; §2 batch strategy, §10 concurrency knob).
    // Current: engine has **no consumer** — original BatchTranslator reading these fields was removed (cross-page concurrency now handled by fork PageTranslator via Semaphore(pipelineDepth), not here). Kept = §4 first layer "config schema mirrors upstream" (upstream tuning => zero code change here); future engine batcher can re-attach directly. **Not a product settings item** (hence not marked [Settings]).
    val batchSize: Int = 8,              // m-i-t --batch-size: concurrent mode = max concurrent pages; merged mode = pages per prompt
    val batchConcurrent: Boolean = true, // m-i-t --batch-concurrent: true = separate requests per page, concurrent in batch (prevents truncation/hallucination); false = merged large prompt
    val filterText: String? = null,   // [Settings] config.filter_text: regex matching translation filters that region (e.g., ".*badtext.*")
)

data class InpainterConfig(
    // [Settings] Inpainting method (proven on device = two options, both pure NCNN; LaMa/per-cell/auto per-region routing/GPU all retired):
    //   boxfill="fast inpaint": flat fill with nearest background color near text area (instant, cleanest for flat/solid bubbles; rough on busy/multicolor where it paints wrong color blocks). No inpaint model run.
    //   aot (default)="AI inpaint": reconstruct background for whole page with AOT-GAN, whole page scaled to [tileSize]. Runs NCNN AOT; inpaint hidden under translation network wait (§8), so tile size barely adds wall time.
    val method: String = "aot",
    val tileSize: Int = 768,          // Whole-page AOT inpaint resolution. AOT fully convolutional, any size; 768 = quality/memory/hidden-under-translation sweet spot (proven on device: 512 busy areas blurry, 1024 memory 2x and hits translation ceiling).
    // Inpaint mask dilation (radius = maskDilate/2). Key: manga text on faces/hair has white outline around glyphs; mask too thin covers only black strokes,
    // white outline remains outside -> white blocks remain after inpaint. Thickened to radius ~12 swallows outline, AOT surrounding context becomes background -> clean reconstruction (desktop inpaint_dev DIL=12 proven ~ MIT).
    val maskDilate: Float = 24f,      // Radius 12px: swallow text white outline (before 7=radius 4 left white blocks)
    val bboxPad: Int = 16,            // Inpaint allow region bbox rectangle expansion px: covers furigana at bbox edge (tight line box would miss)
)

data class RenderConfig(
    val orientation: TextOrientation = TextOrientation.AUTO, // [Settings] corresponds to config.render.direction=auto (CJK -> vertical)
    val fontBorder: Boolean = true,                              // [Settings] config.render.disable_font_border=false
    val artStrokeRatio: Float = 0.16f,                           // White outline width for onArt areas (aot reconstructed, onArt) = font size * this (thicker than normal 0.10; better readability for black text with thick white outline on busy background)
    val fontSizeMax: Int = 60,
    val fontSizeMin: Int = 9,
    // Layout geometry (pure text box method, aligned with parity/typeset_parity.py; rarely changed, keep tunable headroom)
    val expandW: Float = 1.3f,   // Text box expansion factor (width) for breathing room
    val expandH: Float = 1.5f,   // Text box expansion factor (vertical column height / horizontal row height)
    val colTrim: Int = 3,        // Vertical: trim few chars per column (shorten column length, reduce overflow; more columns -> auto shrink font)
    val rowTrim: Int = 3,        // Horizontal: trim few chars per row (horizontal counterpart of colTrim: shorter rows, more rows -> auto shrink font)
    val fontScale: Float = 0.85f, // Overall scale after computing font size (<1 = smaller, more fit into cell, leave margin)
    // Text color: auto = determine black/white from background luminance after inpaint (most robust, white bg black text / black bg white text); mono = always black text white outline;
    // other = user-specified fixed text color (ARGB, e.g., 0xFF000000 pure black; outline still determined by background luminance to ensure readability on any background).
    val colorMode: String = "auto",
    val fixedTextColor: Int = 0xFF000000.toInt(), // [Settings] text color when colorMode=fixed (default pure black)
    val bgDark: Int = 110,       // auto: average luminance of background in text box after inpaint < this = dark background -> white text
    // Tate-chu-yoko: when vertical, merge consecutive short ASCII strings (2-4 chars digits/letters/!? ) into one cell displayed horizontally (age "20", year "2020", "!?" no longer stacked vertically). 
    // §4 third layer informed deviation: m-i-t/parity draws char-by-char, no such logic; only affects short ASCII strings inside vertical, CJK unchanged. Default on, can be turned off to char-by-char.
    val tateChuYoko: Boolean = true,
)
