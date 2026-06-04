package li.joye.yakuyomi.sandbox

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import li.joye.yakuyomi.engine.Detector
import li.joye.yakuyomi.engine.EngineConfig
import li.joye.yakuyomi.engine.Grouping
import li.joye.yakuyomi.engine.InpainterConfig
import li.joye.yakuyomi.engine.Inpainter
import li.joye.yakuyomi.engine.ModelSet
import li.joye.yakuyomi.engine.OcrConfig
import li.joye.yakuyomi.engine.PageResult
import li.joye.yakuyomi.engine.RenderConfig
import li.joye.yakuyomi.engine.TextOrientation
import li.joye.yakuyomi.engine.LlmTranslator
import li.joye.yakuyomi.engine.Ocr
import li.joye.yakuyomi.engine.Renderer
import li.joye.yakuyomi.engine.TranslatorConfig
import li.joye.yakuyomi.engine.Yakuyomi
import li.joye.yakuyomi.sandbox.databinding.ActivityMainBinding

/**
 * 批量測試 sandbox（BYOM）：對 4 張內建 demo 跑完整流程，逐頁 + 總計時，去字模式由開關決定（lama/boxfill）。
 * log + 成品圖寫回所選資料夾，方便撈檔比較兩種去字的速度與品質。
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val prefs by lazy { getSharedPreferences("yakuyomi", MODE_PRIVATE) }

    private val logBuf = StringBuilder()
    private var runTree: DocumentFile? = null
    private var runStamp = ""
    private var runImgIdx = 0
    private var runSaveImg = false

    private val folderPicker = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION,
            )
            prefs.edit().putString(PREF_TREE, uri.toString()).apply()
            binding.logText.text = "已記住資料夾：${DocumentFile.fromTreeUri(this, uri)?.name}"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.pickFolderButton.setOnClickListener { folderPicker.launch(null) }
        binding.detectButton.setOnClickListener { runPipeline() }
        binding.inpaintCompareButton.setOnClickListener { runInpaintCompare() }
        binding.inpaintSpinner.adapter = android.widget.ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, INPAINT_MODES,
        )
        binding.inpaintSpinner.setSelection(3) // 預設＝Auto-逐格（對齊 fork 預設、即正在測的回歸案例）
        binding.orientSpinner.adapter = android.widget.ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, ORIENT_MODES,
        ) // 預設 position 0＝自動（跟原文方向）
        val t = currentTree()
        binding.logText.text =
            if (t == null) "① 先按「選擇模型資料夾」選含 3 個 *.onnx 的資料夾\n② 選去字方式 → 按翻譯跑全 4 張"
            else "資料夾：${t.name}（選去字方式 → 按翻譯跑全 4 張批量計時）"
        // 開機 Toast 標 build 版本：手動安裝後一眼確認裝對版本（沒看到＝還是舊 APK / 同步未完成）
        Toast.makeText(this, "Yakuyomi sandbox $BUILD_TAG", Toast.LENGTH_LONG).show()
    }

    private fun runPipeline() {
        binding.detectButton.isEnabled = false
        val orientation = when (binding.orientSpinner.selectedItemPosition) {
            1 -> TextOrientation.VERTICAL
            2 -> TextOrientation.HORIZONTAL
            else -> TextOrientation.AUTO
        }
        val saveLog = binding.genLogSwitch.isChecked
        runSaveImg = binding.genImgSwitch.isChecked
        val pos = binding.inpaintSpinner.selectedItemPosition // 0=boxfill 1=auto整頁 2=lama整頁 3=auto逐格 4=lama逐格
        val (method, whole, modeLabel) = when (pos) {
            0 -> Triple("boxfill", true, "boxfill")
            1 -> Triple("auto", true, "auto整頁")
            2 -> Triple("lama", true, "lama整頁")
            3 -> Triple("auto", false, "auto逐格")
            else -> Triple("lama", false, "lama逐格")
        }
        lifecycleScope.launch(Dispatchers.Default) {
            clearOutputs()
            logBuf.clear(); runImgIdx = 0; runTree = currentTree(); runStamp = stamp()
            val cfg = EngineConfig(
                ocr = OcrConfig(), // 正式：minProb=0.5 丟低信心誤讀、useXnnpack=false（預設）
                inpainter = InpainterConfig(method = method, wholeImage = whole),
                render = RenderConfig(
                    orientation = orientation,
                ),
            )
            try {
                val tree = runTree
                if (tree == null) {
                    log("✗ 請先按「選擇模型資料夾」")
                    return@launch
                }
                val detF = findOnnx(tree, "detect", "comictext")
                val ocrF = findOnnx(tree, "ocr")
                val lamaF = findOnnx(tree, "lama")
                if (detF == null || ocrF == null || lamaF == null) {
                    log("✗ 模型不齊（需含 detect/ocr/lama 的 3 個 .onnx）")
                    return@launch
                }
                log("▶ 批量測試 ${DEMOS.size} 張｜去字=$modeLabel")
                log("… 載入模型（首次複製到 filesDir 較久）")
                val alphabet = assets.open(ALPHABET).bufferedReader().use { it.readLines() }
                val tf = runCatching { Typeface.createFromAsset(assets, FONT) }.getOrNull()
                val models = ModelSet(ensureLocal(detF), ensureLocal(ocrF), ensureLocal(lamaF))
                log("✓ 模型就緒，開跑")
                // 用工廠取得引擎、`use { }` 自動 close（取代手拼 Detector/Ocr/Inpainter/Pipeline + 逐一 close）
                Yakuyomi.create(models, alphabet, BuildConfig.DEEPSEEK_API_KEY, cfg, tf).use { engine ->
                    var total = 0L
                    DEMOS.forEachIndexed { i, asset ->
                        val tag = "demo${i + 1}"
                        val page = loadAssetBitmap(asset)
                        when (val r = engine.translatePage(page)) { // §11：略過/失敗都保留原圖、不覆蓋
                            is PageResult.Translated -> {
                                val s = r.stats; total += s.totalMs
                                log("[$tag] 偵測${s.detectMs} OCR${s.ocrMs} 譯${s.translateMs} 去字${s.inpaintMs} 排版${s.renderMs}｜頁${s.totalMs} ms｜${s.lines}行${s.regions}區留${s.kept}")
                                addImage("$tag 成品（$modeLabel）", r.page)
                            }
                            is PageResult.Skipped -> {
                                val s = r.stats; total += s.totalMs
                                log("[$tag] 略過：${r.reason}｜偵測${s.detectMs} OCR${s.ocrMs}｜${s.lines}行${s.regions}區（保留原圖、不覆蓋）")
                                addImage("$tag 原圖（略過：${r.reason}）", page)
                            }
                            is PageResult.Failed -> {
                                log("[$tag] ✗ 失敗：${r.reason}（保留原圖、不覆蓋、可重試）")
                                addImage("$tag 原圖（失敗：${r.reason}）", page)
                            }
                        }
                    }
                    log("★ ${DEMOS.size} 頁總計 $total ms（去字=$modeLabel）平均 ${total / DEMOS.size} ms/頁")
                }
            } catch (t: Throwable) {
                Log.e(TAG, "pipeline 失敗", t)
                log("✗✗ 例外：${t.javaClass.simpleName}: ${t.message}")
            } finally {
                if (saveLog) {
                    val ok = writeLog()
                    withContext(Dispatchers.Main) {
                        binding.logText.append(
                            if (ok) "📁 已寫入：${runStamp}_log.txt（+$runImgIdx 圖）\n"
                            else "✗ 寫入失敗：請重按「選擇模型資料夾」重新授權（含寫入）\n",
                        )
                    }
                }
                withContext(Dispatchers.Main) { binding.detectButton.isEnabled = true }
            }
        }
    }

    private fun runInpaintCompare() {
        binding.inpaintCompareButton.isEnabled = false
        lifecycleScope.launch(Dispatchers.Default) {
            clearOutputs()
            logBuf.clear(); runImgIdx = 0; runTree = currentTree(); runStamp = stamp()
            val saveLog = binding.genLogSwitch.isChecked // 去背比較也寫 log 檔（EP/各階段秒才進可讀檔）
            try {
                val tree = runTree
                if (tree == null) {
                    log("✗ 請先按「選擇模型資料夾」")
                    return@launch
                }
                val detF = findOnnx(tree, "detect", "comictext")
                val ocrF = findOnnx(tree, "ocr")
                val lamaF = findOnnx(tree, "lama")
                if (detF == null || ocrF == null || lamaF == null) {
                    log("✗ 模型不齊（需 detect/ocr/lama 的 3 個 .onnx）")
                    return@launch
                }
                val orient = when (binding.orientSpinner.selectedItemPosition) {
                    1 -> TextOrientation.VERTICAL
                    2 -> TextOrientation.HORIZONTAL
                    else -> TextOrientation.AUTO
                }
                val ocrConc = binding.ocrConcSwitch.isChecked // OCR 逐行並發實驗（小圖塊吃不滿 intra-op→單緒並發填核）
                log("▶ 去背比較 3 模式 ×（去字/貼字）+ 總表 — 01.jpg（需連網翻譯）｜OCR=${if (ocrConc) "並發" else "序列"}")
                val page = loadAssetBitmap("test/01.jpg")
                val alphabet = assets.open(ALPHABET).bufferedReader().use { it.readLines() }
                val tf = runCatching { Typeface.createFromAsset(assets, FONT) }.getOrNull()

                // 共用前段（3 模式共用、只跑一次）：偵測→OCR→分群→翻譯→過濾；逐階段計時供總表。
                val detector = Detector(ensureLocal(detF))
                val tD0 = System.currentTimeMillis()
                val detection = detector.detect(page)
                val tDetect = System.currentTimeMillis() - tD0
                val detEp = detector.ep
                detector.close()
                val tO0 = System.currentTimeMillis()
                val ocr = Ocr(ensureLocal(ocrF), alphabet, OcrConfig(concurrent = ocrConc))
                ocr.recognize(page, detection.lines)
                ocr.close()
                val regions = Grouping.group(detection.lines)
                val tOcr = System.currentTimeMillis() - tO0
                val tT0 = System.currentTimeMillis()
                val translator = LlmTranslator(BuildConfig.DEEPSEEK_API_KEY, TranslatorConfig())
                val cht = translator.translate(regions.map { it.sourceText })
                regions.forEachIndexed { j, r -> r.translatedText = cht.getOrElse(j) { r.sourceText } }
                // 等效 engine TextFilter（internal 跨不過 module）：空白/純數字/譯==原 就丟（filterText 此處 null、略過 regex）
                val kept = regions.filter { r ->
                    val t = r.translatedText.trim()
                    t.isNotEmpty() && !t.all { it.isDigit() } && !r.sourceText.trim().equals(t, ignoreCase = true)
                }
                val tTranslate = System.currentTimeMillis() - tT0
                if (kept.isEmpty()) {
                    log("✗ 全數過濾（無有效譯文）：${translator.lastError}｜回應=${translator.lastRaw}")
                    return@launch
                }
                log("共用前段：偵測${"%.1f".format(tDetect / 1000.0)}s OCR${"%.1f".format(tOcr / 1000.0)}s 翻譯${"%.1f".format(tTranslate / 1000.0)}s｜${detection.lines.size}行 ${regions.size}區 留${kept.size}")

                // 3 模式（LaMa-逐格 移除：實測品質≈Auto-逐格、卻多半分鐘）；泡泡三者都平塗，差別只在忙碌區
                val modes = listOf(
                    Triple("BoxFill",   "boxfill", true),
                    Triple("Auto-整頁", "auto",    true),
                    Triple("Auto-逐格", "auto",    false),
                )
                val renderCfg = RenderConfig(orientation = orient)
                val lamaPath = ensureLocal(lamaF)
                val row1 = ArrayList<Bitmap>()   // 第一排：去字（框＋去字秒）
                val row2 = ArrayList<Bitmap>()   // 第二排：貼字（整張秒）
                val timings = ArrayList<Triple<String, Long, Long>>() // name, 去字ms, 排版ms
                var inpEp = "?" // 去字實際 EP（迴圈內捕捉；各模式同設定→同 EP）
                row1.add(labelBmp(page.copy(Bitmap.Config.ARGB_8888, true), "raw", -1L))
                for ((name, method, whole) in modes) {
                    regions.forEach { it.onArt = false; it.dbgStd = -1f } // 重置，避免上一輪 stale 顏色/std
                    val inp = Inpainter(lamaPath, InpainterConfig(method = method, wholeImage = whole))
                    val t0 = System.currentTimeMillis()
                    val cleaned = inp.inpaint(page, kept, detection.textMask)
                    val tInpaint = System.currentTimeMillis() - t0
                    inpEp = inp.ep
                    inp.close()
                    // 第一排：去字 + 路由框（auto 標引擎 std）+ 去字秒（先 copy 乾淨去字圖給第二排貼字）
                    val r1 = cleaned.copy(Bitmap.Config.ARGB_8888, true)
                    markRegions(r1, kept)
                    labelBmp(r1, name, tInpaint)
                    row1.add(r1)
                    // 第二排：貼上譯文 + 整張秒（偵測+OCR+翻譯+去字+排版）
                    val tR0 = System.currentTimeMillis()
                    val rendered = Renderer.render(cleaned, kept, renderCfg, tf)
                    val tRender = System.currentTimeMillis() - tR0
                    labelBmp(rendered, name, tDetect + tOcr + tTranslate + tInpaint + tRender)
                    row2.add(rendered)
                    timings.add(Triple(name, tInpaint, tRender))
                    log("[$name] 去字${"%.1f".format(tInpaint / 1000.0)}s 排版${"%.1f".format(tRender / 1000.0)}s 整張${"%.1f".format((tDetect + tOcr + tTranslate + tInpaint + tRender) / 1000.0)}s")
                }
                // 左下（raw 正下方、本來空白處）＝逐階段時間總表
                row2.add(0, buildTimingTable(page.width, page.height, tDetect, tOcr, tTranslate, timings))

                val big = mergeVertical(listOf(mergeHorizontal(row1), mergeHorizontal(row2)))
                val hw = hwInfoLines(detEp, inpEp, ocrConc)
                hw.forEach { log(it) }
                drawHwInfo(big, hw)
                val cmpName = "${runStamp}_inpaintcmp.png"
                runCatching {
                    tree.findFile(cmpName)?.delete()
                    tree.createFile("image/png", cmpName)?.uri?.let { uri ->
                        contentResolver.openOutputStream(uri)?.use { big.compress(Bitmap.CompressFormat.PNG, 100, it) }
                    }
                }
                log("去背比較完成（${modes.size} 模式 × 去字/貼字 2 排 + 總表）→ 已存圖")
                addImage("去背比較（${modes.size}模式 2排+總表）", big)
            } catch (t: Throwable) {
                Log.e(TAG, "去背比較失敗", t)
                log("✗✗ 例外：${t.javaClass.simpleName}: ${t.message}")
            } finally {
                if (saveLog) {
                    val ok = writeLog()
                    withContext(Dispatchers.Main) {
                        binding.logText.append(
                            if (ok) "📁 已寫入：${runStamp}_log.txt\n"
                            else "✗ log 寫入失敗：請重按「選擇模型資料夾」重新授權（含寫入）\n",
                        )
                    }
                }
                withContext(Dispatchers.Main) { binding.inpaintCompareButton.isEnabled = true }
            }
        }
    }

    /**
     * 在 bmp 上對每個 region 畫彩框：onArt=true→RED(lama 重建)，false→GREEN(boxfill/平塗)。
     * auto 模式另把引擎實測背景 std/亮度標在框左上（dbgStd≥0 才有）＝調 autoStdThreshold 用，直接看引擎真值不靠桌面 parity。
     */
    private fun markRegions(bmp: Bitmap, regions: List<li.joye.yakuyomi.engine.TextRegion>) {
        val canvas = Canvas(bmp)
        val paint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 5f
            isAntiAlias = true
        }
        val txt = Paint().apply {
            color = Color.WHITE; textSize = 26f; isAntiAlias = true; isFakeBoldText = true
        }
        val txtBg = Paint().apply { color = Color.argb(210, 0, 0, 0); style = Paint.Style.FILL }
        for (r in regions) {
            paint.color = if (r.onArt) Color.RED else Color.GREEN
            canvas.drawRect(r.x0, r.y0, r.x1, r.y1, paint)
            if (r.dbgStd >= 0f) {
                val label = "s%.1f w%.0f".format(r.dbgStd, r.dbgWhite)
                val tw = txt.measureText(label)
                val ty = r.y0 + 26f
                canvas.drawRect(r.x0, r.y0, r.x0 + tw + 8f, r.y0 + 32f, txtBg)
                canvas.drawText(label, r.x0 + 4f, ty, txt)
            }
        }
    }

    /** 在 bmp 右上角疊印「name + (秒)」標籤（黑底白字圓角矩形），直接修改傳入的 bmp 並回傳。ms<0 不印時間。 */
    private fun labelBmp(bmp: Bitmap, name: String, ms: Long): Bitmap {
        val canvas = Canvas(bmp)
        val text = if (ms >= 0) "$name %.1fs".format(ms / 1000.0) else name
        val textSize = (bmp.width / 26f).coerceAtLeast(14f)
        val textPaint = Paint().apply {
            color = Color.WHITE
            this.textSize = textSize
            isAntiAlias = true
            typeface = Typeface.MONOSPACE
        }
        val textW = textPaint.measureText(text)
        val pad = textSize * 0.4f
        val boxW = textW + pad * 2
        val boxH = textSize + pad * 2
        val right = bmp.width.toFloat() - pad
        val top = pad
        val bgPaint = Paint().apply {
            color = Color.BLACK
            alpha = 200
            isAntiAlias = true
        }
        canvas.drawRoundRect(RectF(right - boxW, top, right, top + boxH), 8f, 8f, bgPaint)
        canvas.drawText(text, right - boxW + pad, top + pad + textSize * 0.85f, textPaint)
        return bmp
    }

    /** 硬體資訊兩行（裝置/SoC、核數/RAM/去字執行緒）＝去背時間數據的對照背景。 */
    private fun hwInfoLines(detEp: String, inpEp: String, ocrConc: Boolean): List<String> {
        val cores = Runtime.getRuntime().availableProcessors()
        val mi = android.app.ActivityManager.MemoryInfo()
        getSystemService(android.app.ActivityManager::class.java).getMemoryInfo(mi)
        val ramGB = mi.totalMem / (1024.0 * 1024 * 1024)
        val soc = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S)
            "${android.os.Build.SOC_MANUFACTURER} ${android.os.Build.SOC_MODEL}" else android.os.Build.HARDWARE
        val ocrLabel = if (ocrConc) "並發x${OcrConfig().concurrency}(每行單緒)" else "序列(intra-op4)"
        return listOf(
            "$BUILD_TAG ｜ ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL} · $soc",
            "%d核 · %.1fGB RAM ｜ OCR=%s".format(cores, ramGB, ocrLabel),
            "偵測 EP=$detEp ｜ 去字 EP=$inpEp", // EP 確認（應為 XNNPACK）
        )
    }

    /** 在合圖左上角疊印硬體資訊（黑底白字）。 */
    private fun drawHwInfo(bmp: Bitmap, lines: List<String>) {
        val canvas = Canvas(bmp)
        val ts = (bmp.height / 70f).coerceIn(18f, 40f)
        val txt = Paint().apply {
            color = Color.WHITE; textSize = ts; isAntiAlias = true
            typeface = Typeface.MONOSPACE; isFakeBoldText = true
        }
        val bg = Paint().apply { color = Color.BLACK; alpha = 200 }
        val pad = ts * 0.4f
        val maxW = lines.maxOf { txt.measureText(it) }
        canvas.drawRect(0f, 0f, maxW + pad * 2, (ts + pad) * lines.size + pad, bg)
        lines.forEachIndexed { i, line ->
            canvas.drawText(line, pad, pad + ts * 0.85f + i * (ts + pad), txt)
        }
    }

    /** 把多張 Bitmap 橫向拼接（頂部對齊）。 */
    private fun mergeHorizontal(list: List<Bitmap>): Bitmap {
        val h = list.maxOf { it.height }
        val w = list.sumOf { it.width }
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.WHITE)
        var x = 0
        for (bmp in list) {
            canvas.drawBitmap(bmp, x.toFloat(), 0f, null)
            x += bmp.width
        }
        return out
    }

    /** 把多張 Bitmap 縱向堆疊（左邊對齊）。 */
    private fun mergeVertical(list: List<Bitmap>): Bitmap {
        val w = list.maxOf { it.width }
        val h = list.sumOf { it.height }
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.WHITE)
        var y = 0
        for (bmp in list) {
            canvas.drawBitmap(bmp, 0f, y.toFloat(), null)
            y += bmp.height
        }
        return out
    }

    /** 逐階段時間總表（縱＝偵測/OCR/翻譯/去字/排版/整張，橫＝各模式），畫成一張 w×h 格子表填左下空格。 */
    private fun buildTimingTable(
        w: Int, h: Int,
        tDetect: Long, tOcr: Long, tTranslate: Long,
        timings: List<Triple<String, Long, Long>>, // name, 去字ms, 排版ms
    ): Bitmap {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(Color.WHITE)
        val stages = listOf("偵測", "OCR", "翻譯", "去字", "排版", "整張")
        val cols = 1 + timings.size
        val rowsN = 1 + stages.size
        val colW = w.toFloat() / cols
        val rowH = h.toFloat() / rowsN
        val ts = (rowH * 0.3f).coerceIn(20f, 40f)
        val txt = Paint().apply {
            color = Color.BLACK; textSize = ts; isAntiAlias = true; typeface = Typeface.MONOSPACE
        }
        val txtB = Paint().apply {
            color = Color.BLACK; textSize = ts; isAntiAlias = true
            typeface = Typeface.MONOSPACE; isFakeBoldText = true
        }
        val grid = Paint().apply { color = Color.LTGRAY; strokeWidth = 2f }
        for (c in 0..cols) canvas.drawLine(c * colW, 0f, c * colW, rowsN * rowH, grid)
        for (r in 0..rowsN) canvas.drawLine(0f, r * rowH, cols * colW, r * rowH, grid)
        fun put(col: Int, row: Int, s: String, bold: Boolean) {
            val p = if (bold) txtB else txt
            val x = col * colW + (colW - p.measureText(s)) / 2
            val y = row * rowH + rowH / 2 + ts * 0.35f
            canvas.drawText(s, x, y, p)
        }
        fun fmt(ms: Long) = "%.1f".format(ms / 1000.0)
        put(0, 0, "階段/秒", true)
        timings.forEachIndexed { i, t -> put(i + 1, 0, t.first, true) }
        stages.forEachIndexed { si, stage ->
            put(0, si + 1, stage, true)
            timings.forEachIndexed { i, t ->
                val v = when (si) {
                    0 -> fmt(tDetect)
                    1 -> fmt(tOcr)
                    2 -> fmt(tTranslate)
                    3 -> fmt(t.second)
                    4 -> fmt(t.third)
                    else -> fmt(tDetect + tOcr + tTranslate + t.second + t.third)
                }
                put(i + 1, si + 1, v, si == 5) // 整張那列粗體
            }
        }
        return bmp
    }

    private suspend fun log(msg: String) {
        logBuf.append(msg).append('\n')
        withContext(Dispatchers.Main) {
            binding.logText.append("$msg\n")
            binding.scroll.post { binding.scroll.fullScroll(View.FOCUS_DOWN) }
        }
    }

    private suspend fun addImage(label: String, bmp: Bitmap) {
        if (runSaveImg) runTree?.let { saveImage(it, bmp) }
        withContext(Dispatchers.Main) {
            val lbl = TextView(this@MainActivity).apply {
                text = label
                setTypeface(typeface, Typeface.BOLD)
                setPadding(0, 24, 0, 4)
            }
            val iv = ImageView(this@MainActivity).apply {
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT,
                )
                adjustViewBounds = true
                setImageBitmap(scaledForView(bmp))
            }
            binding.container.addView(lbl)
            binding.container.addView(iv)
        }
    }

    private suspend fun clearOutputs() = withContext(Dispatchers.Main) {
        binding.logText.text = ""
        while (binding.container.childCount > FIXED_VIEWS) binding.container.removeViewAt(FIXED_VIEWS)
    }

    private fun stamp(): String =
        java.text.SimpleDateFormat("MMdd_HHmmss", java.util.Locale.US).format(java.util.Date())

    private fun saveImage(tree: DocumentFile, bmp: Bitmap) {
        runImgIdx++
        val name = "${runStamp}_img${"%02d".format(runImgIdx)}.png"
        runCatching {
            tree.findFile(name)?.delete()
            tree.createFile("image/png", name)?.uri?.let { uri ->
                contentResolver.openOutputStream(uri)?.use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
            }
        }
    }

    private fun writeLog(): Boolean = runCatching {
        val tree = runTree ?: return false
        val name = "${runStamp}_log.txt"
        tree.findFile(name)?.delete()
        val uri = tree.createFile("text/plain", name)?.uri ?: return false
        contentResolver.openOutputStream(uri)?.use { it.write(logBuf.toString().toByteArray(Charsets.UTF_8)) }
        true
    }.getOrDefault(false)

    private fun scaledForView(src: Bitmap, maxW: Int = 1000): Bitmap {
        if (src.width <= maxW) return src
        val h = (src.height.toLong() * maxW / src.width).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(src, maxW, h, true)
    }

    /** SAF 模型串流複製到 filesDir（64KB 緩衝、不佔 heap），回傳路徑；已存在且同大小則跳過。 */
    private fun ensureLocal(doc: DocumentFile): String {
        val name = doc.name ?: "model.onnx"
        val out = java.io.File(filesDir, name)
        if (out.exists() && out.length() == doc.length()) return out.absolutePath
        contentResolver.openInputStream(doc.uri)!!.use { input ->
            out.outputStream().use { o -> input.copyTo(o, 1 shl 16) }
        }
        return out.absolutePath
    }

    private fun currentTree(): DocumentFile? {
        val s = prefs.getString(PREF_TREE, null) ?: return null
        return runCatching { DocumentFile.fromTreeUri(this, Uri.parse(s)) }.getOrNull()
    }

    private fun findOnnx(tree: DocumentFile, vararg keywords: String): DocumentFile? =
        tree.listFiles().firstOrNull { f ->
            val n = f.name?.lowercase() ?: return@firstOrNull false
            n.endsWith(".onnx") && keywords.any { n.contains(it) }
        }

    private fun loadAssetBitmap(path: String): Bitmap =
        assets.open(path).use { BitmapFactory.decodeStream(it) }

    companion object {
        private const val TAG = "MainActivity"
        private const val BUILD_TAG = "v0.4-ocrconc8 ｜OCR並發x8 + QNN徹底移除" // 改一次就 bump，手動安裝確認版本用

        // 固定子 view 數：選資料夾鈕/翻譯鈕/去背比較鈕/方向標籤+選單/3 開關(log/圖/OCR並發)/去字標籤+選單/logText＝11。
        // ★ 加/刪任何固定 view（尤其開關）就要同步改這個數，否則 clearOutputs 會把 logText 或末尾固定 view 誤刪（log 消失）。
        private const val FIXED_VIEWS = 11
        private const val PREF_TREE = "modelTree"
        // 排版方向選單（順序＝position：0 自動（跟原文方向）/ 1 直排 / 2 橫排）
        private val ORIENT_MODES = listOf("自動（跟原文方向）", "直排", "橫排")
        // 去字方式選單（順序＝position：0 boxfill / 1 auto整頁 / 2 lama整頁 / 3 auto逐格 / 4 lama逐格，對齊 fork 設定）
        private val INPAINT_MODES = listOf(
            "BoxFill（最速質劣）",
            "Auto-整頁（快速質差）",
            "LaMa-整頁（中速質中）",
            "Auto-逐格（低速質佳）",
            "LaMa-逐格（慢速質高）",
        )
        private val DEMOS = listOf("test/failed.jpg", "test/page.png", "test/demo2.png", "test/demo3.png", "test/demo4.png")
        private const val ALPHABET = "models/alphabet-all-v5.txt"
        private const val FONT = "fonts/NotoSansMonoCJK.ttc"
    }
}
