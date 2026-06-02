package li.joye.yakuyomi.sandbox

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
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
import li.joye.yakuyomi.engine.Inpainter
import li.joye.yakuyomi.engine.LlmTranslator
import li.joye.yakuyomi.engine.Ocr
import li.joye.yakuyomi.engine.OcrConfig
import li.joye.yakuyomi.engine.Pt
import li.joye.yakuyomi.engine.RenderConfig
import li.joye.yakuyomi.engine.Renderer
import li.joye.yakuyomi.engine.TextFilter
import li.joye.yakuyomi.engine.TextLine
import li.joye.yakuyomi.engine.TextOrientation
import li.joye.yakuyomi.sandbox.databinding.ActivityMainBinding

/**
 * 診斷 sandbox（BYOM）：模型不內建，由使用者用 SAF 選一個含 *.onnx 的資料夾，記住偏好。
 * 逐步印 log（載入/各關數量/翻譯錯誤/OCR 樣本）+ 每步出圖（原圖→偵測→OCR→去字→成品），可滾動截圖。
 * 診斷期 OCR 信心門檻設 0（看原始輸出）。字典/字型/測試頁仍在 assets。
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val prefs by lazy { getSharedPreferences("yakuyomi", MODE_PRIVATE) }

    // log system：累積純文字 log + 計數，run 結束寫回所選資料夾（log 檔 + debug 圖），方便撈檔分析
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
            binding.logText.text = "已記住資料夾：${DocumentFile.fromTreeUri(this, uri)?.name}\n按「翻譯這一頁（診斷）」開始"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.pickFolderButton.setOnClickListener { folderPicker.launch(null) }
        binding.detectButton.setOnClickListener { runPipeline() }
        val t = currentTree()
        binding.logText.text =
            if (t == null) "① 先按「選擇模型資料夾」，選含 3 個 *.onnx 的資料夾（OneDrive/下載皆可）\n② 再按「翻譯這一頁（診斷）」"
            else "模型資料夾：${t.name}（已記住，可直接按翻譯）"
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

    private fun readUri(uri: Uri): ByteArray =
        contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: error("讀取失敗 $uri")

    private fun runPipeline() {
        binding.detectButton.isEnabled = false
        val vertical = binding.verticalSwitch.isChecked
        val saveLog = binding.genLogSwitch.isChecked
        runSaveImg = binding.genImgSwitch.isChecked
        lifecycleScope.launch(Dispatchers.Default) {
            clearOutputs()
            logBuf.clear()
            runImgIdx = 0
            runTree = currentTree()
            runStamp = stamp()
            val cfg = EngineConfig(
                ocr = OcrConfig(minProb = 0f), // 【診斷】先不丟低信心，看 OCR 原始輸出
                render = RenderConfig(
                    orientation = if (vertical) TextOrientation.VERTICAL else TextOrientation.HORIZONTAL,
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
                log("模型資料夾：${tree.name}")
                log("  detector: ${detF?.name ?: "✗ 缺"}")
                log("  ocr     : ${ocrF?.name ?: "✗ 缺"}")
                log("  lama    : ${lamaF?.name ?: "✗ 缺"}")
                if (detF == null || ocrF == null || lamaF == null) {
                    log("✗ 模型不齊（資料夾要有含 detect/ocr/lama 的 3 個 .onnx），停止")
                    return@launch
                }

                log("▶ 開始")
                val page = loadAssetBitmap(TEST_PAGE)
                log("✓ 載入測試頁 ${page.width}×${page.height}")
                addImage("① 原圖", page)

                log("… 載入 detector（${detF.name}）")
                val detBytes = readUri(detF.uri)
                log("✓ detector ${detBytes.size / 1048576}MB 讀入")
                val lines = Detector(detBytes, cfg.detector).use { it.detect(page) }
                log("✓ 偵測完成：${lines.size} 框")
                addImage("② 偵測框（${lines.size}）", overlayBoxes(page, lines))

                log("… 載入 OCR（${ocrF.name}）+ 字典 + 字型")
                val alphabet = assets.open(ALPHABET).bufferedReader().use { it.readLines() }
                val ocrBytes = readUri(ocrF.uri)
                log("✓ OCR ${ocrBytes.size / 1048576}MB、字典 ${alphabet.size} 條")
                val tf = runCatching { Typeface.createFromAsset(assets, FONT) }.getOrNull()
                Ocr(ocrBytes, alphabet, cfg.ocr).use { ocr ->
                    lines.take(3).forEachIndexed { i, ln ->
                        val d = ocr.debugOne(page, ln)
                        log("  行$i: ${d.info}")
                        d.strip?.let { addImage("OCR 圖塊 行$i（餵給模型的）", it) }
                    }
                    ocr.recognize(page, lines)
                }
                val ocrCount = lines.count { it.text.isNotBlank() }
                log("✓ OCR：$ocrCount/${lines.size} 行有字")
                log("  樣本：" + lines.take(5).joinToString(" ┊ ") { it.text.ifBlank { "∅" } })
                addImage("③ OCR（綠=有字 紅=無）", overlayOcr(page, lines, tf))

                val regions = Grouping.group(lines)
                log("✓ 分區：${regions.size} 區")

                val key = BuildConfig.DEEPSEEK_API_KEY
                if (key.isBlank()) {
                    log("⚠ 無 API key，跳過翻譯（排版日文）")
                } else {
                    log("… 翻譯中（DeepSeek，${regions.size} 區）")
                    val tr = LlmTranslator(key, cfg.translator)
                    val cht = tr.translate(regions.map { it.sourceText })
                    regions.forEachIndexed { i, r -> r.translatedText = cht.getOrElse(i) { r.sourceText } }
                    if (tr.lastError != null) log("✗ 翻譯失敗：${tr.lastError}") else log("✓ 翻譯回應 OK")
                }
                val tcount = regions.count { it.translatedText.isNotBlank() && it.translatedText != it.sourceText }
                log("  譯成功 $tcount/${regions.size}")

                log("… 載入 LaMa（${lamaF.name}）+ 去字")
                val lamaBytes = readUri(lamaF.uri)
                log("✓ LaMa ${lamaBytes.size / 1048576}MB 讀入")
                val cleaned = Inpainter(lamaBytes, cfg.inpainter).use { it.inpaint(page, regions) }
                log("✓ 去字完成")
                addImage("④ 去字/塗白", cleaned)

                log("… 排版（診斷：未譯則排日文）")
                val finalPage = Renderer.render(cleaned, regions, cfg.render, tf)
                log("✓ 排版完成")
                addImage("⑤ 成品", finalPage)
                log("■ 全部完成")
            } catch (t: Throwable) {
                Log.e(TAG, "pipeline 失敗", t)
                log("✗✗ 例外：${t.javaClass.simpleName}: ${t.message}")
            } finally {
                if (saveLog) {
                    val ok = writeLog()
                    withContext(Dispatchers.Main) {
                        binding.logText.append(
                            if (ok) "📁 已寫入資料夾：${runStamp}_log.txt（+$runImgIdx 圖）\n"
                            else "✗ 寫入資料夾失敗：請重按「選擇模型資料夾」重新授權（含寫入）\n",
                        )
                    }
                }
                withContext(Dispatchers.Main) { binding.detectButton.isEnabled = true }
            }
        }
    }

    private suspend fun log(msg: String) {
        logBuf.append(msg).append('\n')
        withContext(Dispatchers.Main) {
            binding.logText.append("$msg\n")
            binding.scroll.post { binding.scroll.fullScroll(View.FOCUS_DOWN) }
        }
    }

    private suspend fun addImage(label: String, bmp: Bitmap) {
        if (runSaveImg) runTree?.let { saveImage(it, bmp) } // 寫原始解析度 PNG 回資料夾
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

    private fun stamp(): String =
        java.text.SimpleDateFormat("MMdd_HHmmss", java.util.Locale.US).format(java.util.Date())

    /** 存一張 debug 圖（原始解析度 PNG）到所選資料夾。 */
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

    /** 把本次 run 的純文字 log 寫回所選資料夾；回傳是否成功（失敗多半是沒寫入授權）。 */
    private fun writeLog(): Boolean = runCatching {
        val tree = runTree ?: return false
        val name = "${runStamp}_log.txt"
        tree.findFile(name)?.delete()
        val uri = tree.createFile("text/plain", name)?.uri ?: return false
        contentResolver.openOutputStream(uri)?.use { it.write(logBuf.toString().toByteArray(Charsets.UTF_8)) }
        true
    }.getOrDefault(false)

    /** 清掉上次的圖與 log（保留固定的 6 個 view：選資料夾鈕/翻譯鈕/3 開關/logText）。 */
    private suspend fun clearOutputs() = withContext(Dispatchers.Main) {
        binding.logText.text = ""
        while (binding.container.childCount > FIXED_VIEWS) binding.container.removeViewAt(FIXED_VIEWS)
    }

    private fun scaledForView(src: Bitmap, maxW: Int = 1000): Bitmap {
        if (src.width <= maxW) return src
        val h = (src.height.toLong() * maxW / src.width).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(src, maxW, h, true)
    }

    private fun polyPath(quad: List<Pt>): Path = Path().apply {
        if (quad.isEmpty()) return@apply
        moveTo(quad[0].x, quad[0].y)
        for (i in 1 until quad.size) lineTo(quad[i].x, quad[i].y)
        close()
    }

    private fun overlayBoxes(page: Bitmap, lines: List<TextLine>): Bitmap {
        val out = page.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(out)
        val p = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 5f; color = Color.RED; isAntiAlias = true }
        for (ln in lines) c.drawPath(polyPath(ln.quad), p)
        return out
    }

    private fun overlayOcr(page: Bitmap, lines: List<TextLine>, tf: Typeface?): Bitmap {
        val out = page.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(out)
        val box = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f; isAntiAlias = true }
        val halo = Paint().apply {
            color = Color.WHITE; style = Paint.Style.FILL_AND_STROKE; strokeWidth = 6f
            textSize = 30f; isAntiAlias = true; typeface = tf
        }
        val ink = Paint().apply { color = Color.rgb(0, 110, 0); textSize = 30f; isAntiAlias = true; typeface = tf }
        for (ln in lines) {
            val has = ln.text.isNotBlank()
            box.color = if (has) Color.rgb(0, 170, 0) else Color.RED
            c.drawPath(polyPath(ln.quad), box)
            if (has && ln.quad.isNotEmpty()) {
                val x = ln.quad.minOf { it.x }
                val y = ln.quad.minOf { it.y } - 6f
                c.drawText(ln.text, x, y, halo)
                c.drawText(ln.text, x, y, ink)
            }
        }
        return out
    }

    private fun loadAssetBitmap(path: String): Bitmap =
        assets.open(path).use { BitmapFactory.decodeStream(it) }

    companion object {
        private const val TAG = "MainActivity"
        private const val FIXED_VIEWS = 6 // 容器內固定子 view 數（選資料夾鈕/翻譯鈕/3 開關/logText），其後為動態圖
        private const val PREF_TREE = "modelTree"
        private const val TEST_PAGE = "test/page.png"
        private const val ALPHABET = "models/alphabet-all-v5.txt"
        private const val FONT = "fonts/NotoSansMonoCJK.ttc"
    }
}
