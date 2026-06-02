package li.joye.yakuyomi.sandbox

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import li.joye.yakuyomi.engine.InpainterConfig
import li.joye.yakuyomi.engine.LlmTranslator
import li.joye.yakuyomi.engine.Ocr
import li.joye.yakuyomi.engine.OcrConfig
import li.joye.yakuyomi.engine.RenderConfig
import li.joye.yakuyomi.engine.Renderer
import li.joye.yakuyomi.engine.TextOrientation
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
        val t = currentTree()
        binding.logText.text =
            if (t == null) "① 先按「選擇模型資料夾」選含 3 個 *.onnx 的資料夾\n② 切「去字用 LaMa」開關，按翻譯跑全 4 張"
            else "資料夾：${t.name}（切去字開關 → 按翻譯跑全 4 張批量計時）"
    }

    private fun runPipeline() {
        binding.detectButton.isEnabled = false
        val vertical = binding.verticalSwitch.isChecked
        val saveLog = binding.genLogSwitch.isChecked
        runSaveImg = binding.genImgSwitch.isChecked
        val method = if (binding.lamaSwitch.isChecked) "lama" else "boxfill"
        lifecycleScope.launch(Dispatchers.Default) {
            clearOutputs()
            logBuf.clear(); runImgIdx = 0; runTree = currentTree(); runStamp = stamp()
            val cfg = EngineConfig(
                ocr = OcrConfig(minProb = 0f, useXnnpack = false), // 診斷：不丟低信心 + OCR 純 CPU
                inpainter = InpainterConfig(method = method),
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
                if (detF == null || ocrF == null || lamaF == null) {
                    log("✗ 模型不齊（需含 detect/ocr/lama 的 3 個 .onnx）")
                    return@launch
                }
                log("▶ 批量測試 ${DEMOS.size} 張｜去字=$method")
                log("… 載入模型（首次複製到 filesDir 較久）")
                val alphabet = assets.open(ALPHABET).bufferedReader().use { it.readLines() }
                val tf = runCatching { Typeface.createFromAsset(assets, FONT) }.getOrNull()
                val det = Detector(ensureLocal(detF), cfg.detector)
                val ocr = Ocr(ensureLocal(ocrF), alphabet, cfg.ocr)
                val inp = Inpainter(ensureLocal(lamaF), cfg.inpainter)
                val key = BuildConfig.DEEPSEEK_API_KEY
                val translator = if (key.isNotBlank()) LlmTranslator(key, cfg.translator) else null
                log("✓ 模型就緒，開跑")
                try {
                    var total = 0L
                    DEMOS.forEachIndexed { i, asset ->
                        val tag = "demo${i + 1}"
                        val page = loadAssetBitmap(asset)
                        val t0 = System.currentTimeMillis()
                        var ms = System.currentTimeMillis()
                        val lines = det.detect(page); val detMs = System.currentTimeMillis() - ms
                        ms = System.currentTimeMillis()
                        ocr.recognize(page, lines); val ocrMs = System.currentTimeMillis() - ms
                        val regions = Grouping.group(lines)
                        var trMs = 0L
                        if (translator != null && regions.isNotEmpty()) {
                            ms = System.currentTimeMillis()
                            val cht = translator.translate(regions.map { it.sourceText })
                            regions.forEachIndexed { j, r -> r.translatedText = cht.getOrElse(j) { r.sourceText } }
                            trMs = System.currentTimeMillis() - ms
                        }
                        ms = System.currentTimeMillis()
                        val cleaned = inp.inpaint(page, regions); val inMs = System.currentTimeMillis() - ms
                        ms = System.currentTimeMillis()
                        val finalPage = Renderer.render(cleaned, regions, cfg.render, tf); val rnMs = System.currentTimeMillis() - ms
                        val pageMs = System.currentTimeMillis() - t0
                        total += pageMs
                        log("[$tag] 偵測$detMs OCR$ocrMs 譯$trMs 去字$inMs 排版$rnMs｜頁$pageMs ms｜${lines.size}行${regions.size}區")
                        addImage("$tag 成品（$method）", finalPage)
                    }
                    log("★ ${DEMOS.size} 頁總計 $total ms（去字=$method）平均 ${total / DEMOS.size} ms/頁")
                } finally {
                    det.close(); ocr.close(); inp.close()
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
        private const val FIXED_VIEWS = 7 // 固定子 view：選資料夾鈕/翻譯鈕/4 開關/logText
        private const val PREF_TREE = "modelTree"
        private val DEMOS = listOf("test/page.png", "test/demo2.png", "test/demo3.png", "test/demo4.png")
        private const val ALPHABET = "models/alphabet-all-v5.txt"
        private const val FONT = "fonts/NotoSansMonoCJK.ttc"
    }
}
