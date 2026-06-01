package li.joye.yakuyomi.sandbox

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import li.joye.yakuyomi.engine.Detector
import li.joye.yakuyomi.engine.Grouping
import li.joye.yakuyomi.engine.LlmTranslator
import li.joye.yakuyomi.engine.Ocr
import li.joye.yakuyomi.engine.OpenCCS2twp
import li.joye.yakuyomi.engine.TextRegion
import li.joye.yakuyomi.sandbox.databinding.ActivityMainBinding

/**
 * M2 sandbox：偵測 → OCR(日) → 行合併(氣泡) → LLM 翻譯(繁中) → overlay。
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.detectButton.setOnClickListener { runPipeline() }
    }

    private fun runPipeline() {
        binding.detectButton.isEnabled = false
        binding.statusText.text = "載入模型 / 推論中…（首次較久）"
        lifecycleScope.launch(Dispatchers.Default) {
            val result = runCatching {
                val page = loadAssetBitmap(TEST_PAGE)

                val detBytes = assets.open(DETECTOR_MODEL).use { it.readBytes() }
                val lines = Detector(detBytes).use { it.detect(page) }

                val alphabet = assets.open(ALPHABET).bufferedReader().use { it.readLines() }
                val ocrBytes = assets.open(OCR_MODEL).use { it.readBytes() }
                Ocr(ocrBytes, alphabet).use { it.recognize(page, lines) }

                // 同氣泡的行先合併再翻
                val regions = Grouping.group(lines)

                val key = BuildConfig.DEEPSEEK_API_KEY
                if (key.isNotBlank() && regions.isNotEmpty()) {
                    withContext(Dispatchers.Main) { binding.statusText.text = "翻譯中（雲端）…" }
                    val s2twp = OpenCCS2twp(
                        stTexts = listOf(
                            assets.open("opencc/STPhrases.txt").bufferedReader().use { it.readText() },
                            assets.open("opencc/STCharacters.txt").bufferedReader().use { it.readText() },
                        ),
                        twTexts = listOf(assets.open("opencc/TWVariants.txt").bufferedReader().use { it.readText() }),
                    )
                    val cht = LlmTranslator(key, postProcess = { s2twp.convert(it) })
                        .translate(regions.map { it.sourceText })
                    regions.forEachIndexed { i, r -> r.translatedText = cht.getOrElse(i) { r.sourceText } }
                }
                Triple(page, lines.size, regions)
            }
            withContext(Dispatchers.Main) {
                result.onSuccess { (page, lineCount, regions) ->
                    binding.imageView.setImageBitmap(drawResult(page, regions))
                    val translated = regions.count { it.translatedText.isNotBlank() }
                    val keyMsg = if (BuildConfig.DEEPSEEK_API_KEY.isBlank()) "（無 key，略過翻譯）" else ""
                    binding.statusText.text =
                        "完成：$lineCount 行 → ${regions.size} 區；翻譯 $translated $keyMsg"
                }.onFailure { t ->
                    Log.e(TAG, "pipeline 失敗", t)
                    binding.statusText.text = "失敗：${t.message}"
                }
                binding.detectButton.isEnabled = true
            }
        }
    }

    private fun loadAssetBitmap(path: String): Bitmap =
        assets.open(path).use { BitmapFactory.decodeStream(it) }

    private fun drawResult(src: Bitmap, regions: List<TextRegion>): Bitmap {
        val out = src.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val boxPaint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 4f
            isAntiAlias = true
        }
        val textFill = Paint().apply {
            color = Color.rgb(200, 30, 30)
            textSize = 30f
            isAntiAlias = true
        }
        val textStroke = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 6f
            textSize = 30f
            isAntiAlias = true
        }
        for (r in regions) {
            canvas.drawRect(r.x0, r.y0, r.x1, r.y1, boxPaint)
            val label = r.translatedText.ifBlank { r.sourceText }
            if (label.isNotBlank()) {
                val ty = (r.y0 - 6f).coerceAtLeast(30f)
                canvas.drawText(label, r.x0, ty, textStroke)
                canvas.drawText(label, r.x0, ty, textFill)
            }
        }
        return out
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val TEST_PAGE = "test/page.png"
        private const val DETECTOR_MODEL = "models/comictextdetector.pt.onnx"
        private const val OCR_MODEL = "models/ocr_48px_ctc.onnx"
        private const val ALPHABET = "models/alphabet-all-v5.txt"
    }
}
