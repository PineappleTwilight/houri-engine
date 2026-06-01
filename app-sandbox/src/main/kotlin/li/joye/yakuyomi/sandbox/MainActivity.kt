package li.joye.yakuyomi.sandbox

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import li.joye.yakuyomi.engine.Detector
import li.joye.yakuyomi.engine.Ocr
import li.joye.yakuyomi.engine.TextLine
import li.joye.yakuyomi.sandbox.databinding.ActivityMainBinding

/**
 * M1 sandbox：一顆按鈕 → 偵測(旋轉框) → OCR(48px CTC) → overlay 畫框 + 辨識的日文。
 * 為省記憶體，偵測與 OCR 兩個模型依序載入/釋放（不同時持有）。
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

                // 1) 偵測（用完即釋放）
                val detBytes = assets.open(DETECTOR_MODEL).use { it.readBytes() }
                val lines = Detector(detBytes).use { it.detect(page) }

                // 2) OCR
                val alphabet = assets.open(ALPHABET).bufferedReader().use { it.readLines() }
                val ocrBytes = assets.open(OCR_MODEL).use { it.readBytes() }
                Ocr(ocrBytes, alphabet).use { it.recognize(page, lines) }

                page to lines
            }
            withContext(Dispatchers.Main) {
                result.onSuccess { (page, lines) ->
                    binding.imageView.setImageBitmap(drawResult(page, lines))
                    val withText = lines.count { it.text.isNotBlank() }
                    binding.statusText.text = "完成：${lines.size} 行框，OCR 讀出 $withText 行（藍字）"
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

    private fun drawResult(src: Bitmap, lines: List<TextLine>): Bitmap {
        val out = src.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val boxPaint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 4f
            isAntiAlias = true
        }
        val textFill = Paint().apply {
            color = Color.rgb(0, 90, 255)
            textSize = 28f
            isAntiAlias = true
        }
        val textStroke = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 5f
            textSize = 28f
            isAntiAlias = true
        }
        for (line in lines) {
            val q = line.quad
            if (q.size < 4) continue
            val path = Path().apply {
                moveTo(q[0].x, q[0].y)
                for (i in 1..3) lineTo(q[i].x, q[i].y)
                close()
            }
            canvas.drawPath(path, boxPaint)
            if (line.text.isNotBlank()) {
                val tx = q.minOf { it.x }
                val ty = (q.minOf { it.y } - 6f).coerceAtLeast(28f)
                canvas.drawText(line.text, tx, ty, textStroke)
                canvas.drawText(line.text, tx, ty, textFill)
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
