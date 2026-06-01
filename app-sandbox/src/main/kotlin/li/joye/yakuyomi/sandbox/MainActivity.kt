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
import li.joye.yakuyomi.engine.Box
import li.joye.yakuyomi.engine.Detector
import li.joye.yakuyomi.sandbox.databinding.ActivityMainBinding

/**
 * M0 sandbox（CLAUDE.md §13）：一顆按鈕 → engine 偵測 → Canvas 畫紅框。
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.detectButton.setOnClickListener { runDetection() }
    }

    private fun runDetection() {
        binding.detectButton.isEnabled = false
        binding.statusText.text = "載入模型 / 推論中…"
        lifecycleScope.launch(Dispatchers.Default) {
            val result = runCatching {
                val page = loadAssetBitmap(TEST_PAGE)
                val modelBytes = assets.open(MODEL).use { it.readBytes() }
                val boxes = Detector(modelBytes).use { it.detect(page) }
                page to boxes
            }
            withContext(Dispatchers.Main) {
                result.onSuccess { (page, boxes) ->
                    binding.imageView.setImageBitmap(drawBoxes(page, boxes))
                    binding.statusText.text =
                        "完成：偵測到 ${boxes.size} 個框（後處理 M0c 未實作，預期 0；看 logcat 的 output 形狀）"
                }.onFailure { t ->
                    Log.e(TAG, "偵測失敗", t)
                    binding.statusText.text = "失敗：${t.message}"
                }
                binding.detectButton.isEnabled = true
            }
        }
    }

    private fun loadAssetBitmap(path: String): Bitmap =
        assets.open(path).use { BitmapFactory.decodeStream(it) }

    private fun drawBoxes(src: Bitmap, boxes: List<Box>): Bitmap {
        val out = src.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(out)
        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 4f
            isAntiAlias = true
        }
        for (b in boxes) {
            canvas.drawRect(
                b.x.toFloat(),
                b.y.toFloat(),
                (b.x + b.w).toFloat(),
                (b.y + b.h).toFloat(),
                paint,
            )
        }
        return out
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val TEST_PAGE = "test/page.png"

        // engine module 的 assets 會 merge 進 app，故可直接用此路徑開啟
        private const val MODEL = "models/comictextdetector.pt.onnx"
    }
}
