package li.joye.yakuyomi.sandbox

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
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
import li.joye.yakuyomi.engine.OpenCCS2twp
import li.joye.yakuyomi.engine.RenderConfig
import li.joye.yakuyomi.engine.Renderer
import li.joye.yakuyomi.engine.TextOrientation
import li.joye.yakuyomi.sandbox.databinding.ActivityMainBinding

/**
 * 端到端 sandbox：偵測 → OCR(日) → 行合併 → 翻譯(繁中) → 去字(LaMa) → 排版 → 成品頁。
 * 引擎參數走 EngineConfig（這裡只覆寫直/橫排，其餘用預設；未來設定頁覆寫更多）。
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
        val cfg = EngineConfig(
            render = RenderConfig(
                orientation = if (binding.verticalSwitch.isChecked) {
                    TextOrientation.VERTICAL
                } else {
                    TextOrientation.HORIZONTAL
                },
            ),
        )
        lifecycleScope.launch(Dispatchers.Default) {
            val result = runCatching {
                val page = loadAssetBitmap(TEST_PAGE)

                val detBytes = assets.open(DETECTOR_MODEL).use { it.readBytes() }
                val lines = Detector(detBytes, cfg.detector).use { it.detect(page) }

                val alphabet = assets.open(ALPHABET).bufferedReader().use { it.readLines() }
                val ocrBytes = assets.open(OCR_MODEL).use { it.readBytes() }
                Ocr(ocrBytes, alphabet, cfg.ocr).use { it.recognize(page, lines) }

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
                    val cht = LlmTranslator(key, cfg.translator, postProcess = { s2twp.convert(it) })
                        .translate(regions.map { it.sourceText })
                    regions.forEachIndexed { i, r -> r.translatedText = cht.getOrElse(i) { r.sourceText } }
                }

                withContext(Dispatchers.Main) { binding.statusText.text = "去字 + 排版…" }
                val lamaBytes = assets.open(LAMA_MODEL).use { it.readBytes() }
                val cleaned = Inpainter(lamaBytes, cfg.inpainter).use { it.inpaint(page, regions) }
                val finalPage = Renderer.render(cleaned, regions, cfg.render)

                Triple(finalPage, lines.size, regions.size)
            }
            withContext(Dispatchers.Main) {
                result.onSuccess { (finalPage, lineCount, regionCount) ->
                    binding.imageView.setImageBitmap(finalPage)
                    val keyMsg = if (BuildConfig.DEEPSEEK_API_KEY.isBlank()) "（無 key：排版日文）" else ""
                    binding.statusText.text = "完成：$lineCount 行 → $regionCount 區，去字+排版 $keyMsg"
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

    companion object {
        private const val TAG = "MainActivity"
        private const val TEST_PAGE = "test/page.png"
        private const val DETECTOR_MODEL = "models/comictextdetector.pt.onnx"
        private const val OCR_MODEL = "models/ocr_48px_ctc.onnx"
        private const val LAMA_MODEL = "models/lama-manga.onnx"
        private const val ALPHABET = "models/alphabet-all-v5.txt"
    }
}
