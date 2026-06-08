package li.joye.yakuyomi.engine

/**
 * LLM provider 預設表（多 provider + 自動撈模型清單）。
 *
 * 涵蓋 m-i-t 的 LLM translator（openai / deepseek / gemini / groq / custom_openai / sakura / qwen2 @ d5a3eee）
 * ＋ OpenRouter 便利預設。**全部走 OpenAI 相容「聊天」端點**（含 Gemini 的 OpenAI-compat 端點）
 * ⇒ [LlmTranslator] 一個 client 通吃、零新請求 builder。差別只在「列模型」端點（見 [ModelSource]）。
 *
 * 借鏡 nextai-translator 的 listModels：能 `GET /v1/models` 就自動撈、跟著官方更新（模型迭代快、不寫死清單）。
 */
enum class ModelSource {
    /** OpenAI 相容：`GET {modelsUrl}` 帶 Bearer → `data[].id`。涵蓋 deepseek/openai/groq/qwen/openrouter/sakura/custom。 */
    OPENAI,

    /** Google Gemini native：`GET {modelsUrl}?key=` → `models[]`（濾 generateContent）→ name 去掉 `models/` 前綴。 */
    GEMINI,

    /** 無清單端點：使用者自行輸入 model id。 */
    NONE,
}

/**
 * 一個 provider 預設。
 *
 * @param baseEditable 自架 / 自訂（sakura / custom）：由使用者填 base，[chatUrl]/[modelsUrl] 留空、由 base 推導
 *                     （見 [LlmProviders.chatUrlOf]/[LlmProviders.modelsUrlOf]）。
 */
data class LlmProvider(
    val id: String,
    val displayName: String,
    val chatUrl: String,
    val modelsUrl: String,
    val modelSource: ModelSource,
    val defaultModel: String,
    val baseEditable: Boolean = false,
)

object LlmProviders {

    /** m-i-t 全部 LLM provider ＋ OpenRouter 便利預設。順序＝設定頁下拉順序。 */
    val ALL: List<LlmProvider> = listOf(
        LlmProvider(
            "deepseek", "DeepSeek",
            "https://api.deepseek.com/chat/completions",
            "https://api.deepseek.com/v1/models",
            ModelSource.OPENAI, "deepseek-chat",
        ),
        LlmProvider(
            "openai", "OpenAI",
            "https://api.openai.com/v1/chat/completions",
            "https://api.openai.com/v1/models",
            ModelSource.OPENAI, "gpt-4o-mini",
        ),
        LlmProvider(
            "gemini", "Google Gemini",
            // 走 Gemini 的 OpenAI 相容聊天端點 ⇒ 既有 LlmTranslator 直接通；列模型走 native（compat 路徑無 /models）。
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "https://generativelanguage.googleapis.com/v1beta/models",
            ModelSource.GEMINI, "gemini-2.0-flash",
        ),
        LlmProvider(
            "groq", "Groq",
            "https://api.groq.com/openai/v1/chat/completions",
            "https://api.groq.com/openai/v1/models",
            ModelSource.OPENAI, "llama-3.3-70b-versatile",
        ),
        LlmProvider(
            "qwen", "通義千問 Qwen",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
            ModelSource.OPENAI, "qwen-plus",
        ),
        LlmProvider(
            "openrouter", "OpenRouter",
            "https://openrouter.ai/api/v1/chat/completions",
            "https://openrouter.ai/api/v1/models",
            ModelSource.OPENAI, "deepseek/deepseek-chat",
        ),
        // m-i-t sakura：自架 JA→ZH 專精 LLM（SAKURA_API_BASE，OpenAI 相容）。
        LlmProvider(
            "sakura", "Sakura（自架）",
            "", "", ModelSource.OPENAI, "sakura-14b-qwen2.5-v1.0",
            baseEditable = true,
        ),
        // m-i-t custom_openai：OpenRouter / LM Studio / SiliconFlow / 任何 OpenAI 相容端點的傘。
        LlmProvider(
            "custom", "自訂（OpenAI 相容）",
            "", "", ModelSource.OPENAI, "",
            baseEditable = true,
        ),
    )

    val DEFAULT: LlmProvider = ALL.first() // deepseek

    fun byId(id: String?): LlmProvider = ALL.firstOrNull { it.id == id } ?: DEFAULT

    /** 最終聊天端點（→ [TranslatorConfig.apiBase]）：baseEditable 用使用者 [base]、否則用預設 [LlmProvider.chatUrl]。 */
    fun chatUrlOf(p: LlmProvider, base: String): String =
        if (p.baseEditable) deriveChatUrl(base) else p.chatUrl

    /** 最終列模型端點：baseEditable 用使用者 [base]、否則用預設 [LlmProvider.modelsUrl]。 */
    fun modelsUrlOf(p: LlmProvider, base: String): String =
        if (p.baseEditable) deriveModelsUrl(base) else p.modelsUrl

    /** 使用者 base → 聊天端點。容忍填 origin / `.../v1` / 完整端點。 */
    private fun deriveChatUrl(base: String): String {
        val b = base.trim().trimEnd('/')
        return when {
            b.isEmpty() -> ""
            b.contains("/chat/completions") || b.endsWith("/completions") -> b
            b.endsWith("/v1") -> "$b/chat/completions"
            else -> "$b/v1/chat/completions"
        }
    }

    /** 使用者 base → 列模型端點（同 origin 的 `/v1/models`）。 */
    private fun deriveModelsUrl(base: String): String {
        var b = base.trim().trimEnd('/')
        if (b.isEmpty()) return ""
        if (b.contains("/chat/completions")) b = b.substringBefore("/chat/completions").trimEnd('/')
        return when {
            b.endsWith("/models") -> b
            b.endsWith("/v1") -> "$b/models"
            else -> "$b/v1/models"
        }
    }
}
