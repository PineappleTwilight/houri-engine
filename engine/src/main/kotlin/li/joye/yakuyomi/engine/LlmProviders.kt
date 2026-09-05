package li.joye.yakuyomi.engine

/**
 * LLM provider presets (multi-provider + automatic model list fetching).
 *
 * Covers m-i-t LLM translators (openai / deepseek / gemini / groq / custom_openai / sakura / qwen2 @ d5a3eee)
 * + OpenRouter convenience preset. **All go through OpenAI-compatible "chat" endpoint** (including Gemini's OpenAI-compat endpoint)
 * => [LlmTranslator] one client handles all, zero new request builders. Difference only in "list models" endpoint (see [ModelSource]).
 *
 * Inspired by nextai-translator's listModels: if `GET /v1/models` works, automatically fetch and follow official updates (models iterate fast, don't hardcode list).
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
 * One provider preset.
 *
 * @param baseEditable Self-hosted / custom (sakura / custom): base filled by user, [chatUrl]/[modelsUrl] left empty and derived from base
 *                     (see [LlmProviders.chatUrlOf]/[LlmProviders.modelsUrlOf]).
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

/**
 * One **request parameter compatibility rule** (data-driven: add provider / special-case model only changes table, not logic, see [LlmProviders.PARAM_RULES]).
 *
 * Why needed: although all are "OpenAI-compatible", consumable fields are actually inconsistent, **even different generations within same provider differ**, sending all blindly is 400 —
 *   · OpenAI reasoning models (o series / gpt-5 series) **reject entire group** of `temperature`/`top_p`/`max_tokens`...
 *   · "Thinking switch" field shape differs per provider: `thinking` / `reasoning_effort` / `enable_thinking` / `reasoning`
 *   · Valid values for same field also differ per generation (`reasoning_effort` none / minimal)
 * Rule itself is pure data => [LlmProviders.requestParams] is pure function, testable (`LlmParamsTest`).
 *
 * @param modelPattern  model id matching (lowercased before matching, using `containsMatchIn`, so anchor with `^` yourself);
 *                      null = no match = fallback rule for that provider (put at end of list).
 * @param temperature   whether to send `temperature` (OpenAI reasoning models reject -> false).
 * @param temperatureRange valid range, out of range -> clamp (OpenAI-compatible mainstream is 0-2).
 * @param maxTokensField "max output tokens" field name (OpenAI reasoning models only accept `max_completion_tokens`).
 *                      Engine currently does not send this field, keep for future (see [LlmProviders.maxTokensFieldOf]).
 * @param thinkingOff   fields to attach when [TranslatorConfig.thinking]=false (default); empty = provider has no such concept or cannot be turned off -> do not send.
 * @param thinkingOn    fields to attach when [TranslatorConfig.thinking]=true; empty = use provider default (most providers default to thinking).
 */
data class ParamRule(
    val modelPattern: Regex? = null,
    val temperature: Boolean = true,
    val temperatureRange: ClosedFloatingPointRange<Double> = 0.0..2.0,
    val maxTokensField: String = "max_tokens",
    val thinkingOff: Map<String, Any> = emptyMap(),
    val thinkingOn: Map<String, Any> = emptyMap(),
)

object LlmProviders {

    /** All m-i-t LLM providers + OpenRouter convenience preset. Order = settings dropdown order. */
    val ALL: List<LlmProvider> = listOf(
        LlmProvider(
            "deepseek", "DeepSeek",
            "https://api.deepseek.com/chat/completions",
            "https://api.deepseek.com/v1/models",
            // deepseek-chat retired 2026-07-24 15:59 UTC (compat shim removed) -> use corresponding
            // deepseek-v4-flash (original deepseek-chat = this model's non-thinking mode). Old name migration see [RETIRED_MODELS].
            ModelSource.OPENAI, "deepseek-v4-flash",
        ),
        LlmProvider(
            "openai", "OpenAI",
            "https://api.openai.com/v1/chat/completions",
            "https://api.openai.com/v1/models",
            ModelSource.OPENAI, "gpt-4o-mini",
        ),
        LlmProvider(
            "gemini", "Google Gemini",
            // Via Gemini's OpenAI-compatible chat endpoint => existing LlmTranslator works directly; list models via native (compat path has no /models).
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "https://generativelanguage.googleapis.com/v1beta/models",
            // Old default gemini-2.0-flash retired 2026-06-01 **retired** (404) -> replaced with official designated alternative, and listed as stable active generic flash on models page. Old id auto migration see [RETIRED_MODELS].
            // https://ai.google.dev/gemini-api/docs/deprecations
            ModelSource.GEMINI, "gemini-3.6-flash",
        ),
        LlmProvider(
            "groq", "Groq",
            "https://api.groq.com/openai/v1/chat/completions",
            "https://api.groq.com/openai/v1/models",
            // Old default llama-3.3-70b-versatile retired 2026-08-16 **retired** -> replaced with Groq's designated alternative, and listed as **production** openai/gpt-oss-120b on models page (other suggestion qwen/qwen3.6-27b is just preview "evaluation only", not default). Retirement passed => old id auto migration see [RETIRED_MODELS] (collected 2026-08-25).
            // https://console.groq.com/docs/deprecations / https://console.groq.com/docs/models
            ModelSource.OPENAI, "openai/gpt-oss-120b",
        ),
        LlmProvider(
            "qwen", "Qwen",
            // **International endpoint** (dashscope-intl = Singapore; verified 2026-08-25, no key returns 401 invalid_api_key).
            // Previously used dashscope.aliyuncs.com is **mainland China** endpoint — this app's audience mostly uses international console (alibabacloud.com) keys, hitting mainland endpoint always 401. Mainland key users please use "custom" provider with https://dashscope.aliyuncs.com/compatible-mode/v1. Official newly promoted {WorkspaceId}.*.maas dedicated domains contain personal workspace id and cannot be preset; official states old domains remain fully functional.
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
            ModelSource.OPENAI, "qwen-plus",
        ),
        LlmProvider(
            "openrouter", "OpenRouter",
            "https://openrouter.ai/api/v1/chat/completions",
            "https://openrouter.ai/api/v1/models",
            // OpenRouter's deepseek/deepseek-chat still exists (=DeepSeek V3, OpenRouter's own namespace, not retired),
            // but since corresponding new-gen deepseek/deepseek-v4-flash is already on OpenRouter (verified via /api/v1/models 2026-07), default aligns with DeepSeek official entry.
            ModelSource.OPENAI, "deepseek/deepseek-v4-flash",
        ),
        // m-i-t sakura: self-hosted JA->ZH specialized LLM (SAKURA_API_BASE, OpenAI-compatible).
        LlmProvider(
            "sakura", "Sakura (self-hosted)",
            "", "", ModelSource.OPENAI, "sakura-14b-qwen2.5-v1.0",
            baseEditable = true,
        ),
        // m-i-t custom_openai: umbrella for OpenRouter / LM Studio / SiliconFlow / any OpenAI-compatible endpoint.
        LlmProvider(
            "custom", "Custom (OpenAI-compatible)",
            "", "", ModelSource.OPENAI, "",
            baseEditable = true,
        ),
    )

    val DEFAULT: LlmProvider = ALL.first() // deepseek

    fun byId(id: String?): LlmProvider = ALL.firstOrNull { it.id == id } ?: DEFAULT

    /**
     * Retired model id -> current replacement (**per-provider**, key = [LlmProvider.id]).
     *
     * 2026-07-24 15:59 UTC: DeepSeek retired `deepseek-chat` / `deepseek-reasoner` old names, compat shim
     * removed together -> old names always get HTTP 400 (`Model Not Exist`). Key / base URL / request format unchanged, **only name needs change**:
     * both originally mapped to `deepseek-v4-flash` non-thinking / thinking modes, so both migrate to `deepseek-v4-flash`.
     *
     * Why not just change [LlmProvider.defaultModel]: if user's model setting **stores** old name (manually entered /
     * picked from old "fetch models" list), default value cannot save them -> migrate in place before sending request (see [migrateModel] caller
     * `LlmTranslator.request`), user does not need manual fix.
     *
     * **Only recognize provider id**: custom / sakura / self-hosted same-name models untouched (that's other's namespace, may really exist).
     *
     * **Only collect "retired = will error when sent" names, not "deprecated but still usable"** — latter forced migration would secretly change user's
     * chosen model (e.g., Groq's `llama-3.3-70b-versatile` during deprecated period official states "Model remains fully
     * functional during this period" => only changed default, not migration then; only after truly retired 2026-08-16 is it collected, see below).
     */
    private val RETIRED_MODELS: Map<String, Map<String, String>> = mapOf(
        "deepseek" to mapOf(
            "deepseek-chat" to "deepseek-v4-flash",      // 舊＝非思考模式
            "deepseek-reasoner" to "deepseek-v4-flash",  // 舊＝思考模式（v4-flash 預設就是思考模式）
        ),
        // Gemini 2.0 系四個 id 於 **2026-06-01 停役**（已不可存取＝送出去報錯，非只是 deprecated）。
        // 替代照官方 deprecations 頁的建議、再對 models 頁挑「列為 stable」的現役 id：
        // Generic flash -> gemini-3.6-flash, lite -> gemini-3.5-flash-lite (keep original price/latency tier, not forced upgrade).
        // https://ai.google.dev/gemini-api/docs/deprecations / https://ai.google.dev/gemini-api/docs/models
        "gemini" to mapOf(
            "gemini-2.0-flash" to "gemini-3.6-flash",
            "gemini-2.0-flash-001" to "gemini-3.6-flash",
            "gemini-2.0-flash-lite" to "gemini-3.5-flash-lite",
            "gemini-2.0-flash-lite-001" to "gemini-3.5-flash-lite",
        ),
        // 2026-08-25 comprehensive audit of all five providers (DeepSeek/OpenAI/Gemini/Qwen/OpenRouter official deprecation pages verified)
        // Collection criteria: **in addition to "retired", also "user likely has it stored"** — this app first release 2026-06-09, any id retired before
        // that (all Gemini preview series, batches before Qwen 2026-01-30, OpenAI o1-mini/o1-preview etc.) could not appear in our "fetch models" list => not collected (manual old id is extreme exception; mixtral exception because m-i-t docs tell users to enter it).
        // [Expiry board: collect next batch when date arrives]
        //   2026-08-31  OpenRouter moonshotai/kimi-k2.5 (official no designated replacement, disappears from /models after expiry => 400)
        //   2026-10-10  Qwen qwen3 batch (qwen3-32b/-coder-plus/-max-preview...-> official points to qwen3.6-flash/3.7-plus/3.7-max)
        //   2026-10-23  OpenAI batch (gpt-4/gpt-4-turbo/gpt-3.5-turbo/o1/o3-mini/o4-mini/gpt-4.1-nano -> gpt-5.6-sol/terra/luna)
        //   2026-12-11  OpenAI snapshot batch (gpt-5/-mini/-nano/-pro 2025-08-07 snapshots, o3/o3-pro -> gpt-5.6 series)
        //   2027-05-07  Gemini gemini-3.1-flash-lite -> gemini-3.5-flash-lite (our lite migration target already points to 3.5, no chain risk)
        // Default models all active: deepseek-v4-flash (2026-07-31 GA) / gpt-4o-mini (not on any retirement list) /
        // gemini-3.6-flash (stable) / openai/gpt-oss-120b (production) / qwen-plus (stable alias) /
        // openrouter deepseek/deepseek-v4-flash (expiration_date:null).
        // Groq three retirement waves (2026-07-17 / 2026-08-16) all collected after expiry (2026-08-25). Replacement per official deprecations page
        // and keep original size/price tier (8b-instant -> 20b, rest -> 120b). migrateModel runs before PARAM_RULES
        // matching => migration target automatically hits existing gpt-oss rule (reasoning_effort=low), no extra param column needed.
        // mixtral-8x7b-32768 retired earlier (m-i-t also fixed its default only 2026-08, PR #1166): if user manually entered per m-i-t docs
        // it will be stored -> migrate together; tier aligns with m-i-t choice (-> gpt-oss-20b).
        // https://console.groq.com/docs/deprecations
        "groq" to mapOf(
            "llama-3.3-70b-versatile" to "openai/gpt-oss-120b",
            "llama-3.1-8b-instant" to "openai/gpt-oss-20b",
            "qwen/qwen3-32b" to "openai/gpt-oss-120b",
            "meta-llama/llama-4-scout-17b-16e-instruct" to "openai/gpt-oss-120b",
            "mixtral-8x7b-32768" to "openai/gpt-oss-20b",
        ),
    )

    /** Migrate model name before sending request: hit [RETIRED_MODELS] -> replace with alternative, else return as-is. */
    fun migrateModel(providerId: String?, model: String): String =
        RETIRED_MODELS[providerId]?.get(model.trim()) ?: model

    // Request parameter compatibility mapping (per provider / per model)

    /** "Max output tokens" field name for OpenAI reasoning models (Chat Completions only accepts this, sending max_tokens is 400). */
    private const val MAX_COMPLETION = "max_completion_tokens"

    /**
     * Provider -> rule list. **First matching [ParamRule.modelPattern] wins** from top to bottom, pattern=null fallback at end;
     * **Providers not in table (custom / sakura / unknown) -> [DEFAULT_RULE] = only send temperature, add no other fields**
     * (self-hosted endpoint compatibility unknown, sending unknown fields is 400).
     *
     * Adding a provider / special-case model = **only add one row of data**, [requestParams] logic does not move.
     */
    private val PARAM_RULES: Map<String, List<ParamRule>> = mapOf(
        // DeepSeek: thinking switch = **top-level object** {"thinking":{"type":"disabled"}} (OpenAI SDK extra_body = body top-level field).
        // v4-flash / v4-pro both support dual mode, **default thinking on** (slower and more expensive than old deepseek-chat) -> we default off = replicate old behavior.
        // thinking=true sends no field (provider default is thinking). In thinking mode temperature/top_p/presence/frequency
        // "unsupported but not error, just ineffective" => sending is fine (only effective when thinking off).
        // https://api-docs.deepseek.com/guides/thinking_mode/
        "deepseek" to listOf(
            ParamRule(thinkingOff = mapOf("thinking" to mapOf("type" to "disabled"))),
        ),
        // OpenAI: largest pitfall = **reasoning models as a group reject sampling params**. Official (Azure same API spec) lists
        // "currently unsupported with reasoning models: temperature, top_p, presence_penalty, frequency_penalty, logprobs, top_logprobs, logit_bias, max_tokens" -> 400, and length limit must be called max_completion_tokens.
        // reasoning_effort valid values also **differ per generation**: none only from gpt-5.1 onward / minimal only first-gen gpt-5 series
        // (removed from gpt-5.1+, gpt-5-codex also not supported) / o series only low|medium|high (o1-mini has no such param at all) /
        // gpt-5-pro only high (= cannot be turned off). gpt-5*-chat is **non**-reasoning chat model (consumes temperature, sending reasoning_effort is 400) => put first to intercept.
        // https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning
        "openai" to listOf(
            ParamRule(Regex("^gpt-5.*chat")), // gpt-5-chat-latest / gpt-5.1-chat: non-reasoning, follow normal rules
            ParamRule(Regex("^o1-mini"), temperature = false, maxTokensField = MAX_COMPLETION), // No reasoning_effort
            // o series (o1/o3/o4-mini/o3-pro/codex-mini): cannot be turned off, only down to low
            ParamRule(
                Regex("^(o\\d|codex-mini)"), temperature = false, maxTokensField = MAX_COMPLETION,
                thinkingOff = mapOf("reasoning_effort" to "low"),
            ),
            ParamRule(Regex("^gpt-5-pro"), temperature = false, maxTokensField = MAX_COMPLETION), // Only supports high = cannot be turned off
            ParamRule( // gpt-5-codex: does not support minimal -> down to low
                Regex("^gpt-5-codex"), temperature = false, maxTokensField = MAX_COMPLETION,
                thinkingOff = mapOf("reasoning_effort" to "low"),
            ),
            ParamRule( // First-gen gpt-5 / -mini / -nano: lowest = minimal (no none)
                Regex("^gpt-5(-mini|-nano)?$"), temperature = false, maxTokensField = MAX_COMPLETION,
                thinkingOff = mapOf("reasoning_effort" to "minimal"),
            ),
            ParamRule( // gpt-5.1 and later (5.1/5.2/5.4/5.5/5.6...): none = no thinking at all
                Regex("^gpt-5\\."), temperature = false, maxTokensField = MAX_COMPLETION,
                thinkingOff = mapOf("reasoning_effort" to "none"),
            ),
            // Other gpt-5 variants not listed: confirmed reasoning model (do not send temperature), but effort value uncertain -> do not send
            ParamRule(Regex("^gpt-5"), temperature = false, maxTokensField = MAX_COMPLETION),
            ParamRule(), // gpt-4o / gpt-4.1 / others = non-reasoning, normal rules
        ),
        // Gemini (via OpenAI-compatible endpoint): thinking via reasoning_effort, compat layer auto maps to thinkingBudget (2.5 series) /
        // thinking_level (3.x). **none only for 2.5 series**; 3.x lowest is minimal (cannot be turned off, only minimized).
        // 2.0 series is not a thinking model (and retired 2026-06-01) -> send nothing.
        // temperature: official changelog 2026-07-21 marks "latest Gemini models" as deprecated, but compat layer states
        // silently ignore unsupported params => sending is fine (same stance as DeepSeek v4); add temperature=false rule later if it starts erroring.
        // https://ai.google.dev/gemini-api/docs/openai
        "gemini" to listOf(
            ParamRule(Regex("^gemini-2\\.5"), thinkingOff = mapOf("reasoning_effort" to "none")),
            ParamRule(Regex("^gemini-[3-9]"), thinkingOff = mapOf("reasoning_effort" to "minimal")),
            ParamRule(),
        ),
        // Groq: reasoning_effort **only some models consume** — Qwen 3.x supports none|default (can truly be turned off), GPT-OSS only
        // low|medium|high (cannot be turned off, only down to low); sending to llama series is 400 "reasoning_effort is not supported with
        // this model" => send nothing for llama. **Current default openai/gpt-oss-120b goes via gpt-oss path**.
        // https://console.groq.com/docs/reasoning
        "groq" to listOf(
            ParamRule(Regex("qwen"), thinkingOff = mapOf("reasoning_effort" to "none")),
            ParamRule(Regex("gpt-oss"), thinkingOff = mapOf("reasoning_effort" to "low")),
            ParamRule(),
        ),
        // Qwen (DashScope compatible-mode): thinking switch = top-level enable_thinking. This engine always stream=false,
        // and DashScope directly returns 400 for "thinking model + non-streaming"
        // "parameter.enable_thinking must be set to false for non-streaming calls"
        // => **both states send false** (cannot get thinking in non-streaming anyway, better guarantee it runs).
        // qwen-plus/max/flash/turbo default off, Qwen3.5+ default on, uniformly explicitly turn off is most stable.
        // https://www.alibabacloud.com/help/en/model-studio/deep-thinking
        // temperature: DashScope official "Range: [0, 2). Do not set to 0." — both ends illegal => clamp to (0,2).
        "qwen" to listOf(
            ParamRule(
                temperatureRange = 0.01..1.99,
                thinkingOff = mapOf("enable_thinking" to false),
                thinkingOn = mapOf("enable_thinking" to false),
            ),
        ),
        // OpenRouter: unified reasoning object, effort="none" = "Disables reasoning entirely". Safe for non-reasoning models too
        // — OpenRouter defaults to "providers that don't support all the LLM parameters ... will ignore unknown
        // parameters" (to exclude that provider you must set require_parameters yourself). Similarly, `openai/o3` handed-off
        // reasoning models even if they receive temperature will be absorbed by OpenRouter => no need to re-list rules per provider here.
        // https://openrouter.ai/docs/use-cases/reasoning-tokens / https://openrouter.ai/docs/features/provider-routing
        "openrouter" to listOf(
            ParamRule(thinkingOff = mapOf("reasoning" to mapOf("effort" to "none"))),
        ),
    )

    /** 表外 provider（custom / sakura / 未知）的保守預設：只送 temperature，其餘一律不送。 */
    private val DEFAULT_RULE = ParamRule()

    /** 解出這次請求該套哪條規則（純函式；model 大小寫不敏感）。 */
    fun ruleFor(providerId: String?, model: String): ParamRule {
        val m = model.trim().lowercase()
        val rules = PARAM_RULES[providerId] ?: return DEFAULT_RULE
        return rules.firstOrNull { it.modelPattern?.containsMatchIn(m) ?: true } ?: DEFAULT_RULE
    }

    /**
     * 這次請求要**附加**到 body 的參數（**不含** model / messages / stream——那三個由 [LlmTranslator] 固定組）。
     *
     * 純函式、無 IO ⇒ 可單測（見 `LlmParamsTest`）。回傳值型別限 String / Boolean / Number / Map（巢狀物件，
     * 如 DeepSeek 的 `thinking:{type:disabled}`），由呼叫端轉成 JSON。
     *
     * @param model      **已經過 [migrateModel]** 的名稱（規則按實際送出的 model 比對）。
     * @param thinking   [TranslatorConfig.thinking]：false（預設）＝送該家「關思考」欄位；true＝用該家預設。
     * @param temperature 會 clamp 到該家合法範圍；該 model 不吃 temperature（OpenAI reasoning 模型）就整個不送。
     */
    fun requestParams(
        providerId: String?,
        model: String,
        thinking: Boolean,
        temperature: Double,
    ): Map<String, Any> {
        val rule = ruleFor(providerId, model)
        val out = LinkedHashMap<String, Any>()
        if (rule.temperature) out["temperature"] = temperature.coerceIn(rule.temperatureRange)
        out.putAll(if (thinking) rule.thinkingOn else rule.thinkingOff)
        return out
    }

    /**
     * 「最大輸出 token」的欄位名（OpenAI reasoning 模型＝`max_completion_tokens`，其餘＝`max_tokens`）。
     * ★引擎目前不設輸出上限、**沒送**這個欄位；映射層先備著，日後要限長（或 fork 要用）直接查這裡。
     */
    fun maxTokensFieldOf(providerId: String?, model: String): String = ruleFor(providerId, model).maxTokensField

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
