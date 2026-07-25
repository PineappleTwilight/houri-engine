# LLM providers and model lists

English ｜ [中文](PROVIDERS_zh.md)

Translation is the one pipeline stage that leaves the device. It calls a cloud LLM with the line-numbered prompt and protocol ported from manga-image-translator. The engine ships a list of provider presets and can fetch each provider's live model list, so you bring your own key and pick a current model without editing code.

## One chat path for every provider

Every supported provider speaks the OpenAI chat-completions shape (`POST {base}` with `{ model, messages, stream }` and `Authorization: Bearer <key>`). `LlmTranslator` is the only request builder; switching providers changes nothing but `apiBase`, `model`, and the key. Everything beyond those three fields — `temperature`, the thinking switch — goes through a compatibility map, because "OpenAI-compatible" is not the same as "accepts the same fields" (see below). Google Gemini is included by pointing chat at its OpenAI-compatible endpoint, so it needs no separate code either — only its model *listing* uses the native endpoint, because the compat path has no `/models`.

This is deliberate. The providers Yakuyomi targets are exactly manga-image-translator's LLM translators — OpenAI, DeepSeek, Gemini, Groq, custom_openai, Sakura, Qwen — all OpenAI-compatible. The non-LLM machine-translation services m-i-t also supports (DeepL, Caiyun, Youdao, Baidu, Papago) have no chat/prompt protocol and are out of scope.

## Provider presets

`LlmProviders.ALL` (engine) holds the presets. Each is `{ id, displayName, chatUrl, modelsUrl, modelSource, defaultModel, baseEditable }`.

| Preset | Host | Model list | Notes |
|---|---|---|---|
| DeepSeek (default) | api.deepseek.com | `/v1/models` | cheap, strong Chinese |
| OpenAI | api.openai.com | `/v1/models` | |
| Google Gemini | …/v1beta/openai/ | `/v1beta/models` (native) | chat via OpenAI-compat endpoint; list via native |
| Groq | api.groq.com | `/v1/models` | fast, free tier |
| Qwen (通義千問) | dashscope…/compatible-mode | `/v1/models` | Alibaba |
| OpenRouter | openrouter.ai | `/v1/models` (public) | one key, hundreds of models, list auto-updates |
| Sakura (self-hosted) | your base | your base + `/v1/models` | JP→ZH-specialised LLM |
| Custom (OpenAI-compatible) | your base | your base + `/v1/models` | LM Studio, SiliconFlow, any compatible endpoint |

Two presets are `baseEditable` (Sakura, Custom): the user supplies the base URL and the engine derives the chat and model-list URLs from it (`LlmProviders.chatUrlOf` / `modelsUrlOf`).

## Per-provider request parameters

Providers agree on `model` / `messages` / `stream` and disagree on everything else — a field one provider requires is a 400 at the next one, and the rules change between model generations of the *same* provider. `LlmProviders.PARAM_RULES` is a data table of `ParamRule`s (`provider → [rule]`, first rule whose `modelPattern` matches the model id wins, `null` pattern last as the fallback); `LlmProviders.requestParams(provider, model, thinking, temperature)` is a pure function returning the extra body fields, and `LlmTranslator` just merges them into the JSON. Adding a provider or a special-case model is a table row, not a code change.

| Provider / model | thinking off (default) | thinking on | `temperature` | Notes |
|---|---|---|---|---|
| DeepSeek | `thinking: {"type":"disabled"}` | — (thinks by default) | sent | v4-flash / v4-pro both default to thinking; while thinking, sampling params are accepted but ignored ([docs](https://api-docs.deepseek.com/guides/thinking_mode/)) |
| OpenAI · o-series (`o1`, `o3`, `o4-mini`, `codex-mini`) | `reasoning_effort: "low"` | — | **omitted** | reasoning models reject `temperature`, `top_p`, `max_tokens`…; length cap is `max_completion_tokens`; `o1-mini` has no `reasoning_effort`; reasoning can't be disabled, only lowered ([docs](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning)) |
| OpenAI · GPT-5 series | `"none"` (5.1 and later), `"minimal"` (original gpt-5/-mini/-nano), `"low"` (gpt-5-codex) | — | **omitted** | `gpt-5-pro` only accepts `high`, so nothing is sent |
| OpenAI · `gpt-5*-chat`, `gpt-4o`, `gpt-4.1`, … | — | — | sent | non-reasoning models: `reasoning_effort` would 400 |
| Gemini (OpenAI-compat) | `reasoning_effort: "none"` (2.5), `"minimal"` (3.x, incl. the default `gemini-3.6-flash`) | — | sent | `none` is 2.5-only; 3.x can be minimised but not disabled ([docs](https://ai.google.dev/gemini-api/docs/openai)) |
| Groq | `reasoning_effort: "none"` (Qwen), `"low"` (GPT-OSS, incl. the default `openai/gpt-oss-120b`) | — | sent | GPT-OSS only accepts low/medium/high, so `low` is as close to off as it gets; Llama models 400 on the field entirely ([docs](https://console.groq.com/docs/reasoning)) |
| Qwen (DashScope compat) | `enable_thinking: false` | `enable_thinking: false` | sent | DashScope rejects a non-streaming call to a thinking model unless this is explicitly false, and the engine never streams ([docs](https://www.alibabacloud.com/help/en/model-studio/deep-thinking)) |
| OpenRouter | `reasoning: {"effort":"none"}` | — | sent | `effort: none` disables reasoning entirely; unsupported params are ignored by default, so this is safe on any model ([docs](https://openrouter.ai/docs/use-cases/reasoning-tokens)) |
| Sakura, Custom, unknown | — | — | sent | self-hosted endpoints get nothing extra: an unknown field can be a 400 |

"—" means no field is sent, i.e. the provider's own default applies. `ParamRule` also carries `maxTokensField` (`max_tokens` vs `max_completion_tokens`); the engine sets no output cap today, so nothing is sent, but the mapping is there for when it does. `LlmParamsTest` locks the table down (o-series drops `temperature`, DeepSeek disables thinking, self-hosted sends nothing extra, …).

## Retired model ids

Providers retire model names, and a stored setting then fails forever. `LlmProviders.RETIRED_MODELS` (per provider id) renames them at request time, so nobody has to edit a setting they never chose:

| Provider | Retired id | Replaced with | Why |
|---|---|---|---|
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` | `deepseek-v4-flash` | removed 2026-07-24 with no compat shim — the old names 400 with "Model Not Exist" |
| Gemini | `gemini-2.0-flash`, `gemini-2.0-flash-001` | `gemini-3.6-flash` | shut down 2026-06-01 ([deprecations](https://ai.google.dev/gemini-api/docs/deprecations)) |
| Gemini | `gemini-2.0-flash-lite`, `gemini-2.0-flash-lite-001` | `gemini-3.5-flash-lite` | same shutdown; kept on the lite tier so the price/latency class doesn't change |

The table only takes ids that are **actually dead**. A model that is merely *deprecated* but still served — Groq's `llama-3.3-70b-versatile`, functional until its 2026-08-16 shutdown — is left alone: silently swapping a working model the user picked is worse than the deprecation notice. Only the preset default moves in that case (Groq's default is now `openai/gpt-oss-120b`, Groq's own recommended production replacement; the other suggestion, `qwen/qwen3.6-27b`, is preview-only). `LlmParamsTest` asserts no preset default is itself a retired id.

## Fetching the model list

Models iterate fast, so the engine fetches each provider's current list instead of hardcoding it — an idea borrowed from [nextai-translator](https://github.com/nextai-translator/nextai-translator). `LlmModels.list(modelsUrl, source, apiKey)` returns `List<ModelInfo>`; the reader's settings call it behind a "fetch models" button and show the result as a picker.

Two endpoint shapes, by `ModelSource`:

- **`OPENAI`** — `GET {modelsUrl}` with `Authorization: Bearer`, parse `data[].id`. Covers every OpenAI-compatible provider.
- **`GEMINI`** — `GET {modelsUrl}?key=`, parse `models[]`, keep those whose `supportedGenerationMethods` include `generateContent`, strip the `models/` prefix.

Anything else — no key, an endpoint without a listing API, a network error — returns an empty list. The fetch never throws and never blocks configuration; an empty result just means "type the model id by hand". The auto-list is a convenience, not a prerequisite.

## Bring your own key (per provider)

Keys are stored per provider, so switching from DeepSeek to OpenRouter and back keeps each one. The reader stores them encrypted (Android Keystore) and passes only the active provider's key to the engine as a constructor argument; the engine itself holds no keys.

## Adding a provider

Because every provider is OpenAI-compatible, adding one is usually a single entry in `LlmProviders.ALL`:

```kotlin
LlmProvider(
    id = "myprovider",
    displayName = "My Provider",
    chatUrl = "https://api.example.com/v1/chat/completions",
    modelsUrl = "https://api.example.com/v1/models",
    modelSource = ModelSource.OPENAI,
    defaultModel = "some-model",
)
```

No request-builder change is needed. A provider that isn't OpenAI-compatible — a different chat shape, like Anthropic's Messages API — would need a new builder in the engine; none of the in-scope providers do.

## Why these, and why auto-list

The provider set is the intersection of what manga-image-translator targets and what's useful for Japanese→Chinese manga, plus OpenRouter as an aggregator. The auto-list mirrors nextai-translator's `listModels` but leans on it harder: nextai hardcodes static model lists for several providers (Claude, Kimi, Zhipu, MiniMax), whereas Yakuyomi fetches `/v1/models` for all of them and falls back to manual entry — so there is no static list to maintain as models change.
