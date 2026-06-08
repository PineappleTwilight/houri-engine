# LLM providers and model lists

English ｜ [中文](PROVIDERS_zh.md)

Translation is the one pipeline stage that leaves the device. It calls a cloud LLM with the line-numbered prompt and protocol ported from manga-image-translator. The engine ships a list of provider presets and can fetch each provider's live model list, so you bring your own key and pick a current model without editing code.

## One chat path for every provider

Every supported provider speaks the OpenAI chat-completions shape (`POST {base}` with `{ model, messages, temperature }` and `Authorization: Bearer <key>`). `LlmTranslator` is the only request builder; switching providers changes nothing but `apiBase`, `model`, and the key. Google Gemini is included by pointing chat at its OpenAI-compatible endpoint, so it needs no separate code either — only its model *listing* uses the native endpoint, because the compat path has no `/models`.

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
