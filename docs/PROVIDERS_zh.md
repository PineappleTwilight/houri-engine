# LLM 供應商與模型清單

[English](PROVIDERS.md) ｜ 中文

翻譯是整條 pipeline 唯一會離開裝置的階段：用移植自 manga-image-translator 的「行號 prompt + 協定」呼叫雲端 LLM。引擎內建一份供應商預設表，並能撈取各家的即時模型清單——你自備金鑰（BYOK）、選一個當下的模型，全程不必改 code。

## 所有供應商共用一條聊天路徑

每個支援的供應商都講 OpenAI chat-completions 形狀（`POST {base}`，body `{ model, messages, temperature }`，`Authorization: Bearer <key>`）。`LlmTranslator` 是唯一的請求 builder；換供應商只動 `apiBase`、`model`、金鑰，其餘不變。Google Gemini 也納入了——聊天指向它的 OpenAI 相容端點即可，不必另寫 code；只有「列模型」走它的 native 端點，因為 compat 路徑沒有 `/models`。

這是刻意的。Yakuyomi 鎖定的供應商，正是 manga-image-translator 的 LLM translator——OpenAI、DeepSeek、Gemini、Groq、custom_openai、Sakura、Qwen——全部 OpenAI 相容。m-i-t 也支援的非 LLM 機器翻譯（DeepL、彩雲、有道、百度、Papago）沒有 chat/prompt 協定，不在範圍內。

## 供應商預設表

`LlmProviders.ALL`（引擎）放這些預設，每筆是 `{ id, displayName, chatUrl, modelsUrl, modelSource, defaultModel, baseEditable }`。

| 預設 | 主機 | 模型清單 | 備註 |
|---|---|---|---|
| DeepSeek（預設） | api.deepseek.com | `/v1/models` | 便宜、中文強 |
| OpenAI | api.openai.com | `/v1/models` | |
| Google Gemini | …/v1beta/openai/ | `/v1beta/models`（native） | 聊天走 OpenAI-compat 端點；列模型走 native |
| Groq | api.groq.com | `/v1/models` | 快、有免費額度 |
| 通義千問 Qwen | dashscope…/compatible-mode | `/v1/models` | 阿里 |
| OpenRouter | openrouter.ai | `/v1/models`（公開） | 一把 key 通吃上百模型、清單自動更新 |
| Sakura（自架） | 你的 base | 你的 base + `/v1/models` | 日→中專精 LLM |
| 自訂（OpenAI 相容） | 你的 base | 你的 base + `/v1/models` | LM Studio、SiliconFlow、任何相容端點 |

其中兩個是 `baseEditable`（Sakura、自訂）：使用者填 base URL，引擎據此推導聊天與列模型的 URL（`LlmProviders.chatUrlOf` / `modelsUrlOf`）。

## 撈取模型清單

模型迭代很快，所以引擎是「撈各家當下清單」而非寫死——這個點子借鏡 [nextai-translator](https://github.com/nextai-translator/nextai-translator)。`LlmModels.list(modelsUrl, source, apiKey)` 回傳 `List<ModelInfo>`；reader 設定頁在「抓取模型」鈕後面呼叫它、把結果做成挑選清單。

依 `ModelSource` 分兩種端點形狀：

- **`OPENAI`** —— `GET {modelsUrl}` 帶 `Authorization: Bearer`，解析 `data[].id`。涵蓋所有 OpenAI 相容供應商。
- **`GEMINI`** —— `GET {modelsUrl}?key=`，解析 `models[]`，留下 `supportedGenerationMethods` 含 `generateContent` 的、去掉 `models/` 前綴。

其餘情況——沒 key、端點不支援列模型、網路錯——一律回空清單。撈取從不拋例外、也不擋設定；回空只代表「請手動輸入 model id」。自動清單是加分，不是前置條件。

## 自備金鑰（每家一格）

金鑰按供應商分開存，所以 DeepSeek↔OpenRouter 來回切換，各家的 key 都保留。reader 用 Android Keystore 加密儲存，只把「目前供應商」的 key 當建構參數傳給引擎；引擎本身不持有任何金鑰。

## 新增一個供應商

因為每家都 OpenAI 相容，新增通常就是在 `LlmProviders.ALL` 加一筆：

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

不必動請求 builder。若某供應商不是 OpenAI 相容（聊天形狀不同，例如 Anthropic 的 Messages API），才需要在引擎加新 builder——目前範圍內的供應商都不需要。

## 為何選這些、為何自動撈清單

供應商集合 = 「manga-image-translator 鎖定的」∩「對日→中漫畫有用的」，再加 OpenRouter 當聚合器。自動清單鏡射 nextai-translator 的 `listModels`，但用得更徹底：nextai 對好幾家（Claude、Kimi、智譜、MiniMax）寫死靜態清單，而 Yakuyomi 對全部都撈 `/v1/models`、撈不到就退回手填——所以沒有任何「會隨模型變動而要維護的靜態清單」。
