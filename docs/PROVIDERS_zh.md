# LLM 供應商與模型清單

[English](PROVIDERS.md) ｜ 中文

翻譯是整條 pipeline 唯一會離開裝置的階段：用移植自 manga-image-translator 的「行號 prompt + 協定」呼叫雲端 LLM。引擎內建一份供應商預設表，並能撈取各家的即時模型清單——你自備金鑰（BYOK）、選一個當下的模型，全程不必改 code。

## 所有供應商共用一條聊天路徑

每個支援的供應商都講 OpenAI chat-completions 形狀（`POST {base}`，body `{ model, messages, stream }`，`Authorization: Bearer <key>`）。`LlmTranslator` 是唯一的請求 builder；換供應商只動 `apiBase`、`model`、金鑰，其餘不變。這三個欄位以外的（`temperature`、思考開關）走一層相容映射——因為「OpenAI 相容」不等於「吃一樣的欄位」（見下）。Google Gemini 也納入了——聊天指向它的 OpenAI 相容端點即可，不必另寫 code；只有「列模型」走它的 native 端點，因為 compat 路徑沒有 `/models`。

這是刻意的。Yakuyomi 鎖定的供應商，正是 manga-image-translator 的 LLM translator——OpenAI、DeepSeek、Gemini、Groq、custom_openai、Sakura、Qwen——全部 OpenAI 相容。m-i-t 也支援的非 LLM 機器翻譯（DeepL、彩雲、有道、百度、Papago）沒有 chat/prompt 協定，不在範圍內。

## 供應商預設表

`LlmProviders.ALL`（引擎）放這些預設，每筆是 `{ id, displayName, chatUrl, modelsUrl, modelSource, defaultModel, baseEditable }`。

| 預設 | 主機 | 模型清單 | 備註 |
|---|---|---|---|
| DeepSeek（預設） | api.deepseek.com | `/v1/models` | 便宜、中文強 |
| OpenAI | api.openai.com | `/v1/models` | |
| Google Gemini | …/v1beta/openai/ | `/v1beta/models`（native） | 聊天走 OpenAI-compat 端點；列模型走 native |
| Groq | api.groq.com | `/v1/models` | 快、有免費額度 |
| 通義千問 Qwen | dashscope-intl…/compatible-mode | `/v1/models` | 阿里 Model Studio 國際版；大陸 key 請用「自訂」填 dashscope.aliyuncs.com |
| OpenRouter | openrouter.ai | `/v1/models`（公開） | 一把 key 通吃上百模型、清單自動更新 |
| Sakura（自架） | 你的 base | 你的 base + `/v1/models` | 日→中專精 LLM |
| 自訂（OpenAI 相容） | 你的 base | 你的 base + `/v1/models` | LM Studio、SiliconFlow、任何相容端點 |

其中兩個是 `baseEditable`（Sakura、自訂）：使用者填 base URL，引擎據此推導聊天與列模型的 URL（`LlmProviders.chatUrlOf` / `modelsUrlOf`）。

## 各家 request 參數對照

各家只在 `model` / `messages` / `stream` 上一致，其餘全是分歧——這家必填的欄位，換一家就是 400，**同一家不同世代的模型規則還會變**。`LlmProviders.PARAM_RULES` 就是一張 `ParamRule` 資料表（`provider → [規則]`，由上而下第一個命中 `modelPattern` 者勝、`null` pattern 放最後當 fallback）；`LlmProviders.requestParams(provider, model, thinking, temperature)` 是純函式、回傳這次要附加的 body 欄位，`LlmTranslator` 只負責塞進 JSON。要加一家供應商或一個特例模型＝加一列資料，不動邏輯。

| 供應商／模型 | 思考關（預設） | 思考開 | `temperature` | 備註 |
|---|---|---|---|---|
| DeepSeek | `thinking: {"type":"disabled"}` | －（該家預設就思考） | 送 | v4-flash / v4-pro 都預設思考；思考模式下取樣參數「收下但無效」（[文件](https://api-docs.deepseek.com/guides/thinking_mode/)） |
| OpenAI · o 系列（`o1`、`o3`、`o4-mini`、`codex-mini`） | `reasoning_effort: "low"` | － | **不送** | 推理模型拒收 `temperature`、`top_p`、`max_tokens`…；長度上限欄位是 `max_completion_tokens`；`o1-mini` 沒有 `reasoning_effort`；關不掉、只能降檔（[文件](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning)） |
| OpenAI · GPT-5 系列 | `"none"`（5.1 以後）、`"minimal"`（初代 gpt-5/-mini/-nano）、`"low"`（gpt-5-codex） | － | **不送** | `gpt-5-pro` 只吃 `high`，所以什麼都不送 |
| OpenAI · `gpt-5*-chat`、`gpt-4o`、`gpt-4.1`… | － | － | 送 | 非推理模型：送 `reasoning_effort` 反而 400 |
| Gemini（OpenAI 相容） | `reasoning_effort: "none"`（2.5 系）、`"minimal"`（3.x，含預設的 `gemini-3.6-flash`） | － | 送 | `none` 只有 2.5 系吃；3.x 只能最小化、關不掉（[文件](https://ai.google.dev/gemini-api/docs/openai)） |
| Groq | `reasoning_effort: "none"`（Qwen 系）、`"low"`（GPT-OSS，含預設的 `openai/gpt-oss-120b`） | － | 送 | GPT-OSS 只吃 low/medium/high，`low` 是最接近關掉的檔；Llama 系收到這欄位直接 400（[文件](https://console.groq.com/docs/reasoning)） |
| Qwen（DashScope compat） | `enable_thinking: false` | `enable_thinking: false` | 送 | DashScope 對「思考模型 + 非串流」除非顯式 false 否則直接 400，而引擎從不串流（[文件](https://www.alibabacloud.com/help/en/model-studio/deep-thinking)） |
| OpenRouter | `reasoning: {"effort":"none"}` | － | 送 | `effort: none` ＝完全關閉推理；未支援的參數預設會被忽略，所以任何模型都安全（[文件](https://openrouter.ai/docs/use-cases/reasoning-tokens)） |
| Sakura、自訂、未知 | － | － | 送 | 自架端點不加任何額外欄位：未知欄位可能就是 400 |

「－」＝不送欄位、用該家自己的預設。`ParamRule` 另外帶 `maxTokensField`（`max_tokens` vs `max_completion_tokens`）；引擎目前不設輸出上限、沒送這欄，映射先備著。`LlmParamsTest` 把這張表釘住（o 系列不帶 `temperature`、DeepSeek 送關思考、自架什麼都不加…）。

## 已退役的 model id

供應商會下架模型名稱，存在設定裡的舊 id 就此永久失敗。`LlmProviders.RETIRED_MODELS`（依 provider id）在送出請求前就地換名，使用者不必去改一個自己沒選過的設定：

| 供應商 | 退役 id | 換成 | 原因 |
|---|---|---|---|
| DeepSeek | `deepseek-chat`、`deepseek-reasoner` | `deepseek-v4-flash` | 2026-07-24 移除且不留相容 shim——舊名稱一律 400「Model Not Exist」 |
| Gemini | `gemini-2.0-flash`、`gemini-2.0-flash-001` | `gemini-3.6-flash` | 2026-06-01 停役（[deprecations](https://ai.google.dev/gemini-api/docs/deprecations)） |
| Gemini | `gemini-2.0-flash-lite`、`gemini-2.0-flash-lite-001` | `gemini-3.5-flash-lite` | 同批停役；維持 lite 級距，價位／延遲不變 |

這張表**只收真的已經死掉的 id**。只是被標 deprecated、但還在服務的（Groq 的 `llama-3.3-70b-versatile` 可用到 2026-08-16 停役日）不動它：偷偷換掉使用者選的可用模型，比讓他看到停役公告更糟。這種情況只動預設值（Groq 預設已改成 Groq 自己點名的 production 替代 `openai/gpt-oss-120b`；另一個建議 `qwen/qwen3.6-27b` 只是 preview）。`LlmParamsTest` 會檢查「沒有任何預設模型本身是退役 id」。

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
