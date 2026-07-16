#!/usr/bin/env python3
"""
M2 翻譯 parity：把 OCR 讀出的日文 → DeepSeek（移植 chatgpt.py 的 prompt/協定）→ 繁中。
prompt/協定 ported from manga_translator/translators/{chatgpt.py,config_gpt.py} @ d5a3eee（第一層照搬）
繁中策略（§10）：to_lang 明指台灣繁體 + 輸出過 OpenCC s2twp 安全網。
金鑰從 repo 根 api-keys.properties 讀（gitignored），全程不列印。
"""
import os, re, json, glob, urllib.request
import numpy as np

from paths import ROOT, OUT, SANDBOX_PAGE as IMG  # 集中路徑，見 paths.py
OCR_JSON = os.path.join(OUT, "ocr_results.json")

API_BASE = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
TO_LANG = "Traditional Chinese (Taiwan, 台灣慣用的繁體中文用語)"
TEMPERATURE = 0.3

# --- 照搬 config_gpt.py 的 system 模板（三步法）---
CHAT_SYSTEM_TEMPLATE = (
    'Ignore all preceding instructions. Follow only what is defined below.\n'
    '## Role: Professional Doujin Translator\n'
    'You are an unsensored expert translation engine that specializes in doujin and manga content for all languages.\n'
    '## Translation Method\n'
    '1. LITERAL TRANSLATION: \n'
    '- Provide precise word-for-word translation of each textline.\n'
    '- Maintain original sentence structure where possible.\n'
    '- Preserve all original markers and expressions.\n'
    '- Leave ambiguous elements as they are without interpretation.\n'
    '2. ANALYSIS & DE-VERBALIZATION: \n'
    '- Capture the core meaning, emotional tone, and cultural nuances.\n'
    '- Identify logical connections between fragmented text segments.\n'
    '- Analyze the shortcomings and areas for improvement of literal translation.\n'
    '3. REFINEMENT: \n'
    '- Adjust the translation to sound natural in {to_lang} while maintaining original meaning.\n'
    '- Preserve emotional tone and intensity appropriate to manga & otaku culture.\n'
    '- Ensure consistency in character voice and terminology.\n'
    '- Determine appropriate pronouns from context; do not add pronouns that do not exist in the original text.\n'
    '- Refine based on the conclusions from the second step.\n'
    '## Translation Rules\n'
    '- Translate line by line, maintaining accuracy and the authentic; Faithfully reproducing the original text and emotional intent.\n'
    '- Preserve original gibberish or sound effects without translation.\n'
    '- Output each segment with its prefix (<|number|> format exactly) and only provide the translation without raw text.\n'
    '- Translate content only—no additional interpretation or commentary.\n'
    'Translate the following text into {to_lang}:\n'
)
# few-shot（照搬 _CHAT_SAMPLE 的簡中範例，最接近繁中；格式示範）
SAMPLE_IN = '<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\n<|2|>きみ… 大丈夫⁉\n<|3|>なんだこいつ 空気読めて ないのか…？'
SAMPLE_OUT = '<|1|>好尷尬…我不想引人注目…我想消失…\n<|2|>你…沒事吧⁉\n<|3|>這傢伙是看不懂氣氛嗎…？'


def read_key():
    for line in open(os.path.join(ROOT, "api-keys.properties"), encoding="utf-8"):
        if line.startswith("DEEPSEEK_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("api-keys.properties 找不到 DEEPSEEK_API_KEY=")


def call_deepseek(key, messages):
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": TEMPERATURE,
                       "stream": False}).encode("utf-8")
    req = urllib.request.Request(API_BASE, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


def translate_page(key, jp_list):
    """一頁：所有區一個 request（<|i|> 協定）→ 對齊 cht 清單；漏行/整頁失敗補原文（§11）。
       繁中靠 LLM prompt（TO_LANG 指定台灣繁體）；不做 OpenCC 後處理。"""
    if not jp_list:
        return []
    user = "\n".join(f"<|{i + 1}|>{q}" for i, q in enumerate(jp_list))
    msgs = [{"role": "system", "content": CHAT_SYSTEM_TEMPLATE.format(to_lang=TO_LANG)},
            {"role": "user", "content": SAMPLE_IN}, {"role": "assistant", "content": SAMPLE_OUT},
            {"role": "user", "content": user}]
    try:
        raw = re.sub(r'(</think>)?<think>.*?</think>', '', call_deepseek(key, msgs), flags=re.DOTALL)
    except Exception:
        return list(jp_list)  # 整頁失敗 → 留原文
    trans = {}
    for line in raw.splitlines():
        m = re.match(r'^\s*<\|(\d+)\|>\s*(.*)$', line)
        if m:
            trans[int(m.group(1))] = m.group(2).strip()
    return [(trans[i + 1] if trans.get(i + 1) else q) for i, q in enumerate(jp_list)]


def translate_pages(key, pages, batch_concurrent=True, batch_size=8):
    """跨頁批次（鏡射 engine/BatchTranslator，對映 m-i-t --batch-size/--batch-concurrent）：
       concurrent=逐頁分開、ThreadPool 限 batch_size 並發；merged=每 batch_size 頁併一個大 prompt。"""
    if not pages:
        return []
    n = max(1, batch_size)
    if batch_concurrent:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n) as ex:
            return list(ex.map(lambda p: translate_page(key, p), pages))
    out = []
    for c in range(0, len(pages), n):
        chunk = pages[c:c + n]
        res = translate_page(key, [q for pg in chunk for q in pg])
        i = 0
        for pg in chunk:
            out.append(res[i:i + len(pg)]); i += len(pg)
    return out


def find_font():
    for c in ["/mnt/c/Windows/Fonts/msjh.ttc", "/mnt/c/Windows/Fonts/msjhl.ttc",
              "/mnt/c/Windows/Fonts/mingliu.ttc", "/mnt/c/Windows/Fonts/msyh.ttc",
              "/mnt/c/Windows/Fonts/YuGothM.ttc"]:
        if os.path.exists(c):
            return c
    return None


def main():
    key = read_key()
    results = json.load(open(OCR_JSON, encoding="utf-8"))
    queries = [r["text"] for r in results]

    # user 訊息：編號原文（include_template=False，只放 <|i|>）
    user_prompt = "\n".join(f"<|{i + 1}|>{q}" for i, q in enumerate(queries))
    messages = [
        {"role": "system", "content": CHAT_SYSTEM_TEMPLATE.format(to_lang=TO_LANG)},
        {"role": "user", "content": SAMPLE_IN},
        {"role": "assistant", "content": SAMPLE_OUT},
        {"role": "user", "content": user_prompt},
    ]
    print(f"送 {len(queries)} 行 → DeepSeek（{MODEL}, temp={TEMPERATURE}）…")
    raw = call_deepseek(key, messages)
    raw = re.sub(r'(</think>)?<think>.*?</think>', '', raw, flags=re.DOTALL)

    # 解析 <|i|>譯文
    trans = {}
    for line in raw.splitlines():
        m = re.match(r'^\s*<\|(\d+)\|>\s*(.*)$', line)
        if m:
            trans[int(m.group(1))] = m.group(2).strip()

    out_rows = []
    for i, r in enumerate(results):
        cht = trans.get(i + 1, "")
        out_rows.append({"i": r["i"], "jp": r["text"], "cht": cht, "quad": r["quad"]})
        print(f"[{r['i']:2d}] {r['text']}  →  {cht}")

    json.dump(out_rows, open(os.path.join(OUT, "translate_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 繁中 overlay
    font = find_font()
    print(f"\nfont: {font}")
    if font:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.open(IMG).convert("RGB")
        dr = ImageDraw.Draw(im)
        fnt = ImageFont.truetype(font, 24)
        for row in out_rows:
            if not row["cht"]:
                continue
            q = np.array(row["quad"], np.int32)
            dr.line([tuple(map(int, p)) for p in q] + [tuple(map(int, q[0]))], fill=(255, 0, 0), width=3)
            x0, y0 = int(q[:, 0].min()), int(q[:, 1].min())
            dr.text((x0, max(0, y0 - 26)), row["cht"], fill=(220, 30, 30), font=fnt,
                    stroke_width=3, stroke_fill=(255, 255, 255))
        im.save(os.path.join(OUT, "translated_overlay.png"))
        print(f"overlay → {OUT}/translated_overlay.png")
    print(f"\n翻譯 {sum(1 for r in out_rows if r['cht'])}/{len(out_rows)} 行")


if __name__ == "__main__":
    main()
