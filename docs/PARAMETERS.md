# Parameters

English ｜ [中文](PARAMETERS_zh.md)

This is the reference for every tunable parameter: what it controls, its range, and what happens when you push it to either extreme.

The reader app exposes these on its Translation settings page, grouped by stage. Common settings show by default; the rest appear behind the **Show advanced** switch. The sandbox app exposes the same knobs for testing on a device.

Numeric settings are typed in and clamped to the range shown here (out-of-range input is pulled back to the nearest bound, not rejected). The performance dropdown only lists values up to your device's core count.

---

## Translation

### Enable translation
Switch, default off. Translates each chapter as it finishes downloading. Detection, OCR, and text removal run on the device; translation calls the LLM, so it needs a network connection while it runs.

### API key
Your LLM key (bring your own key), kept per provider so switching providers doesn't lose a key. Stored encrypted in the Android Keystore, never bundled.

### Target language
The language the LLM translates into. Default is Taiwan-style Traditional Chinese. The default carries a Japanese-to-Chinese few-shot example; choosing another target drops the example so it doesn't pull the output toward Chinese.

### Source language
A hint passed to the prompt. The actual source is decided by the OCR model, so this is just a label. Leave it on auto-detect unless the OCR output is being mislabelled.

### Provider / model / API base
The provider is a preset picker: manga-image-translator's LLM set (OpenAI, DeepSeek, Gemini, Groq, Qwen, Sakura, custom) plus OpenRouter, all OpenAI-compatible. DeepSeek by default. The model can be typed, or fetched live from the provider with **fetch models**; leave it blank to use the provider's default. API base appears only for the self-hosted/custom presets (Sakura, Custom); the others use their built-in endpoint. See [PROVIDERS.md](PROVIDERS.md) for the full list and how the model fetch works.

---

## Text removal

### Method
Dropdown, default **AI removal**. Two modes, both of which flat-fill speech bubbles cleanly. They differ only in how text drawn over artwork is erased:

- **Fast removal (BoxFill)** — flat-fills everything, sampling the nearest surrounding colour. Fastest (about half a second a page), but rough: text over artwork becomes a flat colour block, so it only looks good on pages that are almost all clean white bubbles.
- **AI removal (AOT-GAN)** — flat-fills bubbles, then runs one whole-image AOT-GAN pass to reconstruct the artwork behind on-art text. High quality; the default.

Why bubbles are always flat-filled: running a clean white bubble through the inpainter is pure downside (a faint halo plus the cost), so only text drawn over artwork is reconstructed.

### Mask padding (bboxPad)
Default 16, in pixels. Range 0–64. How far the removal area extends past the detected text box, to cover furigana and strokes that sit right on the edge.

Lower (toward 0): kana hugging the box edge get missed and leave a residue. Higher (toward 64): removal reaches into neighbouring artwork.

---

## Typesetting

### Orientation
Dropdown, default auto. Auto follows the source text (CJK goes vertical). Force vertical or horizontal if a title or font confuses the detector.

### Text colour (colorMode)
Dropdown, default auto. Auto picks black or white per region from the cleaned background (black on light, white on dark). Mono draws black text with a white outline everywhere; simpler, and it leans on the outline to stay readable on dark backgrounds.

### Text outline (fontBorder)
Switch, default on. Outlines the translated text. On busy or dark backgrounds the outline is what keeps it readable; off gives plain text that can disappear into the art.

### Max font size (fontSizeMax)
Default 60, in pixels. Range 20–120. The largest size the layout will use. Lower: text in big bubbles looks small and lost. Higher: short lines blow up unnaturally large.

### Min font size (fontSizeMin)
Default 9, in pixels. Range 6–40. The smallest size; below this the layout would rather overflow than shrink further. Lower (toward 6): text fits but is hard to read. Higher (toward 20): small bubbles can't fit their text and it overflows.

### On-art outline width (artStrokeRatio)
Default 0.16. Range 0–0.5. Outline width for text over artwork, as a fraction of the font size (a bit thicker than the 0.10 used in plain bubbles, because busy backgrounds need it). Lower (toward 0): the outline is too thin to read against artwork. Higher (toward 0.5): the outline eats into the glyph and looks like a sticker.

### Vertical column trim (colTrim)
Default 3. Range 0–10. How many fewer characters to pack into each vertical column. Trimming columns makes them shorter, which adds more columns and shrinks the font to fit. Lower (0): columns pack full and can overflow the box. Higher: columns get very short and the font is forced small.

### Horizontal row trim (rowTrim)
Default 3. Range 0–10. The horizontal counterpart of column trim — fewer characters per row, which adds rows and shrinks the font.

### Font scale (fontScale)
Default 0.85. Range 0.3–1.5. An overall multiplier applied after the layout picks a size. Below 1 makes text smaller and leaves a margin inside the box; the default leaves breathing room. Lower (toward 0.5): text is generally small with lots of empty space. Above 1: text runs large and tends to touch or overflow the box.

---

## Performance (device-dependent)

This lists only values up to your CPU's core count. Auto adapts to the device, and is the right choice unless you are benchmarking.

### OCR concurrency
Dropdown, default auto (= core count). How many text lines OCR recognizes at once, each on a single thread. OCR strips are small and don't saturate the per-inference threads, so filling the cores with separate lines is faster — a clear win when a page has many lines. With the int8 OCR model the whole OCR pass now runs in roughly 0.25–1.8 s a page. Above the core count there's nothing left to fill, so the list stops there.

---

## Recognition (advanced)

### Removal mask threshold (segThreshold)
Default 0.12. Range 0–1. The cutoff on the detector's per-pixel stroke probability for deciding what counts as text to remove. Lower: more pixels count, so screentone and noise get pulled into the mask and removal spreads into non-text. Higher: only the boldest strokes count, so the faint kana printed next to kanji get left behind as a residue. The default is low specifically to catch those faint kana.

### OCR confidence threshold (minProb)
Default 0.5. Range 0–1. A line whose average OCR confidence falls below this is dropped (not translated, not removed). Lower (toward 0): noise and SFX fragments get recognized as text and translated into garbage. Higher (toward 1): stylized or dramatic lettering, which the detector tends to merge into one wide box and OCR reads with low confidence, gets dropped and goes untranslated.

### Skip SFX (ignoreSfx)
Switch, default off. Skips *translating* coloured / non-bubble sound-effect (SFX) text — it stays on the page in the original, not erased. Some readers prefer decorative SFX left as-is. This is skip-translation, not active removal; under the hood the engine reads a mid-range `ignoreBubble` threshold, not exposed as a number.
