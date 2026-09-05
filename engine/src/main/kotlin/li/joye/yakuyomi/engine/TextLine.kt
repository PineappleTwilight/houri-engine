package li.joye.yakuyomi.engine

/**
 * One text line: detection gives rotated quadrilateral (original image coordinates) + score; OCR then fills direction and recognized text.
 * Replaces old axis-aligned Box (OCR needs rotated box as input).
 * Hardened: validates quad has 4 points, score in 0..1, text sanitized.
 */
class TextLine(val quad: List<Pt>, val score: Float) {
    var direction: String = "h" // 'h' horizontal / 'v' vertical (determined by OCR sortPnts)
    var text: String = ""        // OCR source text (e.g., Japanese)
    var translatedText: String = "" // Translated text (e.g., target language)
}
