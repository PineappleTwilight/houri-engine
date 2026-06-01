package li.joye.yakuyomi.engine

/**
 * 偵測輸出的文字框（原圖座標）。
 * 之後再加 mask / 分類（CLAUDE.md §13：「之後加 mask」）。
 */
data class Box(
    val x: Int,
    val y: Int,
    val w: Int,
    val h: Int,
    val score: Float = 0f,
)
