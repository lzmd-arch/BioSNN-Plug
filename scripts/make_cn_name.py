"""生成作者行的中文名矢量图（供 LaTeX 的 \\includegraphics 使用）。

**为什么是一张矢量 PDF 而不是直接写中文**：TeX Live 自带的 CJK 字体（Fandol）**没有「垚」**
（U+579A，GB2312 之外），本机与 CI 都会渲染成一个方框；而 Noto/思源等覆盖完整的字体又不在
tectonic 的 bundle 里。把三个字画成矢量图并入库，是唯一在**本机与 CI 中渲染一致**的做法。

重新生成需要一台装了 CJK 字体的机器：

    uv run --with pymupdf python scripts/make_cn_name.py

产物 `paper/figures/cn-name.pdf` 随仓库入库（与其它生成图一致）；CI 只读它，不需要字体。
"""

from __future__ import annotations

import sys
from pathlib import Path

TEXT = "沈锦垚"
FONT_CANDIDATES = (
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)
OUT = Path(__file__).resolve().parents[1] / "paper" / "figures" / "cn-name.pdf"


def main() -> int:
    try:
        import pymupdf  # PyMuPDF；不是项目依赖，按需 --with 安装
    except ImportError:
        print("需要 PyMuPDF：uv run --with pymupdf python scripts/make_cn_name.py", file=sys.stderr)
        return 2

    font_path = next((path for path in FONT_CANDIDATES if Path(path).exists()), None)
    if font_path is None:
        print("找不到可用的 CJK 字体文件", file=sys.stderr)
        return 2

    font = pymupdf.Font(fontfile=font_path)
    missing = [char for char in TEXT if not font.has_glyph(ord(char))]
    if missing:
        print(f"字体 {font_path} 缺少字形：{missing}", file=sys.stderr)
        return 2

    size = 100.0
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=200)
    page.insert_font(fontname="cn", fontfile=font_path)
    page.insert_text((40, 140), TEXT, fontname="cn", fontsize=size)

    # 按实际字形外框裁切，留一点边距，避免 \includegraphics 带空白
    spans = [
        span
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span["text"].strip()
    ]
    if not spans:
        print("没有抽到文本外框", file=sys.stderr)
        return 1
    box = pymupdf.Rect(spans[0]["bbox"])
    for span in spans[1:]:
        box |= pymupdf.Rect(span["bbox"])
    pad = size * 0.02
    page.set_cropbox(pymupdf.Rect(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # 只保留用到的字形：不 subset 的话 PyMuPDF 会把 18 MB 的 SimSun 整份嵌进去
    doc.subset_fonts()
    doc.save(OUT, clean=True, garbage=4, deflate=True)
    doc.close()
    print(f"写入 {OUT}（{OUT.stat().st_size} 字节，字体 {font_path}，外框 {box}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
