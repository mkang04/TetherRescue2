"""뷰어 화면 위 한글 글자·그래프.

MuJoCo 뷰어의 기본 글꼴은 한글을 못 그리므로, PIL로 한글 글꼴(맑은 고딕 등)을 써서 그림을 만든 뒤
viewer.set_images() 로 화면 위에 얹습니다. 그림은 불투명한 어두운 상자입니다.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_FILES = [  # 앞에서부터 찾음 (Windows → 우분투)
    (r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf"),
    ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
]

BG = (22, 24, 30)
BG_PANE = (34, 37, 45)
GRID = (70, 75, 88)
LABEL = (165, 175, 192)
WHITE = (240, 240, 240)
GOOD = (90, 220, 120)
BAD = (255, 90, 80)
WARN = (255, 190, 60)
INFO = (110, 190, 255)


@lru_cache(maxsize=32)
def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for regular, heavy in FONT_FILES:
        path = heavy if bold else regular
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)        # 한글 글꼴이 없으면 영어만 제대로 보임


def _text_w(f, s: str) -> int:
    return int(np.ceil(f.getlength(s))) if s else 0


def _fit(img: Image.Image, max_w: int, max_h: int) -> np.ndarray:
    """뷰어 창보다 크면 줄여서 numpy RGB로."""
    if img.width > max_w or img.height > max_h:
        s = min(max_w / img.width, max_h / img.height)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def info_box(rows: list[tuple], size: int, max_w: int, max_h: int) -> np.ndarray:
    """두 칸 표 (이름 | 값). rows: (이름, 값) 또는 (이름, 값, 색). 이름이 ""이면 빈 줄(구분)."""
    f, fb = font(size), font(size, bold=True)
    pad, gap, line = size // 2 + 4, size, int(size * 1.45)
    lw = max(_text_w(f, r[0]) for r in rows)
    vw = max(_text_w(fb, r[1]) for r in rows)
    heights = [line if r[0] or r[1] else line // 2 for r in rows]
    img = Image.new("RGB", (pad * 2 + lw + gap + vw, pad * 2 + sum(heights)), BG)
    d = ImageDraw.Draw(img)
    y = pad
    for r, h in zip(rows, heights):
        if r[0] or r[1]:
            d.text((pad, y), r[0], font=f, fill=LABEL)
            d.text((pad + lw + gap, y), r[1], font=fb, fill=r[2] if len(r) > 2 and r[2] else WHITE)
        y += h
    return _fit(img, max_w, max_h)


def banner(title: str, sub: str, color, size: int, max_w: int, max_h: int) -> np.ndarray:
    """가운데 위 알림 (큰 제목 + 작은 설명)."""
    ft, fs = font(int(size * 1.6), bold=True), font(size)
    pad = size
    subs = [s for s in sub.split("\n") if s]
    w = max([_text_w(ft, title)] + [_text_w(fs, s) for s in subs]) + pad * 2
    h = pad * 2 + int(size * 1.6 * 1.3) + int(size * 1.4) * len(subs)
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, h - 1], outline=color, width=3)
    d.text(((w - _text_w(ft, title)) // 2, pad), title, font=ft, fill=color)
    y = pad + int(size * 1.6 * 1.3)
    for s in subs:
        d.text(((w - _text_w(fs, s)) // 2, y), s, font=fs, fill=WHITE)
        y += int(size * 1.4)
    return _fit(img, max_w, max_h)


class Chart:
    """최근 몇 초 동안의 값을 그리는 실시간 꺾은선 그래프."""

    def __init__(self, title: str, lines: list[tuple[str, tuple[int, int, int]]], n_max: int):
        self.title, self.lines = title, lines
        self.buf = [deque(maxlen=n_max) for _ in range(len(lines) + 1)]   # [t, y0, y1, ...]

    def clear(self):
        for b in self.buf:
            b.clear()

    def push(self, t: float, *ys: float):
        self.buf[0].append(t)
        for b, y in zip(self.buf[1:], ys):
            b.append(y)

    def render(self, w: int, h: int, y_range, seconds: float, size: int) -> np.ndarray:
        f, fb = font(max(10, int(size * 0.85))), font(size, bold=True)
        img = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(img)
        pad = size // 2
        # 제목 + 범례
        d.text((pad, pad // 2), self.title, font=fb, fill=WHITE)
        x = pad + _text_w(fb, self.title) + size
        for name, rgb in self.lines:
            d.rectangle([x, pad // 2 + size // 3, x + size // 2, pad // 2 + size // 3 + size // 2], fill=rgb)
            x += size // 2 + 4
            d.text((x, pad // 2 + 2), name, font=f, fill=LABEL)
            x += _text_w(f, name) + size // 2 + 2
        # 그림 영역
        lo, hi = float(y_range[0]), float(y_range[1])
        tick_w = max(_text_w(f, f"{v:.1f}" if hi - lo < 20 else f"{v:.0f}") for v in (lo, hi)) + 6
        x0, x1 = pad + tick_w, w - pad
        y0, y1 = int(size * 1.9), h - int(size * 1.5)
        d.rectangle([x0, y0, x1, y1], fill=BG_PANE)
        for i in range(5):                     # 가로 눈금 4칸
            v = lo + (hi - lo) * i / 4
            yy = y1 - (y1 - y0) * i / 4
            d.line([x0, yy, x1, yy], fill=GRID)
            d.text((pad, yy - size * 0.55), f"{v:.1f}" if hi - lo < 20 else f"{v:.0f}", font=f, fill=LABEL)
        t = np.fromiter(self.buf[0], float)
        t_end = t[-1] if len(t) else 0.0
        d.text((x0, y1 + 3), f"{max(0.0, t_end - seconds):.0f}초", font=f, fill=LABEL)
        now = f"{t_end:.0f}초 (지금)"
        d.text((x1 - _text_w(f, now), y1 + 3), now, font=f, fill=LABEL)
        if len(t) >= 2:
            px = x0 + (t - (t_end - seconds)) / seconds * (x1 - x0)
            keep = px >= x0
            for (_, rgb), b in zip(self.lines, self.buf[1:]):
                yv = np.clip(np.fromiter(b, float), lo, hi)
                py = y1 - (yv - lo) / max(hi - lo, 1e-9) * (y1 - y0)
                pts = list(zip(px[keep].tolist(), py[keep].tolist()))
                if len(pts) >= 2:
                    d.line(pts, fill=rgb, width=2)
        return np.asarray(img, dtype=np.uint8)
