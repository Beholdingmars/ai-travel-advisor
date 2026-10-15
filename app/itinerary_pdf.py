"""Draw the itinerary as a vector PDF (real text, embedded fonts) with the Meridian look."""

import io
import math
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from app.travel_advisor import Itinerary


# ---------- Fonts (the same ones as the website) ----------

FONT_DIR = Path(__file__).resolve().parent / "fonts"
pdfmetrics.registerFont(TTFont("Serif", FONT_DIR / "InstrumentSerif-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Sans", FONT_DIR / "Inter-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Medium", FONT_DIR / "Inter-Medium.ttf"))
pdfmetrics.registerFont(TTFont("Sans-SemiBold", FONT_DIR / "Inter-SemiBold.ttf"))


# ---------- Design tokens (same palette as the website) ----------

NAVY = HexColor("#0A1019")
DEEP = HexColor("#142236")
WHITE = HexColor("#FAFAF8")
TEXT_2 = HexColor("#C9CFD8")
SLATE = HexColor("#8994A4")
ICE = HexColor("#A8D9E8")
CHAMPAGNE = HexColor("#C7B99B")
PANEL = HexColor("#121D2B")
EDGE = HexColor("#26384A")
RULE = HexColor("#1B2836")
TRACK = HexColor("#1A2836")

PAGE_W, PAGE_H = A4          # 595 x 842 points
MARGIN = 51                  # 18 mm
CONTENT_W = PAGE_W - 2 * MARGIN
TOP = PAGE_H - MARGIN
BOTTOM = 72                  # space kept free for the footer


def style(font, size, leading, color):
    return ParagraphStyle(name=f"{font}{size}", fontName=font, fontSize=size, leading=leading, textColor=color)


TITLE = style("Serif", 36, 40, WHITE)
SUMMARY = style("Sans", 11, 17, TEXT_2)
DAY_TITLE = style("Serif", 19, 23, WHITE)
BODY = style("Sans", 9.5, 14.5, TEXT_2)
LIST = style("Sans", 9.5, 14, TEXT_2)
STOP_TITLE = style("Sans", 7.5, 10, TEXT_2)


def money(amount, currency):
    symbols = {"EUR": "€", "USD": "$", "GBP": "£", "JPY": "¥"}
    value = f"{round(amount):,}"
    return f"{symbols[currency]}{value}" if currency in symbols else f"{value} {currency}"


class ItineraryPDF:
    """Draws the pages one by one, measuring every block so nothing overflows."""

    def __init__(self, itinerary: Itinerary, total_pages: int | None = None):
        self.it = itinerary
        self.total_pages = total_pages
        self.buffer = io.BytesIO()
        self.c = canvas.Canvas(self.buffer, pagesize=A4)
        self.c.setTitle(f"Meridian – {itinerary.destination}")
        self.c.setAuthor("Meridian travel advisor")
        self.page = 0
        b = itinerary.budget
        self.parts = [("Accommodation", b.accommodation), ("Food", b.food),
                      ("Transport", b.transport), ("Activities", b.activities)]
        self.total = sum(v for _, v in self.parts)

    # ---------- small drawing helpers ----------

    def label(self, x, y, text, color=CHAMPAGNE, size=7.5, font="Sans-SemiBold", right=False):
        """Small uppercase label with letter spacing."""
        text = text.upper()
        spacing = size * 0.13
        if right:
            width = pdfmetrics.stringWidth(text, font, size) + spacing * (len(text) - 1)
            x -= width
        self.c.setFillColor(color)
        self.c.setFont(font, size)
        self.c.drawString(x, y, text, charSpace=spacing)

    def paragraph(self, text, st, width):
        p = Paragraph(escape(text or ""), st)
        _, h = p.wrap(width, PAGE_H)
        return p, h

    def panel(self, x, y_top, w, h):
        self.c.setFillColor(PANEL)
        self.c.setStrokeColor(EDGE)
        self.c.setLineWidth(0.8)
        self.c.roundRect(x, y_top - h, w, h, 14, stroke=1, fill=1)

    def gradient_bar(self, x, y, w, h):
        if w <= 0:
            return
        self.c.saveState()
        path = self.c.beginPath()
        path.roundRect(x, y, w, h, h / 2)
        self.c.clipPath(path, stroke=0, fill=0)
        self.c.linearGradient(x, y, x + max(w, 1), y, (ICE, CHAMPAGNE), extend=False)
        self.c.restoreState()

    def icon(self, kind, cx, cy, color=ICE):
        """Tiny line icons drawn as vectors."""
        c = self.c
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(0.9)
        c.setLineCap(1)
        if kind == "morning":                      # sun rising over the horizon
            c.line(cx - 5, cy - 2.5, cx + 5, cy - 2.5)
            path = c.beginPath()
            path.arc(cx - 3, cy - 5.5, cx + 3, cy + 0.5, 0, 180)
            c.drawPath(path, stroke=1, fill=0)
            for a in (45, 90, 135):
                r = math.radians(a)
                c.line(cx + 4.5 * math.cos(r), cy - 2.5 + 4.5 * math.sin(r), cx + 6 * math.cos(r), cy - 2.5 + 6 * math.sin(r))
        elif kind == "afternoon":                  # full sun
            c.circle(cx, cy, 2.6, stroke=1, fill=0)
            for i in range(8):
                r = math.radians(i * 45)
                c.line(cx + 4 * math.cos(r), cy + 4 * math.sin(r), cx + 5.5 * math.cos(r), cy + 5.5 * math.sin(r))
        elif kind == "evening":                    # crescent moon
            c.saveState()
            path = c.beginPath()
            path.circle(cx, cy, 4.5)
            c.clipPath(path, stroke=0, fill=0)
            c.setFillColor(color)
            c.circle(cx, cy, 4.5, stroke=0, fill=1)
            c.setFillColor(PANEL)
            c.circle(cx + 2.4, cy + 1.8, 4.1, stroke=0, fill=1)
            c.restoreState()
        elif kind == "pin":
            c.circle(cx, cy + 1, 2.4, stroke=1, fill=0)
            c.line(cx - 2.1, cy - 0.2, cx, cy - 3.6)
            c.line(cx + 2.1, cy - 0.2, cx, cy - 3.6)
        elif kind == "mark":                       # compass rose of the brand
            c.circle(cx, cy, 6.5, stroke=1, fill=0)
            path = c.beginPath()
            path.moveTo(cx, cy + 4.6); path.lineTo(cx + 1.6, cy); path.lineTo(cx, cy - 4.6); path.lineTo(cx - 1.6, cy); path.close()
            c.drawPath(path, stroke=1, fill=0)

    # ---------- page frame ----------

    def new_page(self, section=None):
        if self.page:
            self.c.showPage()
        self.page += 1
        c = self.c
        c.saveState()
        c.linearGradient(0, PAGE_H, PAGE_W * 0.55, 0, (DEEP, NAVY), extend=True)
        c.restoreState()

        # header
        self.icon("mark", MARGIN + 6.5, TOP - 4)
        c.setFillColor(WHITE)
        c.setFont("Serif", 15)
        c.drawString(MARGIN + 19, TOP - 9, "Meridian")
        self.label(PAGE_W - MARGIN, TOP - 7, "Travel itinerary", color=SLATE, font="Sans", size=7, right=True)

        # footer
        c.setStrokeColor(RULE)
        c.setLineWidth(0.7)
        c.line(MARGIN, 44, PAGE_W - MARGIN, 44)
        c.setFillColor(SLATE)
        c.setFont("Sans", 7)
        left = f"{self.it.destination} · {self.it.trip_title}"
        while pdfmetrics.stringWidth(left, "Sans", 7) > CONTENT_W - 80:
            left = left[:-2].rstrip(" ·…") + "…"
        c.drawString(MARGIN, 32, left)
        total = f" / {self.total_pages}" if self.total_pages else ""
        c.drawRightString(PAGE_W - MARGIN, 32, f"{self.page}{total}")

        y = TOP - 48
        if section:
            self.label(MARGIN, y, section)
            y -= 26
        return y

    # ---------- cover ----------

    def cover(self):
        c = self.c
        y = self.new_page() - 34

        self.icon("pin", MARGIN + 3, y + 2.5, CHAMPAGNE)
        self.label(MARGIN + 11, y, self.it.destination, size=8.5, font="Sans-Medium")
        y -= 16

        p, h = self.paragraph(self.it.trip_title, TITLE, CONTENT_W * 0.92)
        p.drawOn(c, MARGIN, y - h)
        y -= h + 14

        p, h = self.paragraph(self.it.summary, SUMMARY, CONTENT_W * 0.85)
        p.drawOn(c, MARGIN, y - h)
        y -= h + 24

        # facts
        currency = self.it.budget.currency
        facts = [(str(len(self.it.days)), "Days" if len(self.it.days) != 1 else "Day"),
                 (money(self.total, currency), "Budget per person")]
        x = MARGIN
        for value, name in facts:
            w = max(96, pdfmetrics.stringWidth(value, "Serif", 22) + 36, pdfmetrics.stringWidth(name.upper(), "Sans", 7) * 1.15 + 36)
            self.panel(x, y, w, 56)
            c.setFillColor(WHITE)
            c.setFont("Serif", 22)
            c.drawString(x + 16, y - 27, value)
            self.label(x + 16, y - 44, name, color=SLATE, font="Sans", size=7)
            x += w + 10
        y -= 56 + 40

        y = self.journey_line(y)
        self.cover_panels(y)

    def journey_line(self, y):
        """The route motif: one waypoint per day, labels underneath."""
        c = self.c
        days = self.it.days
        n = len(days)
        inset = 48 if n <= 5 else 20                # keep the day titles inside the margins
        left, right = MARGIN + inset, PAGE_W - MARGIN - inset
        base = y - 14
        pts = []
        for i in range(n):
            px = (left + right) / 2 if n == 1 else left + i * (right - left) / (n - 1)
            py = base + math.sin(i * 1.3 + 0.6) * 8
            pts.append((px, py))

        c.setStrokeColor(RULE)
        c.setLineWidth(0.8)
        c.setDash(1.5, 4)
        c.line(MARGIN, base, PAGE_W - MARGIN, base)
        c.setDash()

        # the line itself, coloured from ice to champagne
        if n > 1:
            for i in range(1, n):
                (x0, y0), (x1, y1) = pts[i - 1], pts[i]
                t = (i - 0.5) / (n - 1)
                col = HexColor("#%02X%02X%02X" % tuple(int(a + (b - a) * t) for a, b in ((168, 199), (217, 185), (232, 155))))
                c.setStrokeColor(col)
                c.setLineWidth(1.6)
                mid = (x0 + x1) / 2
                path = c.beginPath()
                path.moveTo(x0, y0)
                path.curveTo(mid, y0, mid, y1, x1, y1)
                c.drawPath(path, stroke=1, fill=0)

        show_titles = n <= 5
        title_w = min(96, (right - left) / max(1, n - 1) - 10) if n > 1 else 160
        lowest = base
        for i, (px, py) in enumerate(pts):
            c.setFillColor(NAVY)
            c.setStrokeColor(ICE)
            c.setLineWidth(1.2)
            c.circle(px, py, 3.6, stroke=1, fill=1)
            name = f"DAY {days[i].day_number or i + 1}"
            width = pdfmetrics.stringWidth(name, "Sans-SemiBold", 7) + 0.9 * (len(name) - 1)
            self.label(px - width / 2, py - 16, name, color=ICE, size=7)
            bottom = py - 16
            if show_titles:
                st = ParagraphStyle("rt", parent=STOP_TITLE, alignment=1)
                p = Paragraph(escape(days[i].title), st)
                _, h = p.wrap(title_w, 200)
                h = min(h, 30)                          # at most three short lines
                p.drawOn(c, px - title_w / 2, py - 22 - h)
                bottom = py - 22 - h
            lowest = min(lowest, bottom)
        return lowest - 30

    def cover_panels(self, y):
        """Budget and 'At a glance' side by side, pinned to the bottom of the cover."""
        c = self.c
        gap = 14
        w = (CONTENT_W - gap) / 2
        currency = self.it.budget.currency

        # measure "At a glance"
        rows = []
        for i, d in enumerate(self.it.days):
            p, h = self.paragraph(d.title, style("Sans", 9, 12.5, TEXT_2), w - 32 - 46)
            rows.append((f"DAY {d.day_number or i + 1}", p, h))
        glance_h = 50 + sum(h + 12 for _, _, h in rows) + 6
        budget_h = 50 + len(self.parts) * 30 + 70
        h = max(budget_h, glance_h)

        top = BOTTOM + h
        if top > y:                                 # not enough room: drop the glance list
            rows, h, top = [], budget_h, BOTTOM + budget_h
        two_cols = bool(rows)
        bw = w if two_cols else CONTENT_W

        # budget
        self.panel(MARGIN, top, bw, h)
        c.setFillColor(WHITE)
        c.setFont("Serif", 17)
        c.drawString(MARGIN + 16, top - 30, "Budget")
        biggest = max([v for _, v in self.parts] + [1])
        ry = top - 54
        for name, value in self.parts:
            c.setFont("Sans", 9)
            c.setFillColor(TEXT_2)
            c.drawString(MARGIN + 16, ry, name)
            c.setFillColor(WHITE)
            c.drawRightString(MARGIN + bw - 16, ry, money(value, currency))
            c.setFillColor(TRACK)
            c.roundRect(MARGIN + 16, ry - 10, bw - 32, 2.6, 1.3, stroke=0, fill=1)
            self.gradient_bar(MARGIN + 16, ry - 10, (bw - 32) * value / biggest, 2.6)
            ry -= 30
        c.setStrokeColor(RULE)
        c.line(MARGIN + 16, ry + 8, MARGIN + bw - 16, ry + 8)
        c.setFont("Sans", 8.5)
        c.setFillColor(SLATE)
        c.drawString(MARGIN + 16, ry - 16, "Total per person")
        c.setFont("Serif", 22)
        c.setFillColor(WHITE)
        c.drawRightString(MARGIN + bw - 16, ry - 18, money(self.total, currency))
        c.setFont("Sans", 7)
        c.setFillColor(SLATE)
        c.drawString(MARGIN + 16, ry - 38, "Estimates from the advisor. Check current prices before booking.")

        # at a glance
        if two_cols:
            x = MARGIN + w + gap
            self.panel(x, top, w, h)
            c.setFillColor(WHITE)
            c.setFont("Serif", 17)
            c.drawString(x + 16, top - 30, "At a glance")
            ry = top - 50
            for i, (name, p, ph) in enumerate(rows):
                self.label(x + 16, ry - 9, name, color=ICE, size=7)
                p.drawOn(c, x + 16 + 46, ry - ph)
                ry -= ph + 12
                if i < len(rows) - 1:
                    c.setStrokeColor(RULE)
                    c.line(x + 16, ry + 6, x + w - 16, ry + 6)

    # ---------- days ----------

    def day_block(self, day, index, draw_at=None):
        """Measure a day (and draw it when draw_at is given). Returns its height."""
        c = self.c
        text_x = MARGIN + 78
        text_w = PAGE_W - MARGIN - text_x - 28
        title, th = self.paragraph(day.title, DAY_TITLE, PAGE_W - MARGIN - text_x)
        slots = []
        for name, kind, body in (("Morning", "morning", day.morning), ("Afternoon", "afternoon", day.afternoon), ("Evening", "evening", day.evening)):
            p, h = self.paragraph(body, BODY, text_w)
            slots.append((name, kind, p, h))
        height = th + 12 + sum(14 + h + 12 for *_, h in slots)
        if draw_at is None:
            return height

        y = draw_at
        c.setFillColor(SLATE)
        c.setFont("Serif", 12)
        c.drawString(MARGIN, y - 10, "Day")
        c.setFillColor(ICE)
        c.setFont("Serif", 40)
        c.drawString(MARGIN, y - 46, str(day.day_number or index + 1))

        title.drawOn(c, text_x, y - th)
        y -= th + 12
        for name, kind, p, h in slots:
            c.setFillColor(PANEL)
            c.setStrokeColor(EDGE)
            c.setLineWidth(0.8)
            c.circle(text_x + 10, y - 9, 10, stroke=1, fill=1)
            self.icon(kind, text_x + 10, y - 9)
            self.label(text_x + 28, y - 7, name, size=7)
            p.drawOn(c, text_x + 28, y - 14 - h)
            y -= 14 + h + 12
        return height

    def days(self):
        y = self.new_page("Day by day")
        for i, day in enumerate(self.it.days):
            h = self.day_block(day, i)
            if y - h < BOTTOM:
                y = self.new_page("Day by day")
            elif i > 0:
                self.c.setStrokeColor(RULE)
                self.c.setLineWidth(0.7)
                self.c.line(MARGIN, y + 12, PAGE_W - MARGIN, y + 12)
            self.day_block(day, i, draw_at=y)
            y -= h + 28
        return y

    # ---------- tips ----------

    def notes(self, y):
        gap = 14
        w = (CONTENT_W - gap) / 2
        columns = [("Good to know", self.it.tips), ("Avoid crowds, save money", self.it.savings_tips)]
        measured = []
        for title, items in columns:
            paras = [self.paragraph(t, LIST, w - 32 - 12) for t in items]
            measured.append((title, paras, 50 + sum(h + 8 for _, h in paras) + 8))
        h = max(m[2] for m in measured)
        if y - 10 - h < BOTTOM:
            y = self.new_page("Before you go")
        else:
            y -= 10
        for col, (title, paras, _) in enumerate(measured):
            x = MARGIN + col * (w + gap)
            self.panel(x, y, w, h)
            self.c.setFillColor(WHITE)
            self.c.setFont("Serif", 16)
            self.c.drawString(x + 16, y - 30, title)
            ry = y - 48
            for p, ph in paras:
                self.c.setStrokeColor(ICE)
                self.c.setLineWidth(0.8)
                self.c.circle(x + 19, ry - 6.5, 2.2, stroke=1, fill=0)
                p.drawOn(self.c, x + 28, ry - ph)
                ry -= ph + 8

    def build(self) -> bytes:
        self.cover()
        y = self.days()
        self.notes(y)
        self.c.save()
        return self.buffer.getvalue()


def itinerary_pdf(itinerary: Itinerary) -> bytes:
    """Two passes: the first counts the pages so the footer can say '2 / 4'."""
    first = ItineraryPDF(itinerary)
    first.build()
    return ItineraryPDF(itinerary, total_pages=first.page).build()
