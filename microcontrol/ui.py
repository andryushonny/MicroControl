"""Общий вид окон: полупрозрачные панели, «чипы», строка подсказок с выделенными клавишами, всплывающее сообщение.
Один шрифт, одни цвета и отступы — в ручном режиме и в просмотре."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT = (238, 238, 238)
DIM = (165, 165, 165)
ACCENT = (0, 205, 255)          # жёлтый (BGR): активное, клавиши
OK = (110, 215, 110)
ERR = (90, 90, 255)
PANEL = (26, 27, 31)
LINE = (78, 80, 88)
PAD = 10


def size(s, scale, th=1):
    (w, h), b = cv2.getTextSize(s, FONT, scale, th)
    return w, h, b


def put(img, s, x, y, scale, color=TEXT, th=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, th, cv2.LINE_AA)


def panel(img, x0, y0, x1, y1, alpha=0.78, border=None):
    """Полупрозрачная тёмная панель (обрезается краями картинки)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    sub = img[y0:y1, x0:x1]
    sub[:] = (sub * (1 - alpha) + np.array(PANEL, np.float32) * alpha).astype(np.uint8)
    if border:
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), border, 1)


def chip(img, s, x, y, scale, color, fg=None):
    """Плашка режима: тонированный фон, рамка и текст цвета color (обычное начертание).
    (x, y) — левый край и базовая линия текста. Возвращает правый край."""
    w, h, _ = size(s, scale)
    x0, y0, x1, y1 = int(x), int(y - h - 7), int(x + w + 18), int(y + 7)
    tint = tuple(int(0.28 * c + 0.72 * p) for c, p in zip(color, PANEL))
    cv2.rectangle(img, (x0, y0), (x1, y1), tint, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
    light = tuple(min(255, int(c * 0.55 + 255 * 0.45)) for c in color)
    put(img, s, x + 9, y, scale, fg or light, 1)
    return x1


def keys_line(img, s, x, y, scale, max_x=None):
    """Строка подсказок «K: что делает  K2: ...»: клавиши (до двоеточия) — акцентом. Группы — через два пробела."""
    sep = size("   ", scale)[0]
    for grp in [g for g in s.split("  ") if g.strip()]:
        grp = grp.strip()
        k, _, d = grp.partition(": ")
        parts = [(k + ":", ACCENT), (" " + d, TEXT)] if d else [(grp, TEXT)]
        wgrp = sum(size(t, scale)[0] for t, _c in parts)
        if max_x and x + wgrp > max_x:
            break
        for t, c in parts:
            put(img, t, x, y, scale, c)
            x += size(t, scale)[0]
        x += sep
    return x


def help_panel(img, rows, title="Keys", W=None, x_off=0):
    """Все клавиши посреди картинки: [(клавиша, что делает)]; длинное переносится по словам, шрифт мельчает, пока
    всё не влезет по высоте. W — ширина области (слева), x_off — её левый край."""
    from .samples import wrap
    H, W = img.shape[0], (W or img.shape[1])
    pw = max(200, W - 40)
    for fs in (0.52, 0.48, 0.44, 0.40, 0.37, 0.34, 0.31):
        lh = size("Xg", fs)[1] + 9
        kw = min(max(size(k, fs)[0] for k, _d in rows) + 18, int(0.26 * pw))
        dw = pw - kw - 40
        lay = [(wrap(k, fs, kw - 14) if k else [""], wrap(d, fs, dw)) for k, d in rows]
        n = sum(max(len(a), len(b)) for a, b in lay)
        if lh * n + 56 <= H - 20:
            break
    ph = min(H - 20, lh * n + 56)
    x0, y0 = x_off + int((W - pw) / 2), max(10, int((H - ph) / 2))
    panel(img, x0, y0, x0 + pw, y0 + ph, 0.93, LINE)
    put(img, title, x0 + 18, y0 + 28, fs + 0.08, ACCENT, 2)
    y = y0 + 28 + lh
    for a, b in lay:
        for i in range(max(len(a), len(b))):
            if y > y0 + ph - 6:
                return
            if i < len(a):
                put(img, a[i], x0 + 18, y, fs, ACCENT, 1)
            if i < len(b):
                put(img, b[i], x0 + 18 + kw, y, fs, TEXT, 1)
            y += lh


def toast(img, msg, cx, y, scale, kind='ok', max_w=None):
    """Сообщение по центру: тёмная плашка с цветной полосой слева (зелёная — готово, красная — ошибка).
    Длинное — переносится по словам (до 5 строк), низ плашки — на y."""
    if not msg:
        return
    from .samples import wrap
    col = ERR if kind == 'err' else (ACCENT if kind == 'info' else OK)
    max_w = max_w or img.shape[1] - 40
    lines = wrap(msg, scale, max_w - 40)[:5]
    lh = size("Xg", scale)[1] + 8
    w = max(size(l_, scale)[0] for l_ in lines)
    h = size("Xg", scale)[1]
    x0 = int(cx - w / 2 - 18)
    yt = int(y - h - 12 - lh * (len(lines) - 1))
    panel(img, x0, yt, x0 + w + 36, y + 12, 0.88)
    cv2.rectangle(img, (x0, yt), (x0 + 4, int(y + 11)), col, -1)
    for i, l_ in enumerate(lines):
        put(img, l_, x0 + 20, int(y - lh * (len(lines) - 1 - i)), scale, TEXT)


def facts(img, items, x, y, scale, color=TEXT):
    """Факты через точку-разделитель: «30 frames · 1.40 um/px · ...». Возвращает правый край."""
    dot = "  |  "
    for k, t in enumerate(items):
        if k:
            put(img, dot, x, y, scale, LINE)
            x += size(dot, scale)[0]
        put(img, t, x, y, scale, color)
        x += size(t, scale)[0]
    return x


def chevron(img, cx, cy, r, direction, color, th=2):
    """Стрелка-шеврон ‹ или › (direction = -1 / +1), сглаженная, по центру (cx, cy)."""
    d = int(direction)
    pts = np.array([[cx - d * r * 0.45, cy - r], [cx + d * r * 0.55, cy], [cx - d * r * 0.45, cy + r]], np.int32)
    cv2.polylines(img, [pts], False, color, th, cv2.LINE_AA)


def round_rect(img, x0, y0, x1, y1, r, color, th=-1):
    """Прямоугольник со скруглёнными углами (заливка th=-1 или контур)."""
    x0, y0, x1, y1, r = int(x0), int(y0), int(x1), int(y1), int(max(1, r))
    if th < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        for a, b in (((x0 + r, y0), (x1 - r, y0)), ((x0 + r, y1), (x1 - r, y1)),
                     ((x0, y0 + r), (x0, y1 - r)), ((x1, y0 + r), (x1, y1 - r))):
            cv2.line(img, a, b, color, th, cv2.LINE_AA)
        for (cx, cy), a0 in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270), ((x0 + r, y1 - r), 90), ((x1 - r, y1 - r), 0)):
            cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a0 + 90, color, th, cv2.LINE_AA)


def plus(img, x, y, color, r=10, gap=4, th=1, halo=True):
    """Отметка креста «+» с точностью до доли пикселя: (x, y) — точка в координатах окна, где пиксель u
    занимает [u, u + 1). В центре — просвет gap, чтобы было видно, что лежит ровно под отметкой."""
    cx, cy = int(round(x - 0.5)), int(round(y - 0.5))      # целые пиксели: чёткие линии, без дробных
    segs = [((cx - r, cy), (cx - gap, cy)), ((cx + gap, cy), (cx + r, cy)),
            ((cx, cy - r), (cx, cy - gap)), ((cx, cy + gap), (cx, cy + r))]
    for col, t in (((0, 0, 0), th + 1), (color, th)) if halo else ((color, th),):
        for a, b in segs:
            cv2.line(img, a, b, col, t, cv2.LINE_8)


def dot(img, x, y, color, r=3):
    """Отметка креста — маленькая точка (без направлений: «+» выглядел бы как оси решётки, а их отметка не задаёт);
    (x, y) — точка окна, пиксель u занимает [u, u + 1); тёмное кольцо — видна на любом фоне."""
    c = (int(round(x - 0.5)), int(round(y - 0.5)))
    cv2.circle(img, c, r + 1, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, c, r, color, -1, cv2.LINE_AA)


def ruler(img, a, b, label, color=(0, 220, 255), scale=0.5):
    """Линейка: отметки «+» в точках a и b (координаты окна, b может быть None), линия между ними и подпись
    с расстоянием у середины, на тёмной плашке."""
    if a is None:
        return
    if b is not None:
        p, q = (int(round(a[0] - 0.5)), int(round(a[1] - 0.5))), (int(round(b[0] - 0.5)), int(round(b[1] - 0.5)))
        cv2.line(img, p, q, (0, 0, 0), 3, cv2.LINE_8)
        cv2.line(img, p, q, color, 1, cv2.LINE_8)
        plus(img, b[0], b[1], color)
    plus(img, a[0], a[1], color)
    if label:
        mx, my = (a if b is None else ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
        x0 = int(min(max(4, mx + 12), img.shape[1] - tw - 14))
        y0 = int(min(max(th + 10, my - 12), img.shape[0] - 6))
        panel(img, x0 - 6, y0 - th - 7, x0 + tw + 6, y0 + 6, 0.85, color)
        cv2.putText(img, label, (x0, y0), FONT, scale, (245, 245, 245), 1, cv2.LINE_AA)


def _bar(img, x0, y, x1, h, frac, t):
    """Полоса: доля frac (0…1); frac None — дело идёт, а доли нет: бегущий блок (видно, что не зависло)."""
    cv2.rectangle(img, (x0, y), (x1, y + h), (60, 60, 66), -1)
    if frac is None:
        w_ = max(12, (x1 - x0) // 5)
        a_ = x0 + int(((t * 0.6) % 1.0) * (x1 - x0 + w_)) - w_
        cv2.rectangle(img, (max(x0, a_), y), (min(x1, a_ + w_), y + h), (90, 200, 90), -1)
    else:
        cv2.rectangle(img, (x0, y), (x0 + int((x1 - x0) * max(0.0, min(1.0, frac))), y + h), (90, 200, 90), -1)
    cv2.rectangle(img, (x0, y), (x1, y + h), (110, 110, 118), 1)


def _tm(s):
    s = max(0, int(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


QROWS = 3                                                   # сколько задач очереди видно внизу (остальные — «+N», пока до них не дойдёт)


_DOTS = {}


def dot_step(key, period=0.18):
    """Какая из трёх точек горит: строго 1, 2, 3, 1, 2, 3 ... — не больше шага на кадр (окно подвисло — точка
    не перескакивает), шаг не чаще period с."""
    import time as _t
    now = _t.monotonic()
    d = _DOTS.get(key)
    if d is None:
        d = _DOTS[key] = [0, now]
    elif now - d[1] >= period:
        d[0], d[1] = (d[0] + 1) % 3, now
    return d[0]


def queue_rows(q):
    """Строки панели очереди: задачи (идущие, ждущие, только что закончившиеся) и «ещё N»."""
    js = q.recent() if q is not None else []
    run_ = [j for j in js if j.state == 'running']
    rest = [j for j in js if j.state != 'running']
    js = (run_ + rest)[:QROWS]
    more = len(run_) + len(rest) - len(js)
    return js, more


def queue_height(q, scale=0.45):
    js, more = queue_rows(q)
    if not js:
        return 0
    lh = size("Xg", scale)[1] + 12
    return lh * (len(js) + 1 + (1 if more else 0)) + 8


def queue_panel(img, x0, y1, x1, q, scale=0.45):
    """Очередь задач над строкой состояния (снизу вверх кончается на y1): заголовок «Queue: ... все — через
    ~T» и Stop all; по строке на задачу — дорожка (instrument / cpu), название, этап, полоса, прошло и осталось,
    x — остановить (ждущую — убрать). -> [(прямоугольник, действие, id задачи)], первым — вся панель."""
    import time as _t
    js, more = queue_rows(q)
    if not js:
        return []
    th = size("Xg", scale)[1]
    lh = th + 12
    h = queue_height(q, scale)
    y0 = y1 - h
    panel(img, x0, y0, x1, y1, 1.0, LINE)                    # непрозрачная: карта под ней не мешает
    out = [((x0, y0, x1, y1), 'none', None)]
    now = _t.monotonic()
    y = y0 + 4 + lh // 2
    head = q.summary() or "Queue: all done"
    put(img, head, x0 + 12, y + th // 2, scale, ACCENT, 1)
    if q.active():
        lab = "Stop all"
        bw = size(lab, scale)[0] + 16
        r = (x1 - 12 - bw, y - lh // 2 + 3, x1 - 12, y + lh // 2 - 3)
        cv2.rectangle(img, r[:2], r[2:], (40, 40, 120), -1)
        put(img, lab, r[0] + 8, y + th // 2, scale, TEXT, 1)
        out.append((r, 'all', None))
    for j in js:
        y += lh
        x = x0 + 12
        col = {'running': ACCENT, 'done': OK, 'failed': ERR, 'stopped': DIM}.get(j.state, DIM)
        if j.state == 'running' and not j.paused:
            for i in range(3):
                cv2.circle(img, (x + 5 + 10 * i, y), 3, ACCENT if dot_step(('q', j.id)) == i else LINE, -1, cv2.LINE_AA)
        else:
            if j.active and q.held(j):                      # ждёт объектив: знак паузы
                cv2.rectangle(img, (x + 10, y - 5), (x + 13, y + 5), ACCENT, -1)
                cv2.rectangle(img, (x + 17, y - 5), (x + 20, y + 5), ACCENT, -1)
            else:
                cv2.circle(img, (x + 15, y), 4, col, -1 if j.state != 'waiting' and j.state != 'queued' else 1, cv2.LINE_AA)
        x += 38
        lane = {'inst': "instrument", 'cpu': "cpu"}.get(j.lane, "on hold" if j.active and q.held(j) else "queued" if j.active else "")
        put(img, lane, x, y + th // 2, scale * 0.85, DIM, 1)
        x += size("instrument", scale * 0.85)[0] + 10
        t_ = j.title
        while t_ and size(t_, scale)[0] > 260:
            t_ = t_[:-2]
        put(img, t_, x, y + th // 2, scale, TEXT if j.active else DIM, 1)
        x += 270
        if j.paused:
            s_ = "paused: > resumes" + (f"  ({j.step_title} {100 * (j.frac or 0):.0f}%)" if j.state == 'running' else "")
        elif j.state == 'running':
            _bar(img, x, y - 6, x + 120, 12, j.frac, now)
            x += 130
            lft = j.remaining()
            s_ = f"{j.step_title}  {100 * (j.frac or 0):.0f}%  {_tm(j.elapsed())}" + (f", ~{_tm(lft)} left" if lft else "")
        elif j.active and q.held(j):
            s_ = f"waits for {j.obj}x: set the objective to {j.obj}x (O) - it starts by itself"
        elif j.active:
            nx_ = getattr(j, '_next', None)
            s_ = f"waiting: {nx_[1] if nx_ else 'start'}  ~{_tm(j.remaining())}"
        else:
            s_ = (j.note or j.state) if j.state != 'failed' else f"failed: {j.error}"
        bw = size("x", scale)[0] + 16
        pw = size("||", scale)[0] + 16
        while s_ and size(s_, scale)[0] > x1 - 32 - bw - pw - x:
            s_ = s_[:-2]
        put(img, s_, x, y + th // 2, scale, DIM if j.state != 'failed' else ERR, 1)
        if j.active:
            r = (x1 - 12 - bw, y - lh // 2 + 3, x1 - 12, y + lh // 2 - 3)
            cv2.rectangle(img, r[:2], r[2:], (40, 40, 120), -1)
            put(img, "x", r[0] + 8, y + th // 2, scale, TEXT, 1)
            out.append((r, 'stop', j.id))
            r2 = (r[0] - 6 - pw, r[1], r[0] - 6, r[3])           # пауза / дальше — у каждой задачи своя
            cv2.rectangle(img, r2[:2], r2[2:], (40, 90, 40) if j.paused else (70, 70, 76), -1)
            lab_ = ">" if j.paused else "||"
            put(img, lab_, r2[0] + (pw - size(lab_, scale)[0]) // 2, y + th // 2, scale, TEXT, 1)
            out.append((r2, 'pause', j.id))
    if more:
        y += lh
        put(img, f"+{more} more (menu Queue)", x0 + 50, y + th // 2, scale, DIM, 1)
    return out


def progress_panel(img, title, stage, frac, text="", passed=0.0, left=None, footer="", stopping=None):
    """Полоса прогресса посреди окна (долгая работа прямо в окне: сшивка, скан): заголовок, этап, полоса
    в процентах 0.00 … 100.00 % (frac None — доли нет: бегущий блок), что делается сейчас, сколько прошло и
    осталось, кнопка Stop (Esc). stopping: None — без кнопки, False — кнопка, True — «останавливается».
    -> прямоугольник кнопки Stop или None."""
    import time as _t
    H, W = img.shape[:2]
    pw, ph = min(700, W - 60), 206
    x0, y0 = int((W - pw) / 2), int((H - ph) / 2)
    panel(img, x0, y0, x0 + pw, y0 + ph, 0.94, LINE)
    put(img, title, x0 + 22, y0 + 38, 0.7, ACCENT, 2)
    pct = f"{100 * frac:.2f}% / 100.00%" if frac is not None else "working"
    st_ = stage
    while st_ and size(st_, 0.55)[0] > pw - 60 - size(pct, 0.5)[0]:
        st_ = st_[:-2]
    put(img, st_, x0 + 22, y0 + 74, 0.55, TEXT, 1)
    put(img, pct, x0 + pw - 22 - size(pct, 0.5)[0], y0 + 74, 0.5, TEXT, 1)
    _bar(img, x0 + 22, y0 + 92, x0 + pw - 22, 18, frac, _t.monotonic())
    t = text or ""
    while t and size(t, 0.45)[0] > pw - 44:
        t = t[:-2]
    put(img, t, x0 + 22, y0 + 136, 0.45, DIM, 1)
    put(img, f"{_tm(passed)} passed" + (f", about {_tm(left)} left  (total ~{_tm(passed + left)})" if left else "")
        + (f"   {footer}" if footer else ""),
        x0 + 22, y0 + 176, 0.45, DIM, 1)
    if stopping is None:
        return None
    lab = "Stopping..." if stopping else "Stop (Esc)"
    bw, bh = size(lab, 0.5)[0] + 28, 30
    r = (x0 + pw - 22 - bw, y0 + ph - 22 - bh, x0 + pw - 22, y0 + ph - 22)
    cv2.rectangle(img, r[:2], r[2:], (40, 40, 120) if not stopping else (50, 50, 56), -1)
    cv2.rectangle(img, r[:2], r[2:], ERR if not stopping else LINE, 1)
    put(img, lab, r[0] + 14, r[3] - 10, 0.5, TEXT, 1)
    return r


def status_bar(img, x0, y0, x1, y1, state=None, facts_=(), scale=0.45):
    """Строка состояния внизу окна: слева — что делается сейчас (бегущая точка, дело, этап, полоса, время,
    Stop) или «Ready»; дальше — факты (положение, объектив ...). state — как у run_busy или None.
    -> прямоугольник Stop или None."""
    import time as _t
    panel(img, x0, y0, x1, y1, 1.0)
    cy = (y0 + y1) // 2
    th = size("Xg", scale)[1]
    x = x0 + 12
    stop_r = None
    if state:
        now = _t.monotonic()
        k_ = dot_step('status')                             # бегущая точка: окно живо
        for i in range(3):
            c_ = ACCENT if k_ == i else LINE
            cv2.circle(img, (x + 5 + 10 * i, cy), 3, c_, -1, cv2.LINE_AA)
        x += 38
        nm = state.get('title', '')
        put(img, nm, x, cy + th // 2, scale, ACCENT, 1)
        x += size(nm, scale)[0] + 12
        _bar(img, x, cy - 6, x + 140, 12, state.get('frac'), now)
        x += 150
        el = now - state.get('t0', now)
        lft = state.get('left')
        t_ = ("working" if state.get('frac') is None else f"{100 * state['frac']:.0f}%") + f"  {_tm(el)}" + \
            (f", ~{_tm(lft)} left, total ~{_tm(el + lft)}" if lft else "")
        put(img, t_, x, cy + th // 2, scale, TEXT, 1)
        x += size(t_, scale)[0] + 12
        lab = "Stopping..." if state.get('stopping') else "Stop (Esc)"
        bw = size(lab, scale)[0] + 16
        stop_r = (x, y0 + 3, x + bw, y1 - 3)
        cv2.rectangle(img, stop_r[:2], stop_r[2:], (40, 40, 120) if not state.get('stopping') else (50, 50, 56), -1)
        put(img, lab, x + 8, cy + th // 2, scale, TEXT, 1)
        x += bw + 12
        sub = state.get('stage', '')
        if state.get('text'):
            sub = f"{sub}: {state['text']}" if sub and sub != nm else state['text']
        while sub and size(sub, scale)[0] > x1 - x - 10:
            sub = sub[:-2]
        put(img, sub, x, cy + th // 2, scale, DIM, 1)
    else:
        cv2.circle(img, (x + 5, cy), 4, OK, -1, cv2.LINE_AA)
        x += 16
        put(img, "Ready", x, cy + th // 2, scale, OK, 1)
        x += size("Ready", scale)[0] + 22
        if facts_:
            facts(img, list(facts_), x, cy + th // 2, scale)
    return stop_r


_BUSY_T = {}


def _busy_times(title=None, sec=None):
    """Сколько обычно идёт дело без своих долей (по прошлым разам): для полосы по времени."""
    import json, os
    p = os.path.join(os.path.expanduser('~'), '.microcontrol', 'busy_times.json')
    if not _BUSY_T:
        try:
            _BUSY_T.update(json.load(open(p, encoding='utf-8')))
        except (OSError, ValueError):
            _BUSY_T['_'] = 0
    if sec is None:
        return _BUSY_T.get(title)
    old = _BUSY_T.get(title)
    _BUSY_T[title] = round(sec if old is None else 0.6 * old + 0.4 * sec, 3)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(_BUSY_T, open(p, 'w', encoding='utf-8'))
    except OSError:
        pass


def run_busy(win, title, fn, frame, plan=None, expect=None, wait_ms=60, on_stop=None):
    """Долгое дело fn() в потоке, окно живёт: frame(state) -> (картинка, прямоугольник Stop в окне | None),
    показывается каждые wait_ms. Esc или щелчок по Stop — остановка: дело выходит на ближайшей полосе прогресса
    (util.check_stop -> Stopped); не вышло за 2.5 с — Stopped подбрасывается в поток (чистый Python), on_stop()
    (например, стоп столика) — сразу. Доля: от полос Progress; plan = [(начало названия полосы, вес)] — этапы
    одного дела, доля общая (не 0 -> 100 % на каждом этапе); полос нет — по времени: expect с или прошлые разы,
    иначе бегущий блок. -> dict(r=результат | e=ошибка, stopped=bool)."""
    import threading, time as _t, ctypes
    from .util import Progress, stop_begin, stop_end, stop_request, stop_active, stop_requested
    from .errors import Stopped
    st = dict(title=title, stage=title, frac=0.0, text="starting", left=None, t0=_t.monotonic(), n=0,
              stopping=False, ov=0.0, target=0.0)
    plan = list(plan or [])
    wsum = sum(w for _n, w in plan) or 1.0

    def hook(t, f, txt, el, eta):
        st['n'] += 1
        st['stage'], st['text'] = t, txt
        k = next((i for i, (nm, _w) in enumerate(plan) if t.startswith(nm)), None)
        if plan and k is not None:                          # общая доля всего дела: этапы друг за другом
            ov = (sum(w for _n, w in plan[:k]) + plan[k][1] * f) / wsum
            st['ov'] = max(st['ov'], ov)
            st['target'] = st['ov']
            st['left'] = (_t.monotonic() - st['t0']) * (1 - st['ov']) / st['ov'] if st['ov'] > 0.02 else None
        elif plan:
            st['target'] = st['ov']
        else:
            st['target'], st['left'] = f, eta
    res = {}

    def work():
        stop_begin()
        try:
            res['r'] = fn()
        except Stopped as e:
            res['e'], res['stopped'] = e, True
        except Exception as e:
            res['e'] = e
        finally:
            stop_end()
    exp_ = expect if expect is not None else _busy_times(title)
    Progress._open = None                                   # полосы прерванного дела не делают новые «вложенными»
    Progress.hook = hook
    th = threading.Thread(target=work, daemon=True)
    th.start()
    click = {}

    def cb(ev, x, y, *_a):
        if ev == cv2.EVENT_LBUTTONDOWN:
            click['xy'] = (x, y)
    try:
        cv2.setMouseCallback(win, cb)
    except cv2.error:
        pass
    t_stop, forced = None, False
    try:
        while th.is_alive():
            if st['n'] == 0:                                # своих долей нет: по времени или бегущий блок
                el = _t.monotonic() - st['t0']
                st['frac'] = min(0.95, el / exp_) if exp_ else None
                st['left'] = max(0.0, exp_ - el) if exp_ else None
            else:                                           # доля догоняет пришедшую плавно (без рывков)
                tg_ = st['target']
                fr_ = st['frac'] if st['frac'] is not None else 0.0
                if abs(tg_ - fr_) < 0.002 or (not plan and tg_ < fr_):   # новый этап без плана — с его доли
                    st['frac'] = tg_
                elif tg_ > fr_:
                    st['frac'] = fr_ + (tg_ - fr_) * 0.35
            img, r = frame(st)
            if img is not None:
                cv2.imshow(win, img)
            k_ = cv2.waitKeyEx(wait_ms)
            xy = click.pop('xy', None)
            hit = r is not None and xy is not None and r[0] <= xy[0] <= r[2] and r[1] <= xy[1] <= r[3]
            if (k_ == 27 or hit or stop_requested()) and not st['stopping']:
                st['stopping'], t_stop = True, _t.monotonic()
                st['text'] = "stopping..."
                stop_request()
                if on_stop is not None:
                    try:
                        on_stop()
                    except Exception:
                        pass
            if t_stop is not None and not forced and _t.monotonic() - t_stop > 2.5 and th.is_alive() \
                    and stop_active(th.ident):
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(th.ident), ctypes.py_object(Stopped))
                forced = True                               # чистый Python: выйдет на следующей строке
    finally:
        th.join()
        Progress.hook = None
        Progress._open = None
    if 'e' not in res and st['n'] == 0:
        _busy_times(title, _t.monotonic() - st['t0'])       # в следующий раз — полоса по времени
    res['seconds'] = _t.monotonic() - st['t0']
    if isinstance(res.get('e'), Stopped):
        res['stopped'] = True
    return res


def confirm_panel(img, title, lines, buttons=("Done", "Cancel"), hover=None, W=None, low=False):
    """Окно-условие посреди картинки: заголовок, строки, кнопки снизу. -> {подпись: (x0, y0, x1, y1)}.
    W — ширина области (слева), low — ниже середины (камера в центре видна)."""
    H, W = img.shape[0], (W or img.shape[1])
    pw = min(720, W - 60)
    wrapped = []
    for ln in lines:                                       # перенос по ширине
        words, cur = ln.split(), ""
        for w_ in words:
            t_ = (cur + " " + w_).strip()
            if size(t_, 0.52)[0] > pw - 48 and cur:
                wrapped.append(cur)
                cur = w_
            else:
                cur = t_
        wrapped.append(cur)
    ph = 70 + 26 * len(wrapped) + 70
    x0, y0 = int((W - pw) / 2), int((H - ph) * (0.88 if low else 0.5))
    panel(img, x0, y0, x0 + pw, y0 + ph, 0.95, LINE)
    put(img, title, x0 + 24, y0 + 40, 0.72, ACCENT, 2)
    for i, ln in enumerate(wrapped):
        put(img, ln, x0 + 24, y0 + 76 + 26 * i, 0.52, TEXT, 1)
    rects, bx = {}, x0 + pw - 24
    for b in reversed(buttons):
        tw = size(b, 0.55)[0]
        r = (bx - tw - 36, y0 + ph - 52, bx, y0 + ph - 18)
        on = hover is not None and r[0] <= hover[0] <= r[2] and r[1] <= hover[1] <= r[3]
        cv2.rectangle(img, r[:2], r[2:], (70, 150, 70) if b == buttons[0] else ((84, 84, 92) if on else (60, 60, 66)), -1)
        cv2.rectangle(img, r[:2], r[2:], (120, 200, 120) if b == buttons[0] else (110, 110, 118), 1)
        put(img, b, r[0] + 18, r[3] - 10, 0.55, TEXT, 1)
        rects[b] = r
        bx = r[0] - 12
    return rects


def star(img, cx, cy, r, color, fill=True):
    """Звезда о пяти лучах (основная съёмка)."""
    import math
    pts = [(cx + (r if i % 2 == 0 else r * 0.45) * math.sin(math.pi * i / 5),
            cy - (r if i % 2 == 0 else r * 0.45) * math.cos(math.pi * i / 5)) for i in range(10)]
    pts = np.array([[int(round(x * 8)), int(round(y * 8))] for x, y in pts], np.int32)
    if fill:
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA, shift=3)
    else:
        cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA, shift=3)


def pick_panel(img, title, header, rows, star_i=None, hover=None, scroll=0, W=None, note=None, cur=None):
    """Список посреди картинки: в каждой строке звезда (основная — жёлтая), поля по колонкам, кнопка View.
    rows — [(поле, ...)], header — подписи колонок. -> [((x0, y0, x1, y1), ('star' | 'view' | 'row', i) | 'close'
    | 'up' | 'down' | 'panel')] (первым — всё окно)."""
    H, W = img.shape[0], (W or img.shape[1])
    fs, rh, n_vis = 0.48, 28, max(3, min(len(rows), int((H - 200) / 28), 14))
    scroll = max(0, min(scroll, len(rows) - n_vis))
    cols = [max([size(str(h), fs)[0]] + [size(str(r[c]), fs)[0] for r in rows]) + 22 for c, h in enumerate(header)]
    bw = size("View", fs)[0] + 22
    pw = min(W - 40, max(sum(cols) + 44 + bw + 40, size(title, 0.62)[0] + 90, size(note or "", 0.42)[0] + 40))
    ph = 56 + 26 + rh * n_vis + (40 if note else 16)
    x0, y0 = int((W - pw) / 2), int((H - ph) / 2)
    panel(img, x0, y0, x0 + pw, y0 + ph, 0.96, LINE)
    put(img, title, x0 + 20, y0 + 34, 0.62, ACCENT, 1)
    out = [((x0, y0, x0 + pw, y0 + ph), 'panel')]
    cx1, cy0 = x0 + pw - 12, y0 + 12                        # закрыть
    on = hover is not None and cx1 - 24 <= hover[0] <= cx1 and cy0 <= hover[1] <= cy0 + 24
    cv2.rectangle(img, (cx1 - 24, cy0), (cx1, cy0 + 24), (96, 96, 104) if on else (62, 62, 70), -1)
    put(img, "x", cx1 - 16, cy0 + 17, 0.5, TEXT, 1)
    out.append(((cx1 - 24, cy0, cx1, cy0 + 24), 'close'))
    xs, x_ = [], x0 + 44
    for c in cols:
        xs.append(x_)
        x_ += c
    yh = y0 + 72
    for c, h in enumerate(header):
        put(img, str(h), xs[c], yh, fs, DIM, 1)
    for j in range(n_vis):
        i = scroll + j
        if i >= len(rows):
            break
        ry0 = yh + 8 + rh * j
        rr = (x0 + 10, ry0, x0 + pw - 10, ry0 + rh - 2)
        on = hover is not None and rr[0] <= hover[0] <= rr[2] and rr[1] <= hover[1] <= rr[3]
        if on or i == cur:
            cv2.rectangle(img, rr[:2], rr[2:], (70, 70, 78) if on else (56, 56, 64), -1)
        sc = (ry0 + rh // 2)
        star(img, x0 + 26, sc, 9, (40, 200, 255) if i == star_i else (150, 150, 156), fill=i == star_i)
        out.append(((x0 + 14, ry0, x0 + 40, ry0 + rh - 2), ('star', i)))
        for c, v in enumerate(rows[i]):
            put(img, str(v), xs[c], sc + 6, fs, (40, 200, 255) if i == star_i else TEXT, 1)
        br = (x0 + pw - 20 - bw, ry0 + 3, x0 + pw - 20, ry0 + rh - 5)
        onb = hover is not None and br[0] <= hover[0] <= br[2] and br[1] <= hover[1] <= br[3]
        cv2.rectangle(img, br[:2], br[2:], (96, 96, 104) if onb else (62, 62, 70), -1)
        put(img, "View", br[0] + 11, br[3] - 6, fs, TEXT, 1)
        out.insert(1, (br, ('view', i)))
        out.append((rr, ('row', i)))
    if len(rows) > n_vis:                                   # ещё строки: колесо
        put(img, f"{scroll + 1}-{scroll + n_vis} of {len(rows)}  (wheel)", x0 + pw - 200, y0 + 34, 0.42, DIM, 1)
    if note:
        put(img, note, x0 + 20, y0 + ph - 16, 0.42, DIM, 1)
    return out


MENU_H = 30


def dashed_poly(img, pts, color, thick=1, dash=9, gap=6):
    """Замкнутый пунктирный контур (точки — float, пиксели): так рисуются маски-черновики."""
    pts = np.asarray(pts, float)
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        L = float(np.hypot(*(b - a)))
        if L < 1e-6:
            continue
        d = (b - a) / L
        t = 0.0
        while t < L:
            p0, p1 = a + d * t, a + d * min(L, t + dash)
            cv2.line(img, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))),
                     color, thick, cv2.LINE_8)
            t += dash + gap


TOOL_H = 36                  # строка шагов и главных кнопок под меню


def tool_bar(W, steps, current, buttons, hover=None):
    """Под меню: шаги работы (1 Overview > 2 Pick areas > ...; текущий подсвечен) и большие кнопки справа.
    Возвращает (картинка TOOL_H x W, [(x0, x1, ('step', i) | ('button', подпись))])."""
    img = np.full((TOOL_H, W, 3), (28, 28, 32), np.uint8)
    cv2.line(img, (0, TOOL_H - 1), (W, TOOL_H - 1), (62, 62, 70), 1)
    hits, x = [], 10
    for i, st in enumerate(steps):
        tw = size(st, 0.46)[0]
        x0, x1 = x, x + tw + 16
        on = i == current
        hv = hover is not None and x0 <= hover <= x1
        cv2.rectangle(img, (x0, 6), (x1, TOOL_H - 7), (120, 80, 30) if on else ((58, 58, 66) if hv else (40, 40, 46)), -1)
        put(img, st, x0 + 8, TOOL_H - 13, 0.46, (255, 235, 200) if on else TEXT, 1)
        hits.append((x0, x1, ('step', i)))
        x = x1 + 4
        if i < len(steps) - 1:
            put(img, ">", x + 1, TOOL_H - 13, 0.46, DIM, 1)
            x += 16
    xr = W - 10
    for b in reversed(buttons):
        tw = size(b, 0.48)[0]
        x0, x1 = xr - tw - 20, xr
        hv = hover is not None and x0 <= hover <= x1
        cv2.rectangle(img, (x0, 5), (x1, TOOL_H - 6), (78, 80, 92) if hv else (52, 54, 62), -1)
        cv2.rectangle(img, (x0, 5), (x1, TOOL_H - 6), ACCENT if hv else (90, 92, 104), 1)
        put(img, b, x0 + 10, TOOL_H - 13, 0.48, (245, 245, 245), 1)
        hits.append((x0, x1, ('button', b)))
        xr = x0 - 8
    return img, hits


def menu_bar(W, title, items, hover=None, active=None):
    """Строка меню сверху окна: слева — открытый проект, дальше пункты (New, Open, ...). hover — x мыши
    в строке (подсветка). Возвращает (картинка MENU_H x W, [(x0, x1, пункт)])."""
    img = np.full((MENU_H, W, 3), (34, 34, 38), np.uint8)
    cv2.line(img, (0, MENU_H - 1), (W, MENU_H - 1), (62, 62, 70), 1)
    x = 10
    hits = []
    for it in items:
        tw = size(it, 0.5)[0]
        x0, x1 = x - 8, x + tw + 8
        if (hover is not None and x0 <= hover <= x1) or it == active:
            cv2.rectangle(img, (x0, 3), (x1, MENU_H - 4), (66, 66, 74), -1)
        put(img, it, x, MENU_H - 10, 0.5, TEXT, 1)
        hits.append((x0, x1, it))
        x = x1 + 12
    t = title
    while t and size(t, 0.5)[0] > W - x - 20:
        t = t[:-2]
    if t:
        put(img, t, W - 12 - size(t, 0.5)[0], MENU_H - 10, 0.5, ACCENT, 1)
    return img, hits


def tint_outside(img, shapes, color, alpha):
    """Закрасить полупрозрачно всё, КРОМЕ фигур (внутри — как есть: там и ищут).
    shapes: [(наружный контур Nx2 в px, [дыры Nx2])]. Без фигур — ничего не делает."""
    if not shapes:
        return img
    mask = np.zeros(img.shape[:2], np.uint8)
    for ext, holes in shapes:
        cv2.fillPoly(mask, [np.int32(np.round(np.asarray(ext, float)))], 255, cv2.LINE_8)
        for h in holes or ():
            cv2.fillPoly(mask, [np.int32(np.round(np.asarray(h, float)))], 0, cv2.LINE_8)
    tint = cv2.addWeighted(img, 1 - alpha, np.full_like(img, color), alpha, 0)
    cv2.copyTo(tint, cv2.bitwise_not(mask), img)
    return img


def geom_shapes(g, to_px):
    """shapely (Multi)Polygon -> [(наружный контур, [дыры])] в px окна (to_px — для массива точек)."""
    ps = [g] if g.geom_type == 'Polygon' else [p for p in getattr(g, 'geoms', []) if p.geom_type == 'Polygon']
    return [(to_px(np.array(p.exterior.coords)[:-1]), [to_px(np.array(h.coords)[:-1]) for h in p.interiors])
            for p in ps if not p.is_empty]
