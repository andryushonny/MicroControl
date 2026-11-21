"""Разметка сшитого скана: решётка крестов, выделенные квадраты, вывод только выделенного, очистка.

Всё в «пикселях мозаики» — координатах окна просмотра (положения кадров после mc.stitch, начало в 0).
Решётка: крест (i, j) в точке O + i·v1 + j·v2. Квадрат (i, j) — между крестами i..i+1, j..j+1.
Файл <скан>/grid.json: отметки крестов, выделенные квадраты.
Вывод (mc.export): только выделенные квадраты, ровно (поворот снят), фон прозрачный (PNG) или белый (JPG);
DXF для AutoCAD — квадраты в мкм.
"""
from .samples import per_sample as _per_sample
import json, math, os, shutil
import numpy as np
import cv2
from . import log as L
from .crosses import Crosses, fit_square
from .storage import ScanData
from .util import Result, Progress


# ------------------------------------------------------------------ состояние разметки
SENSITIVITY = 50                 # чувствительность жёлтой подсказки по умолчанию, 0…100


def sens_frac(S):
    """Чувствительность 0…100 -> доля квадрата, которую должно занять «что-то»: 5 % … 0,05 %."""
    return 0.05 * 10 ** (-2 * float(S) / 100)


def _stage_layout(folder, P):
    """Положения кадров по столику (как в просмотре до сшивки), если сетка размечена раньше, чем сделана
    сшивка (grid.json старее join/positions_all.csv). Иначе None."""
    g_, j_ = os.path.join(folder, 'grid.json'), os.path.join(folder, 'join', 'positions_all.csv')
    if not (os.path.exists(g_) and os.path.exists(j_)) or os.path.getmtime(g_) >= os.path.getmtime(j_):
        return None
    try:
        from .storage import ScanData
        from .stitch import meta_A
        st = ScanData(folder)
        H, W = st.shape[:2]
        p = np.array([[r['x_p'], r['y_p']] for r in st.index], float)
        st.close()
        Q = -(p @ meta_A(st.meta, W, H).T)
        Q = Q - Q.min(0)
        return Q if Q.shape == np.asarray(P).shape else None
    except Exception:
        return None


class Grid:
    def __init__(self, folder):
        self.folder = folder
        self.cr = Crosses()
        self.selected = set()
        self.labels = {}                                    # (i, j) -> (X, Y): номера, написанные литографией
        self.arm = None                                     # плечо креста, px мозаики (находится само)
        self.useful = {}                                    # подсказка: квадраты, где что-то есть -> доля
        self.sensitivity = SENSITIVITY
        self.refined = None
        self.frames_px = None                               # где стояли кадры, когда ставились кресты (px мозаики)
        p = os.path.join(folder, 'grid.json')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                d = json.load(f)
            for q, ij in zip(d.get('marks', []), d.get('index', [])):
                self.cr.marks.append(np.array(q, float))
                self.cr.ij.append(tuple(ij))
            self.cr._fit()
            v1_, v2_ = np.array(d.get('v1_px', [1, 0]), float), np.array(d.get('v2_px', [0, 1]), float)
            square = abs(v1_ @ v2_) < 0.005 * (v1_ @ v1_) and abs(v1_ @ v1_ - v2_ @ v2_) < 0.01 * (v1_ @ v1_)
            if d.get('refined') and 'v1_px' in d and square:      # решётка, уточнённая по всем крестам
                self.cr.O, self.cr.v1, self.cr.v2 = (np.array(d['origin_px']), np.array(d['v1_px']),
                                                     np.array(d['v2_px']))
                self.refined = d['refined']
            self.selected = {tuple(c) for c in d.get('selected', [])}
            self.labels = {(int(a), int(b)): (int(X), int(Y)) for a, b, X, Y in d.get('labels', [])}
            self.arm = d.get('arm_px')
            self.useful = {(int(a), int(b)): f for a, b, f in d.get('useful', [])}
            self.sensitivity = d.get('sensitivity', SENSITIVITY)
            self.frames_px = np.array(d['frames_px'], float) if 'frames_px' in d else None
            if 'hint_scores' in d:
                self.scores = {(int(a), int(b)): f for a, b, f in d['hint_scores']}
                self.hint_squares = d.get('hint_squares', len(self.scores))

    def sync(self, P, W, H):
        """Кадры встали по-другому (после сшивки или пересшивки): перенести кресты вместе с их кадром.
        P — положения кадров сейчас (px мозаики, левый верх), W, H — кадр. Крест остаётся в том же месте
        своего кадра (кадр выбирается тот, у центра которого крест ближе всего). Возвращает, сколько перенесено."""
        P = np.asarray(P, float)
        old = self.frames_px
        if old is None:                                     # старый grid.json: кресты ставились до сшивки?
            old = _stage_layout(self.folder, P)
        moved = 0
        if old is not None and old.shape == P.shape and self.cr.marks and np.abs(old - P).max() > 0.5:
            c_old = old + (W / 2, H / 2)
            for n, q in enumerate(self.cr.marks):
                inside = np.all((q >= old) & (q <= old + (W, H)), axis=1)
                d = np.hypot(*(c_old - q).T)
                d[~inside] = np.inf
                k = int(np.argmin(d))
                if np.isfinite(d[k]):
                    self.cr.marks[n] = q - old[k] + P[k]
                    moved += 1
            self.cr._fit()
        self.frames_px = P.copy()
        return moved

    def save(self, px_um=None):
        v1, v2 = self.cr.vectors()
        d = dict(marks=[list(map(float, q)) for q in self.cr.marks], index=[list(t) for t in self.cr.ij],
                 selected=sorted([list(c) for c in self.selected]),
                 labels=[[i, j, X, Y] for (i, j), (X, Y) in sorted(self.labels.items())])
        P, t, _ = self.numbering()
        if self.labels:
            d['numbering'] = dict(P=P.tolist(), t=t.tolist(), note="(X, Y) = P (i, j) + t")
        if self.arm:
            d['arm_px'] = int(self.arm)
        d['sensitivity'] = self.sensitivity
        if self.useful:
            d['useful'] = [[i, j, f] for (i, j), f in sorted(self.useful.items())]
        sc_ = getattr(self, 'scores', None)
        if sc_:                                             # доли всех квадратов: чувствительность — мгновенно
            d['hint_scores'] = [[i, j, round(f, 5)] for (i, j), f in sorted(sc_.items()) if f > 0]
            d['hint_squares'] = getattr(self, 'hint_squares', len(sc_))
        if self.refined:
            d['refined'] = self.refined
        if self.frames_px is not None:
            d['frames_px'] = np.round(self.frames_px, 2).tolist()
        if v1 is not None:
            d.update(origin_px=list(map(float, self.cr.O)), v1_px=list(map(float, v1)), v2_px=list(map(float, v2)),
                     turn_deg=self.cr.angle())
            if px_um:
                d['pitch_um'] = [float(np.hypot(*v1) * px_um), float(np.hypot(*v2) * px_um)]
        with open(os.path.join(self.folder, 'grid.json'), 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=1)

    def refine(self, render, size, px_um, min_resp=0.1):
        """Уточнить решётку по ВСЕМ крестам мозаики: в каждом узле, куда попадает решётка, центр креста
        ищется по симметрии; затем O, v1, v2 — МНК по всем найденным (выбросы > 3σ отбрасываются).
        render(cx, cy, w, h) -> кусок мозаики в масштабе 1. Ошибка ручных отметок перестаёт копиться
        вдоль решётки. Возвращает (число крестов, СКО отклонения от решётки, px)."""
        if not self.ready():
            return 0, None
        r = int(min(150, max(8, 15.0 / px_um)))
        nodes, regs = [], []
        for i, j in self.cells_range((0, 0), size):
            q = self.cr.point(i, j)
            if not (2 * r <= q[0] <= size[0] - 2 * r and 2 * r <= q[1] <= size[1] - 2 * r):
                continue
            reg = render(q[0], q[1], 4 * r + 1, 4 * r + 1)
            g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY) if reg.ndim == 3 else reg
            if g.std() < 2:                                      # пусто (вне кадров)
                continue
            nodes.append(((i, j), q)); regs.append(g)
        if not self.arm and len(nodes) >= 6:
            # плечо креста: при каком размере шаблона найденные центры лучше всего ложатся на решётку
            # (у креста бывают цифры номера — с ними шаблон не того размера уходит на цифру)
            pick = np.linspace(0, len(nodes) - 1, min(40, len(nodes))).astype(int)
            score = {}
            for L in [a for a in ARMS if a <= 0.9 * r]:
                P_, IJ_ = [], []
                for k in pick:
                    d, resp = snap_cross(regs[k], r, arm=L)
                    if resp >= min_resp and np.any(d):              # не нашёл (d = 0) — не считается
                        P_.append(nodes[k][1] + d); IJ_.append(nodes[k][0])
                if len(P_) < max(4, 0.5 * len(pick)):
                    continue
                M_ = np.c_[np.ones(len(P_)), np.array(IJ_, float)]
                sol_ = np.linalg.lstsq(M_, np.array(P_), rcond=None)[0]
                res_ = np.hypot(*(M_ @ sol_ - np.array(P_)).T)
                for _ in range(2):                                   # без выбросов
                    ok_ = res_ < max(1.5, 2.5 * float(np.median(res_)))
                    sol_ = np.linalg.lstsq(M_[ok_], np.array(P_)[ok_], rcond=None)[0]
                    res_ = np.hypot(*(M_ @ sol_ - np.array(P_)).T)
                score[L] = (-int((res_ < 2.0).sum()), float(np.median(res_)))   # больше крестов точно на решётке
            self.arm = min(score, key=score.get) if score else None
        IJ, Pts = [], []
        for (ij, q), g in zip(nodes, regs):
            d, resp = snap_cross(g, r, arm=self.arm)
            if resp >= min_resp:
                IJ.append(ij); Pts.append(q + d)
        if len(Pts) < 4:
            return len(Pts), None
        IJ, Pts = np.array(IJ, float), np.array(Pts)
        keep = np.ones(len(Pts), bool)
        for _ in range(3):                                      # квадратная решётка: сдвиг, шаг, наклон
            O_, a_ = fit_square(IJ[keep], Pts[keep])
            pred = O_ + IJ[:, :1] * a_ + IJ[:, 1:] * np.array([-a_[1], a_[0]])
            res = np.hypot(*(pred - Pts).T)
            sd = max(np.median(res[keep]) * 1.48, 0.3)
            new = res < 3 * sd + 0.5
            if (new == keep).all():
                break
            keep = new
        self.cr.O, self.cr.v1, self.cr.v2 = O_, a_, np.array([-a_[1], a_[0]])
        rms = float(np.sqrt((res[keep] ** 2).mean()))
        self.refined = dict(crosses=int(keep.sum()), rms_px=round(rms, 2))
        return int(keep.sum()), rms

    def numbering(self):
        """Как индексы решётки (i, j) переходят в настоящие номера квадратов (X, Y): (X, Y) = P (i, j) + t,
        P — перестановка осей со знаками (8 вариантов). Подбирается по подписанным квадратам: одна подпись —
        только сдвиг (направления — по умолчанию: X вправо, Y вниз на экране); подписи в другом ряду и столбце —
        направления обеих осей. Возвращает (P, t, текст состояния)."""
        P0 = self._default_P()
        if not self.labels:
            return P0, np.zeros(2, int), "no labels: lattice indices"
        L = [(np.array(c, int), np.array(v, int)) for c, v in self.labels.items()]
        c0, v0 = L[0]
        best = []
        for sw in (False, True):
            for sx in (1, -1):
                for sy in (1, -1):
                    P = np.array([[0, sx], [sy, 0]] if sw else [[sx, 0], [0, sy]], int)
                    t = v0 - P @ c0
                    bad = sum(int(np.abs(P @ c + t - v).sum()) for c, v in L)
                    best.append((bad, int(np.abs(P - P0).sum()), P, t))
        best.sort(key=lambda q: (q[0], q[1]))
        bad, _, P, t = best[0]
        free = [q for q in best if q[0] == 0]
        if bad:
            note = f"labels disagree (off by {bad}): check them"
        elif len(free) > 1:
            note = f"{len(self.labels)} label(s): {len(free)} ways fit, label one more square (another row and column)"
        else:
            note = f"numbering fixed by {len(self.labels)} labels"
        return P, t, note

    def _default_P(self):
        """X — по оси решётки, что ближе к горизонтали (растёт вправо), Y — по другой (растёт вниз)."""
        v1, v2 = self.cr.vectors()
        if v1 is None:
            return np.eye(2, dtype=int)
        if abs(v1[0]) >= abs(v2[0]):
            return np.array([[1 if v1[0] >= 0 else -1, 0], [0, 1 if v2[1] >= 0 else -1]], int)
        return np.array([[0, 1 if v2[0] >= 0 else -1], [1 if v1[1] >= 0 else -1, 0]], int)

    def name(self, i, j, num=None):
        """Настоящий номер квадрата (X, Y) по подписям; без подписей — индекс решётки."""
        P, t, _ = num or self.numbering()
        X, Y = P @ np.array([i, j]) + t
        return int(X), int(Y)

    def find_useful(self, render, size, n=160, sensitivity=None, dark=False, model=None, info=None):
        """Подсказка: квадраты, в которых что-то есть (чешуйки, кристаллы, любые предметы) — поиск аномалий
        (mask -> anomaly.py, принцип PaDiM): подложка описывается смесью гауссиан признаков маленьких участков,
        «что-то» — участки, далёкие от неё. Учится без разметки на самой мозаике (почти все квадраты пустые)
        или берёт модель пустой подложки (model: файл hint_model.npz / папка скана с ним / anomaly.Normal).
        Для слабого ПК вся мозаика — один уменьшенный снимок (≤ ~6 Мпикс). Кресты с номерами (около узлов
        решётки), края скана и пыль (мелкие тёмные бесцветные точки; dark=True — считать и их) не считаются.
        sensitivity 0…100 (по умолчанию 50; больше — больше подсказок): доля квадрата, которую должно занять
        «что-то», от 5 % (0) до 0,05 % (100). render(cx, cy, s, w, h) -> кусок мозаики в масштабе s.
        Возвращает {(i, j): доля}; info (dict) получает время, число патчей и модель."""
        import time as _t
        from . import anomaly as AD
        if not self.ready():
            return {}
        t0 = _t.monotonic()
        v1, v2 = self.cr.vectors()
        pitch = min(np.hypot(*v1), np.hypot(*v2))
        cells = list(self.cells_range((0, 0), size))
        n = int(min(n, max(32, (6e6 / max(1, len(cells))) ** 0.5)))   # слабый ПК: вся мозаика ≤ ~6 Мпикс
        s = min(1.0, n / pitch)
        W, H = int(np.ceil(size[0] * s)), int(np.ceil(size[1] * s))
        big = render(size[0] / 2, size[1] / 2, s, W, H)
        void = (big <= 22).all(2).astype(np.uint8)            # вне кадров (фон просмотра, чёрные поля обзора)
        net = AD.load_backbone()
        F = AD.features(big, 0.75 * n, net)
        del big
        Hp, Wp = F.shape[:2]
        p = AD.PATCH
        valid = cv2.resize(void, (Wp, Hp), interpolation=cv2.INTER_AREA) == 0
        del void
        # кресты и номера у них: круг около каждого узла решётки
        cm = np.zeros((Hp, Wp), np.uint8)
        r = max(1, int(round(0.25 * n / p)))
        for i, j in cells:
            q = self.cr.point(i, j) * s / p
            if -r <= q[0] < Wp + r and -r <= q[1] < Hp + r:
                cv2.circle(cm, (int(round(q[0])), int(round(q[1]))), r, 1, -1)
        valid &= cm == 0
        if isinstance(model, str):
            path = os.path.join(model, AD.MODEL_FILE) if os.path.isdir(model) else model
            model = AD.Normal.load(path)
        if model is None:
            model = AD.Normal.fit(F[valid], kind='cnn' if net is not None else 'hand')
        m, _z = AD.anomaly_mask(F, model, valid, dust=not dark)
        # доля «чего-то» в каждом квадрате: центр патча -> квадрат решётки
        yy, xx = np.mgrid[0:Hp, 0:Wp]
        P = np.stack([(xx.ravel() + 0.5) * p / s, (yy.ravel() + 0.5) * p / s], 1) - self.cr.O
        del xx, yy
        ij = np.floor(P @ np.linalg.inv(np.column_stack([v1, v2])).T).astype(np.int64)
        del P
        i0, j0 = ij.min(0)
        key = (ij[:, 0] - i0) * (ij[:, 1].max() - j0 + 1) + (ij[:, 1] - j0)
        nk = int(key.max()) + 1
        tot = np.bincount(key, minlength=nk)                       # все патчи квадрата
        good = np.bincount(key, weights=valid.ravel(), minlength=nk)
        hit = np.bincount(key, weights=m.ravel(), minlength=nk)
        out = {}
        nj = ij[:, 1].max() - j0 + 1
        for kk in np.nonzero(good)[0]:
            if good[kk] < 0.5 * tot[kk] or tot[kk] < 0.5 * (pitch * s / p) ** 2:   # квадрат почти вне скана
                continue
            out[(int(kk // nj + i0), int(kk % nj + j0))] = float(hit[kk] / good[kk])
        self.scores, self.hint_squares = out, len(out)
        self.hint_model = model
        if info is not None:
            info.update(seconds=round(_t.monotonic() - t0, 2), patches=int(valid.sum()), square_px=n,
                        model='CNN (ONNX)' if net is not None else 'colour + texture', components=len(model.means))
        return self.pick_useful(sensitivity)

    def pick_useful(self, sensitivity=None):
        """Подсказка по уже посчитанным долям (мгновенно): квадраты, где «что-то» занимает больше порога."""
        S = self.sensitivity if sensitivity is None else float(min(100, max(0, sensitivity)))
        self.sensitivity = S
        lim = sens_frac(S)
        self.useful = {c: round(x, 4) for c, x in getattr(self, 'scores', {}).items() if x > lim}
        return self.useful

    def ready(self):
        return self.cr.vectors()[0] is not None

    def cell_at(self, pt):
        v1, v2 = self.cr.vectors()
        a = np.linalg.solve(np.column_stack([v1, v2]), np.asarray(pt, float) - self.cr.O)
        return int(math.floor(a[0])), int(math.floor(a[1]))

    def corners(self, i, j):
        P = self.cr.point
        return np.array([P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)])

    def cells_in(self, poly, size):
        """Квадраты, чьи центры внутри многоугольника poly (px мозаики); size = (W, H) мозаики."""
        poly = np.asarray(poly, np.float32)
        out = set()
        for i, j in self.cells_range(poly.min(0), poly.max(0)):
            c = self.cr.point(i + 0.5, j + 0.5)
            if cv2.pointPolygonTest(poly, (float(c[0]), float(c[1])), False) >= 0:
                out.add((i, j))
        return out

    def cells_touching(self, poly):
        """Квадраты, которых многоугольник касается хоть краем (для рамки: не нужно дотягивать до центров)."""
        poly = np.asarray(poly, np.float32)
        out = set()
        for i, j in self.cells_range(poly.min(0), poly.max(0)):
            sq = self.corners(i, j).astype(np.float32)
            inside = lambda P, q: cv2.pointPolygonTest(P, (float(q[0]), float(q[1])), False) >= 0
            if any(inside(poly, q) for q in sq) or any(inside(sq, q) for q in poly) or \
                    inside(poly, self.cr.point(i + 0.5, j + 0.5)):
                out.add((i, j))
        return out

    def cells_range(self, lo, hi):
        v1, v2 = self.cr.vectors()
        V = np.column_stack([v1, v2])
        ij = np.array([np.linalg.solve(V, np.array(q, float) - self.cr.O)
                       for q in ((lo[0], lo[1]), (hi[0], lo[1]), (lo[0], hi[1]), (hi[0], hi[1]))])
        for i in range(int(math.floor(ij[:, 0].min())) - 1, int(math.ceil(ij[:, 0].max())) + 1):
            for j in range(int(math.floor(ij[:, 1].min())) - 1, int(math.ceil(ij[:, 1].max())) + 1):
                yield i, j


def _plus(L, t):
    """Шаблон креста: плечо L, толщина t, с тёмной (−1) каймой — подходит и к тонким тёмным крестам,
    и к рельефным светлым с контуром."""
    k = max(2, t // 2)
    n = 2 * (L + k) + 3
    T = np.zeros((n, n), np.float32)
    c = n // 2
    T[c - t // 2:c + t // 2 + 1, c - L:c + L + 1] = 1
    T[c - L:c + L + 1, c - t // 2:c + t // 2 + 1] = 1
    return T - (cv2.dilate(T, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) - T)


def _sym(g, p, h):
    """Насколько окно (полуразмер h) вокруг p совпадает с собой при повороте на 90° и на 180°
    (крест — да, цифры рядом с ним — нет). От −1 до 1."""
    x, y = int(round(p[0])), int(round(p[1]))
    if h < 3 or x - h < 0 or y - h < 0 or x + h + 1 > g.shape[1] or y + h + 1 > g.shape[0]:
        return -1.0
    P = g[y - h:y + h + 1, x - h:x + h + 1]
    P = P - P.mean()
    n = (P * P).sum() + 1e-6
    return float(min((P * np.rot90(P)).sum(), (P * P[::-1, ::-1]).sum()) / n)


ARMS = (6, 7, 9, 11, 13, 15, 19, 23, 27, 33, 40, 48)   # плечо креста, px (перебор, пока не известно)


def snap_cross(region, r, arm=None, info=None):
    """Центр креста в окне region (серое). Крест ищется шаблоном «+» (плечо arm, px; без arm — перебор),
    кандидат оценивается по симметрии креста при повороте на 90° и 180°: цифры у креста (номера литографии)
    так не симметричны и центр не уводят. Запасной вариант — корреляция окна с повёрнутым на 180°.
    Возвращает ((dx, dy) от середины окна, px; отклик 0…1). info (dict) получает плечо 'arm'."""
    g0 = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 1.0)
    g = g0 - cv2.GaussianBlur(g0, (0, 0), max(4.0, (arm or r / 3) * 0.8))     # убрать фон: только детали
    mid = (np.array(g.shape[::-1], float) - 1) / 2
    best = (-1.0, None, None)
    for L in ([int(arm)] if arm else [a for a in ARMS if a <= 0.9 * r]):
        for t in sorted({max(3, L // 2), max(3, L // 5)}):
            T = _plus(L, t)
            if T.shape[0] + 2 >= min(g.shape):
                continue
            R = np.abs(cv2.matchTemplate(g, T, cv2.TM_CCOEFF_NORMED))
            _, v, _, (px, py) = cv2.minMaxLoc(R)
            c = np.array([px + T.shape[1] // 2, py + T.shape[0] // 2], float)
            if 0 < px < R.shape[1] - 1 and 0 < py < R.shape[0] - 1:          # доля пикселя: центр масс 3x3
                w = R[py - 1:py + 2, px - 1:px + 2] - R[py - 1:py + 2, px - 1:px + 2].min()
                if w.sum() > 0:
                    c += (w.sum(0) @ [-1, 0, 1] / w.sum(), w.sum(1) @ [-1, 0, 1] / w.sum())
            sc = v * max(0.0, _sym(g, c, int(1.3 * L)))
            if sc > best[0]:
                best = (sc, c, L)
    sc, c, L = best
    if c is None or sc < 0.03:                           # шаблон не нашёл — старый способ (симметрия 180°)
        win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        (sx, sy), resp = cv2.phaseCorrelate(g * win, np.ascontiguousarray(g[::-1, ::-1]) * win)
        d = -np.array([sx, sy]) / 2.0                    # повёрнутое окно сдвинуто на -2·d
        if info is not None:
            info['arm'] = None
        return (d if np.hypot(*d) < r else np.zeros(2)), resp
    if info is not None:
        info['arm'] = L
    d = c - mid
    return (d if np.hypot(*d) < r else np.zeros(2)), float(min(1.0, sc * 3))


# ------------------------------------------------------------------ сборка нужных квадратов из кадров
class Frames:
    """Кадры скана с положениями (px мозаики) — для сборки любого участка без всей мозаики в памяти."""
    def __init__(self, folder):
        from .stitch import load_positions
        self.st = ScanData(folder)
        self.H, self.W = self.st.shape[:2]
        P = load_positions(folder, self.st)
        self.P = P - P.min(0)
        self.size = self.P.max(0) + (self.W, self.H)
        y, x = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        d = np.minimum(np.minimum(x + 1, self.W - x), np.minimum(y + 1, self.H - y))
        self.feather = np.minimum(d / max(8.0, 0.05 * min(self.W, self.H)), 1.0)   # вес: к краю кадра — 0

    def warp(self, Mout2mos, size, bg, blend='multiband'):
        """Картинка size=(w, h): пиксель (u, v) берётся из точки мозаики Mout2mos @ (u, v, 1).
        blend='multiband' — смешивание по частотам (без двоения и без ступеньки яркости), 'feather' — веса к краю."""
        w, h = size
        quad = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ Mout2mos.T
        lo, hi = quad.min(0), quad.max(0)
        acc = np.zeros((h, w, 3), np.float32)
        ws = np.zeros((h, w), np.float32)
        imgs, wts = [], []
        hit = np.flatnonzero((self.P[:, 0] < hi[0]) & (self.P[:, 0] + self.W > lo[0]) &
                             (self.P[:, 1] < hi[1]) & (self.P[:, 1] + self.H > lo[1]))
        for k in hit:
            img = self.st.frame(int(k))
            img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            M = Mout2mos.copy()
            M[:, 2] -= self.P[k]                                     # вывод -> пиксель кадра
            M = M.astype(np.float32)
            t = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP).astype(np.float32)
            f = cv2.warpAffine(self.feather, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            acc += t * f[..., None]
            ws += f
            imgs.append(t); wts.append(f)
        avg = np.where(ws[..., None] > 1e-3, acc / np.maximum(ws, 1e-6)[..., None], bg)
        if blend == 'multiband' and len(imgs) > 1:
            from .optics import multiband
            for t, f in zip(imgs, wts):                         # вне кадра — среднее (без тёмных ореолов)
                bad = f <= 1e-6
                t[bad] = avg[bad]
            mb, _ = multiband(imgs, wts)
            avg = np.where(ws[..., None] > 1e-3, mb, bg)
        return avg.astype(np.uint8), ws > 1e-3

    def close(self):
        self.st.close()


@_per_sample
def export(folder, fmt='png', scale=1.0, cells=False, dxf=True, max_mb=2000, margin=0, blend='multiband'):
    """Вывести только выделенные квадраты: ровно (поворот подложки снят), стык строго по крестам.
    fmt='png' — фон прозрачный, 'jpg' — белый. cells=True — ещё и каждый квадрат отдельным файлом.
    dxf=True — export/selection.dxf: квадраты в мкм (AutoCAD). Возвращает папку export/."""
    g = Grid(folder)
    if not g.ready() or not g.selected:
        raise ValueError("Nothing to export: in mc.view(folder) mark the crosses (G) and select squares (S).")
    fr = Frames(folder)
    px_um = fr.st.meta.get('pixel_size_um') or 1.0
    v1, v2 = g.cr.vectors()
    ex, ey = (1 if v1[0] >= 0 else -1), (1 if v2[1] >= 0 else -1)   # вывод — как на экране (без зеркала)
    a, b = np.hypot(*v1), np.hypot(*v2)
    sel = sorted(g.selected)
    cols = {i: ex * i for i, _ in sel}
    rows = {j: ey * j for _, j in sel}
    c0, c1 = min(cols.values()), max(cols.values())
    r0, r1 = min(rows.values()), max(rows.values())
    cw, ch = int(round(a * scale)), int(round(b * scale))
    Wc, Hc = (c1 - c0 + 1) * cw, (r1 - r0 + 1) * ch
    k = min(1.0, (max_mb * 2 ** 20 / (Wc * Hc * 4)) ** 0.5)
    if k < 1:
        cw, ch, scale = int(cw * k), int(ch * k), scale * k
        Wc, Hc = (c1 - c0 + 1) * cw, (r1 - r0 + 1) * ch
    jpg = fmt.lower() in ('jpg', 'jpeg')
    canvas = np.full((Hc, Wc, 3 if jpg else 4), 255 if jpg else 0, np.uint8)
    out = os.path.join(folder, 'export')
    os.makedirs(out, exist_ok=True)
    if cells:
        os.makedirs(os.path.join(out, 'squares'), exist_ok=True)
    num = g.numbering()
    pr = Progress(len(sel), "Export")
    for i, j in sel:
        pr.update(1)
        # угол квадрата на экране — верхний левый: при ex = -1 это крест i + 1
        corner = g.cr.point(i + (1 if ex < 0 else 0), j + (1 if ey < 0 else 0))
        pad = 16
        Mq = np.c_[ex * v1 / cw, ey * v2 / ch, corner]              # (u, v, 1) -> мозаика
        Mq[:, 2] -= Mq[:, :2] @ np.array([pad, pad], float)
        img, ok = fr.warp(Mq, (cw + 2 * pad, ch + 2 * pad), 255, blend)
        img, ok = img[pad:pad + ch, pad:pad + cw], ok[pad:pad + ch, pad:pad + cw]
        x0, y0 = (cols[i] - c0) * cw, (rows[j] - r0) * ch
        if jpg:
            canvas[y0:y0 + ch, x0:x0 + cw] = img
        else:
            canvas[y0:y0 + ch, x0:x0 + cw, :3] = img
            canvas[y0:y0 + ch, x0:x0 + cw, 3] = np.where(ok, 255, 0)
        if cells:
            ok_, buf = cv2.imencode('.png', img)
            X, Y = g.name(i, j, num)
            buf.tofile(os.path.join(out, 'squares', f"square_X{X:+04d}_Y{Y:+04d}.png"))
    pr.close()
    path = os.path.join(out, 'selection.' + ('jpg' if jpg else 'png'))
    ok_, buf = cv2.imencode(os.path.splitext(path)[1], canvas, [cv2.IMWRITE_JPEG_QUALITY, 95] if jpg else
                            [cv2.IMWRITE_PNG_COMPRESSION, 6])
    buf.tofile(path)
    pitch = (a * px_um, b * px_um)
    info = dict(squares=[list(c) for c in sel], names=[list(g.name(i, j, num)) for i, j in sel],
                numbering=num[2], pitch_um=list(pitch), turn_deg=g.cr.angle(),
                image=os.path.basename(path), image_px=[Wc, Hc], scale=scale,
                um_per_px=[pitch[0] / cw, pitch[1] / ch], origin_square=[c0, r0])
    with open(os.path.join(out, 'selection.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1)
    if dxf:
        write_dxf(os.path.join(out, 'selection.dxf'), [(cols[i] - c0, rows[j] - r0, *g.name(i, j, num)) for i, j in sel],
                  pitch)
    fr.close()
    L.event('MSK-001', f"Export {path}: {len(sel)} squares, {Wc}x{Hc} px.")
    return Result("Exported", Squares=len(sel), **{"Image, px": f"{Wc} x {Hc}", "Pitch, um": f"{pitch[0]:.2f} x {pitch[1]:.2f}"},
                  File=path, DXF=os.path.join(out, 'selection.dxf') if dxf else '-')


def write_dxf(path, cells, pitch):
    """DXF R12: каждый квадрат — 4 линии на слое SQUARES, внешний контур выделения — слой OUTLINE,
    номер квадрата (X, Y — по подписям) — TEXT. Единицы — мкм, Y вверх (строки идут вниз)."""
    a, b = pitch
    have = {(c, r) for c, r, _, _ in cells}
    L_ = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "13", "0", "ENDSEC",   # 13 = микрометры
          "0", "SECTION", "2", "ENTITIES"]

    def line(x0, y0, x1, y1, layer):
        L_.extend(["0", "LINE", "8", layer, "10", f"{x0:.3f}", "20", f"{y0:.3f}", "30", "0",
                   "11", f"{x1:.3f}", "21", f"{y1:.3f}", "31", "0"])

    for c, r, i, j in cells:
        x0, x1, y0, y1 = c * a, (c + 1) * a, -r * b, -(r + 1) * b
        edges = {('t', c, r): (x0, y0, x1, y0), ('b', c, r): (x0, y1, x1, y1),
                 ('l', c, r): (x0, y0, x0, y1), ('r', c, r): (x1, y0, x1, y1)}
        for (side, _, _), e in edges.items():
            nb = {'t': (c, r - 1), 'b': (c, r + 1), 'l': (c - 1, r), 'r': (c + 1, r)}[side]
            line(*e, 'OUTLINE' if nb not in have else 'SQUARES')
        L_.extend(["0", "TEXT", "8", "LABELS", "10", f"{x0 + 0.05 * a:.3f}", "20", f"{y0 - 0.15 * b:.3f}",
                   "30", "0", "40", f"{0.08 * b:.3f}", "1", f"{i},{j}"])
    L_.extend(["0", "ENDSEC", "0", "EOF"])
    with open(path, 'w', encoding='ascii') as f:
        f.write("\n".join(L_) + "\n")


@_per_sample
def clean(folder, confirm=False):
    """Удалить рабочие файлы скана: кадры, пачки, превью-кэш, большие мозаики. Остаются meta.json, index.csv,
    grid.json, join/positions_all.csv, stitch.json и папка export/. confirm=False — только показать."""
    big = ['frames', 'batches', 'jpg', 'viewer']
    items = []
    for d in big:
        p = os.path.join(folder, d)
        if os.path.isdir(p):
            items.append(p)
    jd = os.path.join(folder, 'join')
    if os.path.isdir(jd):
        items += [os.path.join(jd, f) for f in os.listdir(jd) if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))]
    size = lambda p: sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs) \
        if os.path.isdir(p) else os.path.getsize(p)
    mb = sum(size(p) for p in items) / 2 ** 20
    if not os.path.isdir(os.path.join(folder, 'export')) and confirm:
        raise ValueError("No export/ in this scan yet: run mc.export(folder) first, then clean.")
    if not confirm:
        return Result("Would delete (run mc.clean(folder, confirm=True))", Items=len(items),
                      **{"Space, MB": round(mb, 1)}, List=", ".join(os.path.relpath(p, folder) for p in items) or "-")
    for p in items:
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    L.event('MSK-002', f"Cleaned {folder}: {len(items)} items, {mb:.1f} MB.")
    return Result("Cleaned", Items=len(items), **{"Freed, MB": round(mb, 1)})
