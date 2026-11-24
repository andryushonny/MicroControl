"""Разметка сшитого скана: решётка крестов, выделенные квадраты, вывод только выделенного, очистка.

Всё в «пикселях мозаики» — координатах окна просмотра (положения кадров после mc.stitch, начало в 0).
Решётка: крест (i, j) в точке O + i·v1 + j·v2. Квадрат (i, j) — между крестами i..i+1, j..j+1.
Файл <скан>/grid.json: отметки крестов, выделенные квадраты.
Вывод (mc.export): только выделенные квадраты, ровно (поворот снят), фон прозрачный (PNG) или белый (JPG);
DXF для AutoCAD — квадраты в мкм.
"""
from .samples import per_sample as _per_sample
import json
import time, math, os, shutil
import numpy as np
import cv2
from . import log as L
from .crosses import Crosses, fit_square
from .storage import ScanData
from .util import Result, Progress, save_image
from .stats import timed_fn


# ------------------------------------------------------------------ состояние разметки
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


KEEP = ('selected_removed', 'refine', 'seeded', 'seeded_labels', 'brush_t')         # ключи grid.json, которые пишет не только сетка

class Grid:
    def __init__(self, folder):
        self.folder = folder
        self.cr = Crosses()
        self.selected = set()
        self.labels = {}                                    # (i, j) -> (X, Y): номера, написанные литографией
        self.arm = None                                     # плечо креста, px мозаики (находится само)
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
            self.frames_px = np.array(d['frames_px'], float) if 'frames_px' in d else None

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
        if self.refined:
            d['refined'] = self.refined
        if self.frames_px is not None:
            d['frames_px'] = np.round(self.frames_px, 2).tolist()
        if v1 is not None:
            d.update(origin_px=list(map(float, self.cr.O)), v1_px=list(map(float, v1)), v2_px=list(map(float, v2)),
                     turn_deg=self.cr.angle())
            if px_um:
                d['pitch_um'] = [float(np.hypot(*v1) * px_um), float(np.hypot(*v2) * px_um)]
        gp_ = os.path.join(self.folder, 'grid.json')
        try:                                                # записи окна микроскопа: сдвиг маски, снятые части
            old_ = json.load(open(gp_, encoding='utf-8'))
            d.update({k_: old_[k_] for k_ in KEEP if k_ in old_})
            if sorted(map(list, old_.get('selected', []))) != d['selected']:
                d['brush_t'] = time.time()                  # здесь выбор поменяли: эта съёмка главная для кисти
            om_, nm_ = np.array(old_.get('marks', []), float), np.array(d['marks'], float)
            if d['marks'] and (om_.shape != nm_.shape or old_.get('index') != d['index']
                               or (om_.size and np.abs(om_ - nm_).max() > 0.5)):
                d['marks_t'] = time.time()                  # кресты отметили заново: решётка образца — по этой сетке
            elif 'marks_t' in old_:
                d['marks_t'] = old_['marks_t']
        except (OSError, ValueError):
            if d['selected']:
                d['brush_t'] = time.time()
            if d['marks']:
                d['marks_t'] = time.time()
        with open(gp_, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=1)

    def angle_deg(self):
        """Наклон креста (сетки) на мозаике, градусы, -45..45 (крест симметричен на 90°); нет сетки — 0."""
        if not self.ready():
            return 0.0
        v1, _ = self.cr.vectors()
        a = np.degrees(np.arctan2(v1[1], v1[0]))
        return float((a + 45) % 90 - 45)

    def snap_to_crosses(self, render, size, px_um, min_resp=0.1, render_s=None):
        """Сетка поставлена не по этим кадрам (по решётке образца через координаты столика): сдвинуть её на
        настоящие кресты этой мозаики. Каждый крест, куда попадает решётка, ищется по симметрии (до 16 узлов — быстро); сдвиг — медиана.
        Точно по всем крестам — F (refine) или сшивка по крестам. -> (найдено крестов, сдвиг px или None);
        self.sure_crosses — сколько из них точно кресты (отклик ≥ 0.8 и тот же сдвиг, ±2 мкм): пыль и края
        чешуек дают слабый отклик и случайный сдвиг; self.nodes_tried — сколько узлов решётки проверено.
        self.crosses_seen() — кресты здесь правда есть."""
        if not self.ready():
            return 0, None
        # окно ±30 мкм: сетка, перенесённая с другого объектива, может промахнуться на 10-15 мкм;
        # на 50x/100x крест крупнее перебора плеч (ARMS) — участок берётся уменьшенным в k раз
        k = min(1.0, 4.0 * px_um) if render_s is not None else 1.0
        r = int(min(150, max(8, 30.0 * k / px_um)))
        R0 = 10.0 / px_um                                     # узел не ближе 10 мкм к краю (окно может выйти за край)
        nodes = [self.cr.point(i, j) for i, j in self.cells_range((0, 0), size)]
        nodes = [q for q in nodes if R0 <= q[0] <= size[0] - R0 and R0 <= q[1] <= size[1] - R0]
        if len(nodes) > 16:                                   # быстро: до 16 узлов, разбросанных по мозаике
            c_ = np.asarray(size, float) / 2
            nodes.sort(key=lambda q: np.hypot(*(q - c_)))
            nodes = [nodes[int(t)] for t in np.linspace(0, len(nodes) - 1, 16)]
        ds, arms, rs = [], [], []
        self.sure_crosses, self.nodes_tried = 0, len(nodes)
        arm = None if k < 1 else self.arm
        for q in nodes:
            reg = render_s(q[0], q[1], k, 4 * r + 1, 4 * r + 1) if render_s is not None else \
                render(q[0], q[1], 4 * r + 1, 4 * r + 1)
            gq = meas_gray(reg)
            if gq.std() < 2:
                continue
            info = {}
            d, resp = snap_cross(gq, r, arm=arm, info=info, angle=self.angle_deg())
            if resp >= min_resp and np.any(d):
                ds.append(np.asarray(d, float) / k)
                rs.append(resp)
                if arm is None and info.get('arm'):
                    arms.append(info['arm'])
                    if len(arms) >= 3:                        # плечо найдено — дальше без перебора
                        arm = float(np.median(arms))
            if len(ds) >= 8:
                break
        if not ds:
            return 0, None
        D, rs = np.array(ds), np.array(rs)
        strong = rs >= 0.8
        if strong.any():                                      # сдвиг — по уверенным крестам, если они есть
            d = np.median(D[strong], axis=0)
            self.sure_crosses = int((strong & (np.hypot(*(D - d).T) <= 2.0 / px_um)).sum())
        else:
            d = np.median(D, axis=0)
        self.cr.marks = [np.asarray(p, float) + d for p in self.cr.marks]
        self.cr._fit()
        self.refined = None
        return len(ds), d

    def crosses_seen(self):
        """После snap_to_crosses: кресты в мозаике правда есть. Один уверенный крест — только когда проверено меньше
        4 узлов (маленький участок); иначе нужны два с одним сдвигом: случайный узор подложки или край кристалла
        изредка даёт сильный отклик в одном узле, но не два одинаково."""
        n_ = getattr(self, 'sure_crosses', 0) or 0
        return n_ >= (2 if (getattr(self, 'nodes_tried', 0) or 0) >= 4 else 1)

    def refine(self, render, size, px_um, min_resp=0.1, render_s=None, progress=None):
        """Уточнить решётку по ВСЕМ крестам мозаики: в каждом узле, куда попадает решётка, центр креста
        ищется по симметрии; затем O, v1, v2 — МНК по всем найденным (выбросы > 3σ отбрасываются).
        render(cx, cy, w, h) -> кусок мозаики в масштабе 1. Ошибка ручных отметок перестаёт копиться
        вдоль решётки. render_s(cx, cy, s, w, h) — то же в масштабе s: на 50x/100x крест крупнее перебора
        плеч, участок берётся уменьшенным. progress(доля 0..1, подпись) — для полосы в окне.
        Возвращает (число крестов, СКО отклонения от решётки, px)."""
        if not self.ready():
            return 0, None
        ang_ = self.angle_deg()
        ks = min(1.0, 4.0 * px_um) if render_s is not None else 1.0
        r = int(min(150, max(8, 15.0 * ks / px_um)))
        R0 = 2 * r / ks
        arm0 = self.arm
        if ks < 1:
            self.arm = self.arm * ks if self.arm else None
        cand = [(i, j) for i, j in self.cells_range((0, 0), size)
                if R0 <= self.cr.point(i, j)[0] <= size[0] - R0 and R0 <= self.cr.point(i, j)[1] <= size[1] - R0]
        nodes, regs = [], []
        for n_, (i, j) in enumerate(cand):
            if progress and n_ % 8 == 0:
                progress(0.8 * n_ / max(1, len(cand)), f"cross {n_ + 1} of {len(cand)}")
            q = self.cr.point(i, j)
            reg = render_s(q[0], q[1], ks, 4 * r + 1, 4 * r + 1) if ks < 1 else render(q[0], q[1], 4 * r + 1, 4 * r + 1)
            g = meas_gray(reg)
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
                    d, resp = snap_cross(regs[k], r, arm=L, angle=ang_, edges=False)
                    if resp >= min_resp and np.any(d):              # не нашёл (d = 0) — не считается
                        P_.append(nodes[k][1] + np.asarray(d, float) / ks); IJ_.append(nodes[k][0])
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
        if progress:
            progress(0.85, "fitting the grid")
        IJ, Pts = [], []
        for (ij, q), g in zip(nodes, regs):
            d, resp = snap_cross(g, r, arm=self.arm, angle=ang_)
            if resp >= min_resp:
                IJ.append(ij); Pts.append(q + np.asarray(d, float) / ks)
        if ks < 1:                                              # плечо — в px мозаики
            self.arm = (self.arm / ks) if self.arm else arm0
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


def meas_gray(img):
    """Картинка для измерений: зелёный канал (самый резкий у объектива, без цветной каймы на краях; у
    серой суммы каналов края размыты хроматизмом), float32."""
    img = np.asarray(img)
    return (img[..., 1] if img.ndim == 3 else img).astype(np.float32)


def _edges1d(p):
    """Два края полоски в профиле p (полоска светлее или темнее фона): (левый, правый, сила) с долями
    пикселя (парабола по 3 точкам производной) или None."""
    d = np.diff(p)
    if len(d) < 5:
        return None
    sgn = 1 if (p[len(p) // 2 - 1:len(p) // 2 + 2].mean() > 0.5 * (p[:2].mean() + p[-2:].mean())) else -1
    d = d * sgn                                          # теперь полоска светлая: слева рост, справа спад
    m = len(d) // 2
    a, b = int(np.argmax(d[:m + 1])), m + int(np.argmin(d[m:]))

    def sub(k):
        if 0 < k < len(d) - 1:
            y0, y1, y2 = d[k - 1], d[k], d[k + 1]
            den = y0 - 2 * y1 + y2
            return k + 0.5 + (0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0)
        return k + 0.5
    st = min(d[a], -d[b])
    if st <= 0:
        return None
    return sub(a), sub(b), float(st)


def edges_any(p, smooth=0.8):
    """Два края полоски где угодно в профиле p (полоска светлее или темнее фона): из самых сильных подъёмов
    и спадов производной — пара (левый < правый) с наибольшей слабейшей из двух силой. Доли пикселя —
    парабола по 3 точкам производной. -> (левый, правый) в индексах p или None."""
    p = np.asarray(p, np.float32)
    if len(p) < 6:
        return None
    if smooth:
        p = cv2.GaussianBlur(p.reshape(1, -1), (0, 0), smooth).ravel()
    d = np.diff(p)

    def peaks(v):
        k = np.where((v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]) & (v[1:-1] > 0))[0] + 1
        return k[np.argsort(-v[k])][:6]

    def sub(k):
        if 0 < k < len(d) - 1:
            y0, y1, y2 = d[k - 1], d[k], d[k + 1]
            den = y0 - 2 * y1 + y2
            return k + 0.5 + (0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0)
        return k + 0.5
    best = None
    for sg in (1, -1):                                   # светлая полоска: подъём, затем спад; тёмная — наоборот
        up, dn = peaks(sg * d), peaks(-sg * d)
        for a_ in up:
            for b_ in dn:
                if b_ - a_ >= 2:
                    st_ = min(sg * d[a_], -sg * d[b_])
                    if best is None or st_ > best[0]:
                        best = (st_, a_, b_)
    if best is None:
        return None
    return sub(best[1]), sub(best[2])


def _edge_center(g, c, L, t, ang=0.0):
    """Уточнение центра креста по краям плеч: поперёк каждого плеча — профили, в каждом два края полоски,
    середина между ними; по серединам горизонтальных плеч — одна прямая, вертикальных — другая; центр —
    их пересечение. g — окно (без вычитания фона), c — грубый центр (x, y), L, t — плечо и толщина, px,
    ang — наклон креста, рад. -> (центр, наклон рад) или (None, None)."""
    H_, W_ = g.shape
    gs = cv2.GaussianBlur(g, (0, 0), 0.7)
    h = int(max(3, 1.5 * t + 3))                          # полуширина профиля поперёк плеча
    out = []
    for vert in (False, True):
        a_ = ang + (np.pi / 2 if vert else 0.0)
        e = np.array([np.cos(a_), np.sin(a_)])           # вдоль плеча
        n = np.array([-e[1], e[0]])                      # поперёк
        pts, ws, wd = [], [], []
        np_ = int(np.clip(L // 3, 7, 40))                  # профилей на плечо: крупный крест — больше
        for s_ in np.r_[np.linspace(-0.92 * L, -0.4 * L, np_), np.linspace(0.4 * L, 0.92 * L, np_)]:
            q = np.asarray(c, float) + s_ * e
            off = np.arange(-h, h + 1, dtype=np.float32)
            xs = (q[0] + off * n[0]).astype(np.float32)
            ys = (q[1] + off * n[1]).astype(np.float32)
            if xs.min() < 1 or ys.min() < 1 or xs.max() > W_ - 2 or ys.max() > H_ - 2:
                continue
            prof = cv2.remap(gs, xs[None, :], ys[None, :], cv2.INTER_LINEAR)[0]
            r_ = _edges1d(prof)
            if r_ is None:
                continue
            l_, rr_, st = r_
            wdt = rr_ - l_
            if not (0.4 * t <= wdt <= 2.5 * t + 2):
                continue
            mid_ = 0.5 * (l_ + rr_) - h                      # смещение середины полоски поперёк, px
            pts.append(q + mid_ * n)
            ws.append(st)
            wd.append(wdt)
        if len(pts) < 6:
            return None, None
        pts, ws, wd = np.array(pts), np.array(ws), np.array(wd)
        ok = (ws > 0.3 * np.median(ws)) & (np.abs(wd - np.median(wd)) < max(1.0, 0.2 * np.median(wd)))   # цифры, грязь
        pts, ws = pts[ok], ws[ok]
        if len(pts) < 6:
            return None, None
        for _ in range(2):                               # прямая (МНК, вес — сила краёв), выбросы прочь
            m_ = (pts * ws[:, None]).sum(0) / ws.sum()
            C = ((pts - m_).T * ws) @ (pts - m_)
            w_, V = np.linalg.eigh(C)
            dvec = V[:, 1]
            res = (pts - m_) @ V[:, 0]
            keep = np.abs(res) < max(0.5, 3 * 1.48 * np.median(np.abs(res)))
            if keep.all() or keep.sum() < 6:
                break
            pts, ws = pts[keep], ws[keep]
        out.append((m_, dvec))
    (p1, d1), (p2, d2) = out
    A = np.c_[d1, -d2]
    if abs(np.linalg.det(A)) < 0.5:
        return None, None
    s1 = np.linalg.solve(A, p2 - p1)[0]
    cc = p1 + s1 * d1
    a1 = np.arctan2(d1[1], d1[0])
    a1 = (a1 + np.pi / 2) % np.pi - np.pi / 2            # наклон горизонтальной прямой, -90..90°
    return cc, a1


def cross_profiles(img, c, L, t, ang=0.0, h=None):
    """Профили яркости поперёк плеч креста (для окошка в просмотре): для горизонтальных и вертикальных плеч —
    среднее по профилям вдоль обоих плеч, по каждому цвету. h — полуширина профиля, px (по умолчанию 2t + 4).
    -> {'h'|'v': dict(x — смещение поперёк, px; prof — (n, 3) B, G, R; edges — (левый, правый) края по зелёному
    или None; e, n — направления вдоль и поперёк плеча)}."""
    img = np.asarray(img, np.float32)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    H_, W_ = img.shape[:2]
    h = int(max(4, h if h else 2.0 * t + 4))
    off = np.arange(-h, h + 1, dtype=np.float32)
    out = {}
    for key, vert in (('h', False), ('v', True)):
        a_ = ang + (np.pi / 2 if vert else 0.0)
        e = np.array([np.cos(a_), np.sin(a_)])
        n = np.array([-e[1], e[0]])
        acc, k = np.zeros((len(off), 3), np.float32), 0
        for s_ in np.r_[np.linspace(-0.85 * L, -0.45 * L, 8), np.linspace(0.45 * L, 0.85 * L, 8)]:
            q = np.asarray(c, float) + s_ * e
            xs = (q[0] + off * n[0]).astype(np.float32)[None, :]
            ys = (q[1] + off * n[1]).astype(np.float32)[None, :]
            if xs.min() < 1 or ys.min() < 1 or xs.max() > W_ - 2 or ys.max() > H_ - 2:
                continue
            acc += cv2.remap(img, xs, ys, cv2.INTER_LINEAR)[0]
            k += 1
        if not k:
            continue
        prof = acc / k
        ed = _edges1d(prof[:, 1])
        out[key] = dict(x=off, prof=prof, edges=(ed[0] - h, ed[1] - h) if ed else None, e=e, n=n)
    return out


def snap_cross(region, r, arm=None, info=None, angle=0.0, edges=True):
    """Центр креста в окне region (серое). Крест ищется шаблоном «+» (плечо arm, px; без arm — перебор),
    кандидат оценивается по симметрии креста при повороте на 90° и 180°: цифры у креста (номера литографии)
    так не симметричны и центр не уводят. Запасной вариант — корреляция окна с повёрнутым на 180°.
    angle — наклон креста, градусы (наклон сетки): окно сначала поворачивается так, чтобы крест стоял ровно.
    edges — затем центр уточняется по краям плеч (пересечение двух прямых, доли пикселя; _edge_center).
    Возвращает ((dx, dy) от середины окна, px; отклик 0…1). info (dict) получает плечо 'arm', 'angle'
    (наклон по краям, градусы) и 'edges' (уточнено ли)."""
    region = np.asarray(region, np.float32)
    mid = (np.array(region.shape[::-1], float) - 1) / 2
    a0 = float(angle or 0.0)
    if abs(a0) > 0.3:                                    # повернуть окно: крест — ровно
        Rm = cv2.getRotationMatrix2D((float(mid[0]), float(mid[1])), a0, 1.0)
        reg_r = cv2.warpAffine(region, Rm, region.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    else:
        Rm, reg_r = None, region
    d, resp, L, t = _snap_plain(reg_r, r, arm, info)
    if Rm is not None and np.any(d):                     # обратно в исходное окно
        Ri = cv2.invertAffineTransform(Rm)
        d = Ri[:, :2] @ (mid + d) + Ri[:, 2] - mid
    if info is not None:
        info['edges'] = False
    if edges and L and np.any(d) and resp > 0:
        th = np.radians(a0)
        c_ = mid + d
        for _ in range(2):                               # наклон по краям уточняется — второй проход ровнее
            cc, a1 = _edge_center(region, c_, L, t, th)
            if cc is None or np.hypot(*(cc - c_)) > max(1.5, 0.25 * t + 1):
                break
            c_, th = cc, a1
            if info is not None:
                info['edges'], info['angle'] = True, float(np.degrees(a1))
        d = c_ - mid
    return (d if np.hypot(*d) < r else np.zeros(2)), resp


def _snap_plain(region, r, arm=None, info=None):
    """snap_cross без поворота и краёв: (d, отклик, плечо, толщина)."""
    g0 = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 1.0)
    mid = (np.array(g0.shape[::-1], float) - 1) / 2
    cand = []
    for L in ([int(arm)] if arm else [a for a in ARMS if a <= 0.9 * r]):
        g = g0 - cv2.GaussianBlur(g0, (0, 0), max(4.0, L * 0.8))     # убрать фон своего размера: только детали
        for t in sorted({max(3, L // 2), max(3, L // 4), max(3, L // 5)}):
            T = _plus(L, t)
            if T.shape[0] + 2 >= min(g.shape):
                continue
            R = np.abs(cv2.matchTemplate(g, T, cv2.TM_CCOEFF_NORMED))
            yy_, xx_ = np.ogrid[0:R.shape[0], 0:R.shape[1]]  # центр — не дальше r от середины окна
            if not arm:
                R[np.hypot(xx_ + T.shape[1] // 2 - mid[0], yy_ + T.shape[0] // 2 - mid[1]) > r] = 0
            _, v, _, (px, py) = cv2.minMaxLoc(R)
            c = np.array([px + T.shape[1] // 2, py + T.shape[0] // 2], float)
            if 0 < px < R.shape[1] - 1 and 0 < py < R.shape[0] - 1:          # доля пикселя: центр масс 3x3
                w = R[py - 1:py + 2, px - 1:px + 2] - R[py - 1:py + 2, px - 1:px + 2].min()
                if w.sum() > 0:
                    c += (w.sum(0) @ [-1, 0, 1] / w.sum(), w.sum(1) @ [-1, 0, 1] / w.sum())
            cand.append((v * max(0.0, _sym(g, c, int(1.3 * L))), c, L, t))
    best = max(cand, key=lambda q: q[0]) if cand else (-1.0, None, None, None)
    if best[0] > 0:                                      # крест — самая крупная «+»: точки и пылинки мельче
        best = max((q for q in cand if q[0] >= 0.9 * best[0]), key=lambda q: (q[2], q[0]))
    sc, c, L, t = best
    if c is None or sc < (0.15 if not arm else 0.03):                        # шаблон не нашёл — старый способ (симметрия 180°)
        g = g0 - cv2.GaussianBlur(g0, (0, 0), max(4.0, (arm or r / 3) * 0.8))
        win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        (sx, sy), resp = cv2.phaseCorrelate(g * win, np.ascontiguousarray(g[::-1, ::-1]) * win)
        d = -np.array([sx, sy]) / 2.0                    # повёрнутое окно сдвинуто на -2·d
        if info is not None:
            info['arm'] = None
        return (d if np.hypot(*d) < r else np.zeros(2)), resp, None, None
    if info is not None:
        info['arm'], info['thick'] = L, t
    d = c - mid
    return (d if np.hypot(*d) < r else np.zeros(2)), float(min(1.0, sc * 3)), L, t


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

    def remap(self, mapx, mapy, bg=255, tile=1024):
        """Картинка, где пиксель (u, v) берётся из точки мозаики (mapx[v, u], mapy[v, u]) — любое (не только
        аффинное) отображение: по клеткам решётки от её 4 крестов. Смешивание — весами к краю кадра.
        -> (BGR uint8, покрыто кадрами)."""
        h, w = mapx.shape
        out = np.full((h, w, 3), bg, np.uint8)
        cov = np.zeros((h, w), bool)
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                mx = mapx[y0:y0 + tile, x0:x0 + tile]
                my = mapy[y0:y0 + tile, x0:x0 + tile]
                lo = np.array([np.nanmin(mx), np.nanmin(my)])
                hi = np.array([np.nanmax(mx), np.nanmax(my)])
                hit = np.flatnonzero((self.P[:, 0] < hi[0]) & (self.P[:, 0] + self.W > lo[0]) &
                                     (self.P[:, 1] < hi[1]) & (self.P[:, 1] + self.H > lo[1]))
                acc = np.zeros(mx.shape + (3,), np.float32)
                ws = np.zeros(mx.shape, np.float32)
                for k in hit:
                    img = self.st.frame(int(k))
                    img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    lx = (mx - self.P[k, 0]).astype(np.float32)
                    ly = (my - self.P[k, 1]).astype(np.float32)
                    t = cv2.remap(img, lx, ly, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
                    f = cv2.remap(self.feather, lx, ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    acc += t * f[..., None]
                    ws += f
                ok = ws > 1e-3
                blk = np.where(ok[..., None], acc / np.maximum(ws, 1e-6)[..., None], bg)
                out[y0:y0 + tile, x0:x0 + tile] = np.clip(blk, 0, 255).astype(np.uint8)
                cov[y0:y0 + tile, x0:x0 + tile] = ok
        return out, cov

    def close(self):
        self.st.close()


def measure_crosses(folder, g, nodes, fr=None):
    """Кресты сетки g в узлах nodes [(i, j)] на мозаике скана — точно (шаблон + края плеч, с наклоном
    сетки). -> {(i, j): (x, y) px мозаики}; не нашёлся или далеко от сетки (> 3 px и 0.5 мкм) — нет в словаре."""
    from .storage import ScanData as _SD
    own = fr is None
    fr = fr or Frames(folder)
    px = fr.st.meta.get('pixel_size_um') or 1.0
    ks = min(1.0, 4.0 * px)                              # 50x, 100x: крест крупнее перебора — уменьшить окно
    r = int(min(150, max(8, 15.0 * ks / px)))
    ang = g.angle_deg()
    arm = (g.arm * ks) if g.arm else None
    out = {}
    try:
        for i, j in nodes:
            q = g.cr.point(i, j)
            n = 4 * r + 1
            M = np.array([[1 / ks, 0, q[0] - (n - 1) / 2 / ks], [0, 1 / ks, q[1] - (n - 1) / 2 / ks]])
            reg, cov = fr.warp(M, (n, n), 0, blend='feather')
            if cov.mean() < 0.9:
                continue
            gq = meas_gray(reg)
            if gq.std() < 2:
                continue
            info = {}
            d, resp = snap_cross(gq, r, arm=arm, info=info, angle=ang)
            if resp < 0.1 or not np.any(d):
                continue
            if arm is None and info.get('arm'):
                arm = info['arm']
            p_ = q + np.asarray(d, float) / ks
            if np.hypot(*(p_ - q)) > max(3.0, 0.5 / px) * 3:    # не тот крест
                continue
            out[(int(i), int(j))] = p_
    finally:
        if own:
            fr.close()
    return out


@_per_sample
@timed_fn('Export (all of it)')
def export(folder, fmt='png', scale=1.0, cells=False, dxf=True, max_mb=2000, margin=0, blend='multiband', dwg=True):
    """Вывести только выделенные квадраты: ровно (поворот подложки снят), стык строго по крестам.
    fmt='png' — фон прозрачный, 'jpg' — белый. cells=True — ещё и каждый квадрат отдельным файлом.
    dxf=True — export/selection.dxf: квадраты в мкм (AutoCAD).
    dwg=True — итоговый чертёж: export/drawing.dwg (и drawing.dxf) — каждый квадрат отдельной картинкой
    (export/squares/*.png) на своём месте в мкм, контуры квадратов и номера по слоям; поверх рисуют бонды.
    DWG пишет бесплатный ODA File Converter (если установлен), иначе — drawing.dxf (AutoCAD: «Сохранить как» DWG).
    Возвращает Result."""
    g = Grid(folder)
    if not g.ready() or not g.selected:
        raise ValueError("Nothing to export: in mc.view(folder) mark the crosses (C) and select squares (M).")
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
    cells = cells or dwg                                      # для чертежа нужен каждый квадрат отдельно
    if cells:
        os.makedirs(os.path.join(out, 'squares'), exist_ok=True)
    sq_files = {}
    num = g.numbering()
    pr = Progress(len(sel), "Export: squares", unit='squares')
    for i, j in sel:
        X_, Y_ = g.name(i, j, num)
        pr.update(1, f"square X {X_:+d}, Y {Y_:+d}")
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
            sq_files[(i, j)] = f"square_X{X:+04d}_Y{Y:+04d}.png"
            buf.tofile(os.path.join(out, 'squares', sq_files[(i, j)]))
    pr.close()
    path = os.path.join(out, 'selection.' + ('jpg' if jpg else 'png'))
    save_image(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95] if jpg else [cv2.IMWRITE_PNG_COMPRESSION, 6],
               "Export: saving")
    mp_ = float(fr.st.meta.get('mask_pitch') or 0)
    pitch = (mp_, mp_) if mp_ else (a * px_um, b * px_um)   # шаг маски (точно), иначе по пикселю
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
    r = Result("Exported", Squares=len(sel), **{"Image, px": f"{Wc} x {Hc}", "Pitch, um": f"{pitch[0]:.2f} x {pitch[1]:.2f}"},
               File=path, DXF=os.path.join(out, 'selection.dxf') if dxf else '-')
    if dwg:
        with Progress(3, "Export: drawing", unit='steps') as pr2:
            pr2.stage("drawing.dxf: square pictures, outlines, numbers")
            cells_ = [(cols[i] - c0, rows[j] - r0, *g.name(i, j, num), sq_files[(i, j)]) for i, j in sel]
            dp = write_drawing(os.path.join(out, 'drawing.dxf'), cells_, pitch, (cw, ch))
            pr2.stage("drawing.dwg (ODA File Converter)", 1)
            if dp:
                r['Drawing'] = dp
                dw = dxf_to_dwg(dp)
                r['DWG'] = dw or ("not made: install the free ODA File Converter (opendesign.com), or open "
                                  "drawing.dxf in AutoCAD and save as DWG")
            else:
                r['Drawing'] = "not made: pip install ezdxf"
            pr2.stage("done", 2)
    L.event('MSK-001', f"Export {path}: {len(sel)} squares, {Wc}x{Hc} px.")
    return r


def write_drawing(path, cells, pitch, px):
    """Итоговый чертёж (DXF 2010, единицы — мкм, Y вверх): каждый квадрат — своя картинка (слой IMAGES,
    файл рядом: squares/...), его 4 стороны (SQUARES, каждый квадрат целиком), внешний край выделения (OUTLINE —
    те же грани ещё раз: общая грань есть на каждом слое) и номер (LABELS).
    cells: [(столбец, строка, X, Y, файл)], pitch — шаг квадрата, мкм; px — размер картинки квадрата, px.
    Картинки в DWG всегда внешние файлы: папку export/ переносить целиком."""
    try:
        import ezdxf
    except ImportError:
        return None
    a, b = pitch
    doc = ezdxf.new('R2010', setup=True)
    doc.units = 13                                            # микрометры
    doc.header['$INSUNITS'] = 13
    for name, color in (('IMAGES', 7), ('SQUARES', 3), ('OUTLINE', 1), ('LABELS', 2), ('BONDS', 6)):
        if name not in doc.layers:
            doc.layers.add(name, color=color)
    msp = doc.modelspace()
    have = {(c, r) for c, r, *_ in cells}
    for c, r, X, Y, fn in cells:
        x0, y1 = c * a, -r * b                                # верх квадрата; строки идут вниз
        y0 = y1 - b
        idef = doc.add_image_def(filename=os.path.join('squares', fn), size_in_pixel=px)
        msp.add_image(idef, insert=(x0, y0), size_in_units=(a, b), dxfattribs={'layer': 'IMAGES'})
        # каждый квадрат — целиком на SQUARES (все 4 стороны, даже общие с соседом или с краем выделения);
        # край выделения — ещё и на OUTLINE: общая грань есть на каждом слое, которому она принадлежит
        msp.add_lwpolyline([(x0, y0), (x0 + a, y0), (x0 + a, y1), (x0, y1)], close=True,
                           dxfattribs={'layer': 'SQUARES'})
        for (xa, ya, xb, yb), nb in (((x0, y1, x0 + a, y1), (c, r - 1)), ((x0, y0, x0 + a, y0), (c, r + 1)),
                                     ((x0, y0, x0, y1), (c - 1, r)), ((x0 + a, y0, x0 + a, y1), (c + 1, r))):
            if nb not in have:
                msp.add_line((xa, ya), (xb, yb), dxfattribs={'layer': 'OUTLINE'})
        msp.add_text(f"{X},{Y}", height=0.06 * b,
                     dxfattribs={'layer': 'LABELS', 'insert': (x0 + 0.04 * a, y1 - 0.1 * b)})
    doc.saveas(path)
    return path


def dxf_to_dwg(path):
    """DXF -> DWG бесплатным ODA File Converter (если установлен; путь можно задать в MC_ODA). DWG или None."""
    import glob
    import shutil
    import subprocess
    import tempfile
    exe = os.environ.get('MC_ODA') or shutil.which('ODAFileConverter')
    if not exe:
        for pat in (r"C:\Program Files\ODA\ODAFileConverter*\ODAFileConverter.exe",
                    r"C:\Program Files (x86)\ODA\ODAFileConverter*\ODAFileConverter.exe",
                    "/usr/bin/ODAFileConverter*", "/opt/ODAFileConverter*/ODAFileConverter"):
            hits = sorted(glob.glob(pat))
            if hits:
                exe = hits[-1]
                break
    if not exe:
        return None
    src = tempfile.mkdtemp()
    shutil.copy(path, src)
    dst = os.path.dirname(os.path.abspath(path))
    try:
        subprocess.run([exe, src, dst, "ACAD2013", "DWG", "0", "1", "*.DXF"], check=True, timeout=300,
                       capture_output=True)
    except Exception:
        return None
    dw = os.path.splitext(os.path.abspath(path))[0] + '.dwg'
    return dw if os.path.exists(dw) else None


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
