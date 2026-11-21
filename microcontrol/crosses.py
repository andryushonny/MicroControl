"""Решётка крестов на подложке: шаг, поворот вокруг Z и крайние кресты.

Кресты отмечаются в ручном режиме (клавиша P) — точка пересечения красных направляющих.
Все точки — в настоящих координатах столика, мкм (смещение нуля на них не влияет).
Модель: крест (i, j) находится в точке O + i·v1 + j·v2, где v1 — вектор к соседнему кресту
вдоль ряда (близко к оси X столика), v2 — вдоль столбца (близко к оси Y).
O, v1, v2 находятся методом наименьших квадратов по всем отметкам, поэтому дальние кресты
уточняют шаг и угол: ошибка угла ~ ошибка отметки / расстояние между отметками.
"""
import math
import numpy as np


def fit_square(ij, P, w=None):
    """МНК для квадратной решётки P ≈ O + i·a + j·rot90(a): 4 неизвестных (O, a) — сдвиг, шаг и наклон.
    Возвращает (O, a)."""
    i, j = ij[:, 0], ij[:, 1]
    n = len(P)
    M = np.zeros((2 * n, 4))
    M[0::2] = np.column_stack([np.ones(n), np.zeros(n), i, -j])
    M[1::2] = np.column_stack([np.zeros(n), np.ones(n), j, i])
    b = np.asarray(P, float).reshape(-1)
    if w is not None:
        ww = np.repeat(np.asarray(w, float), 2)
        M, b = M * ww[:, None], b * ww
    x = np.linalg.lstsq(M, b, rcond=None)[0]
    return x[:2], x[2:]


class Crosses:
    def __init__(self):
        self.marks, self.ij = [], []          # точки, мкм; их номера (i, j) в решётке
        self.O = self.v1 = self.v2 = None
        self.template, self.template_px = None, None   # картинка первого креста (для сшивки), мкм/px
        self.drift = np.zeros(2)                       # сдвиг решётки по найденным камерой крестам, мкм
        self.labels = {}                               # подписи крестов: (i, j) -> (a, b)

    # ---------- отметки ----------
    def add(self, p):
        """Добавить крест в точке p (мкм). Возвращает его номер (i, j)."""
        p = np.asarray(p, float)
        if not self.marks:
            ij = (0, 0)
        else:
            d = p - self.O
            if self.v1 is not None and self.v2 is not None:      # номер — по измеренной (повёрнутой) решётке
                i, j = np.linalg.solve(np.column_stack([self.v1, self.v2]), d)
                ij = (int(round(i)), int(round(j)))
            elif self.v1 is None and self.v2 is None:          # первый сосед задаёт первый вектор
                if abs(d[0]) >= abs(d[1]):
                    ij = (1 if d[0] > 0 else -1, 0)
                else:
                    ij = (0, 1 if d[1] > 0 else -1)
            else:                                               # один вектор известен
                known, idx = (self.v1, 0) if self.v1 is not None else (self.v2, 1)
                t = float(d @ known / (known @ known))
                n = d - round(t) * known
                if np.linalg.norm(n) < 0.5 * np.linalg.norm(known):
                    ij = (round(t), 0) if idx == 0 else (0, round(t))       # тот же ряд / столбец
                else:
                    other = 1 if n[1 - idx] > 0 else -1                     # соседний ряд / столбец
                    ij = (round(t), other) if idx == 0 else (other, round(t))
            if ij == (0, 0) or ij in self.ij:
                raise ValueError(f"This is the cross {ij} again. Mark another cross.")
        self.marks.append(p)
        self.ij.append(ij)
        self._fit()
        return ij

    def remove(self, k):
        """Убрать отметку k (остальные сохраняют свои номера)."""
        if 0 <= k < len(self.marks):
            self.marks.pop(k); self.ij.pop(k)
            self._fit()

    def undo(self):
        if self.marks:
            self.marks.pop(); self.ij.pop()
            self._fit()

    def clear(self):
        self.__init__()

    def live(self, i, j):
        """Где крест (i, j) сейчас: решётка + сдвиг по крестам, найденным камерой."""
        return self.point(i, j) + self.drift

    def _fit(self):
        """Решётка по отметкам — всегда квадратная: v2 = v1, повёрнутый на 90° (без перекоса и «проекции»).
        Две отметки (крест и сосед) уже задают наклон и шаг; дальние отметки их уточняют (МНК)."""
        if not self.marks:
            self.O = self.v1 = self.v2 = None
            return
        if len(self.marks) == 1:
            self.O, self.v1, self.v2 = np.array(self.marks[0], float), None, None
            return
        self.O, self.v1 = fit_square(np.array(self.ij, float), np.array(self.marks, float))
        self.v2 = np.array([-self.v1[1], self.v1[0]])

    # ---------- величины ----------
    def vectors(self, straight=False):
        """v1, v2 для рисования и номеров. straight=True — строго по осям столика (красная решётка ровная:
        подложку докручивают руками, пока кресты не лягут на неё); False — как измерено (с поворотом).
        Неизвестный вектор — перпендикуляр к известному той же длины (кресты ставят квадратом)."""
        v1, v2 = self.v1, self.v2
        if straight:
            v1 = None if v1 is None else np.array([math.copysign(np.linalg.norm(v1), v1[0]), 0.0])
            v2 = None if v2 is None else np.array([0.0, math.copysign(np.linalg.norm(v2), v2[1])])
        if v1 is None and v2 is None:
            return None, None
        if v1 is None:
            v1 = np.array([v2[1], -v2[0]])
        if v2 is None:
            v2 = np.array([-v1[1], v1[0]])
        return v1, v2

    def extent(self):
        """Номера крайних крестов: (i0, i1, j0, j1)."""
        ij = np.array(self.ij)
        return int(ij[:, 0].min()), int(ij[:, 0].max()), int(ij[:, 1].min()), int(ij[:, 1].max())

    def point(self, i, j):
        v1, v2 = self.vectors()
        return self.O + i * v1 + j * v2

    def angle(self):
        """Поворот решётки вокруг Z относительно осей столика, градусы (по ряду, иначе по столбцу)."""
        if self.v1 is not None:
            return math.degrees(math.atan2(self.v1[1], self.v1[0]))
        if self.v2 is not None:
            return math.degrees(math.atan2(-self.v2[0], self.v2[1]))
        return None

    def skew(self):
        """Отклонение угла между рядом и столбцом от 90°, градусы (если известны оба)."""
        if self.v1 is None or self.v2 is None:
            return None
        c = self.v1 @ self.v2 / np.linalg.norm(self.v1) / np.linalg.norm(self.v2)
        return 90.0 - math.degrees(math.acos(max(-1.0, min(1.0, c))))

    def residual(self):
        """Наибольшее отклонение отметок от решётки, мкм (точность отметок)."""
        if len(self.marks) < 2 or self.v1 is None and self.v2 is None:
            return None
        return max(float(np.linalg.norm(p - self.point(*ij))) for p, ij in zip(self.marks, self.ij))

    def summary(self):
        if not self.marks:
            return "Crosses: none. Put the red lines on a cross and press Enter"
        v1, v2 = self.vectors()
        if v1 is None:
            return "Crosses: 1 mark. Go to the next cross, put the lines on it, press Enter"
        i0, i1, j0, j1 = self.extent()
        a1 = f"{np.linalg.norm(v1):.1f}" + ("" if self.v1 is not None else "?")
        a2 = f"{np.linalg.norm(v2):.1f}" + ("" if self.v2 is not None else "?")
        r = self.residual()
        t = self.angle()
        return (f"Crosses: {len(self.marks)} marks   pitch {a1} x {a2} um   cells {i1 - i0} x {j1 - j0}"
                + (f"   marks off the grid up to {r:.1f} um" if r is not None else "")
                + f"   turn {t:+.3f} deg (the red grid follows it)")

    def to_dict(self):
        v1, v2 = self.vectors()
        d = dict(marks_um=[list(map(float, p)) for p in self.marks], index=[list(t) for t in self.ij],
                 drift_um=list(map(float, self.drift)),
                 labels=[[int(i), int(j), int(a), int(b)] for (i, j), (a, b) in self.labels.items()])
        if v1 is not None:
            i0, i1, j0, j1 = self.extent()
            d.update(origin_um=list(map(float, self.O)), v1_um=list(map(float, v1)), v2_um=list(map(float, v2)),
                     pitch_um=[float(np.linalg.norm(v1)), float(np.linalg.norm(v2))], turn_deg=self.angle(),
                     grid="turned: cross (i, j) at origin + i*v1 + j*v2 (stage um); turn_deg = turn of v1 from stage X",
                     skew_deg=self.skew(), extent=[i0, i1, j0, j1],
                     corners_um=[list(map(float, self.point(i, j))) for i, j in ((i0, j0), (i1, j0), (i1, j1), (i0, j1))])
        return d
