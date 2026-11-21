"""Маска держится за настоящие кресты: живой кадр камеры сравнивается со сшитой мозаикой съёмки (там же,
где лежит маска), и сдвиг маски (grid.json, 'shift_um') поправляется так, чтобы квадраты легли на кресты.

    d, resp = offset(folder, img, P_um, K, step)     # поправка сдвига маски, мкм, и надёжность (0..1)

Кадр -> столик: S = P - step · K⁻¹ (u - c) (как live_map в окне), мозаика -> столик: аффинно (marked_squares).
Мозаика перерисовывается в геометрию кадра, затем фазовая корреляция (верхние частоты, окно Ханна).
"""
import numpy as np
import cv2

_SCENES = {}
F = 0.5             # масштаб сравнения (быстрее на старом ПК)
MIN_RESP = 0.12     # ниже — не доверять (пустое место, расфокус)


def _scene(folder):
    from .viewer import _Scene
    sc = _SCENES.get(folder)
    if sc is None:
        if len(_SCENES) > 4:
            _SCENES.pop(next(iter(_SCENES))).st.close()
        sc = _SCENES[folder] = _Scene(folder)
    return sc


def _hp(g):
    g = g.astype(np.float32)
    s = max(2.0, min(g.shape) / 60)
    return g - cv2.GaussianBlur(g, (0, 0), s)


def offset(folder, img, P_um, K, step):
    """Насколько поправить сдвиг маски съёмки folder по живому кадру img (камера стоит в P_um, настоящие мкм).
    -> (поправка (dx, dy) мкм, надёжность) или (None, причина)."""
    from .selections import marked_squares
    mk = marked_squares(folder)
    if not mk or mk.get('M') is None:
        return None, "no mask"
    M, sh = np.asarray(mk['M'], float), np.asarray(mk['shift_um'], float)
    ih, iw = img.shape[:2]
    c = np.array([iw / 2, ih / 2])
    Ki = np.linalg.inv(np.asarray(K, float))
    Mi = np.linalg.inv(M[:2])
    B = -step * Ki.T @ Mi                                   # кадр (строка u) -> мозаика: p = p0 + u @ B
    p0 = (np.asarray(P_um, float) - sh - M[2] + step * c @ Ki.T) @ Mi
    r = float(np.sqrt(abs(np.linalg.det(B))))               # пикселей мозаики на пиксель кадра
    if not 0.25 < r < 4:
        return None, "other magnification"
    sc = _scene(folder)
    cor = np.array([[0, 0], [iw, 0], [iw, ih], [0, ih]], float) @ B + p0
    x0, y0 = cor.min(0)
    x1, y1 = cor.max(0)
    if x1 < 0 or y1 < 0 or x0 > sc.size[0] or y0 > sc.size[1]:
        return None, "outside the scan"
    s = min(1.0, F / r)                                     # масштаб вырезки мозаики
    Wc, Hc = int((x1 - x0) * s) + 4, int((y1 - y0) * s) + 4
    crop = sc.render((x0 + x1) / 2, (y0 + y1) / 2, s, Wc, Hc)
    o = np.array([(x0 + x1) / 2 - Wc / 2 / s, (y0 + y1) / 2 - Hc / 2 / s])
    Bq = B * s / F                                          # кадр (уменьшенный в F) -> вырезка
    tq = (p0 - o) * s
    Mw = np.array([[Bq[0, 0], Bq[1, 0], tq[0]], [Bq[0, 1], Bq[1, 1], tq[1]]], np.float64)
    gs = cv2.resize(img, (int(iw * F), int(ih * F)), interpolation=cv2.INTER_AREA)
    gs = cv2.cvtColor(gs, cv2.COLOR_BGR2GRAY) if gs.ndim == 3 else gs
    cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ok = (cg > 20).astype(np.uint8)                         # вне мозаики — тёмный фон
    ren = cv2.warpAffine(cg, Mw, gs.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    val = cv2.warpAffine(ok, Mw, gs.shape[::-1], flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    if val.mean() < 0.5:
        return None, "the frame is mostly outside the scan"
    a, b = _hp(gs), _hp(ren)
    a[val == 0] = 0
    b[val == 0] = 0
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(b, a, win)          # кадр = перерисованная мозаика, сдвинутая на (dx, dy)
    if resp < MIN_RESP:
        return None, f"weak match ({resp:.2f})"
    du = np.array([dx, dy]) / F                             # пиксели полного кадра
    return -step * Ki @ du, float(resp)                     # что на картинке в u + du, маска думала — в u
