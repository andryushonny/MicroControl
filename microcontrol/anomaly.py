"""Жёлтая подсказка через поиск аномалий (принцип PaDiM): «нормальная» подложка описывается гауссовым
распределением признаков маленьких участков (патчей), всё, что далеко от него по Махаланобису, — аномалия:
чешуйки, кристаллы, любые предметы. Обучение без разметки: на самой мозаике (почти все квадраты пустые,
выбросы отбрасываются) или на скане пустой подложки (модель сохраняется в файл и используется для других сканов).

Для старого ПК: вся мозаика обрабатывается одним уменьшенным снимком (≤ ~6 Мпикс), признаки считаются
по одному каналу и сразу усредняются по патчам 4 × 4 px — пик памяти ~100–150 МБ, время — секунды.

Признаки патча (по умолчанию, без нейросети, D = 8): отклонение цвета от локального фона в Lab (3; неровная
подсветка убрана), контраст краёв и текстуры (лапласиан, градиент, локальное СКО яркости, разность гауссиан —
4), насыщенность отклонения (1). С файлом нейросети (ONNX, например ResNet18 до layer2) признаки берутся
из неё — это настоящий PaDiM; см. mc.anomaly.BACKBONE.
"""
import json
import os
import numpy as np
import cv2

PATCH = 4                    # патч, px уменьшенной мозаики
BACKBONE = os.path.join(os.path.expanduser('~'), '.microcontrol', 'backbone.onnx')   # необязательная нейросеть
MODEL_FILE = 'hint_model.npz'


# ------------------------------------------------------------------ признаки
def _pool(ch, p=PATCH):
    h, w = ch.shape[:2]
    return cv2.resize(ch, (max(1, w // p), max(1, h // p)), interpolation=cv2.INTER_AREA)


def features(img, bg_sigma, net=None):
    """img — BGR uint8 (уменьшенная мозаика). Возвращает (Hp, Wp, D) float32 по патчам PATCH × PATCH."""
    if net is not None:
        return _cnn_features(img, net)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    out = []
    dev = []
    for c in range(3):                                    # по одному каналу: мало памяти
        x = lab[..., c].astype(np.float32)
        x = cv2.GaussianBlur(x, (0, 0), 0.8)
        d = x - cv2.GaussianBlur(x, (0, 0), bg_sigma)     # отклонение от локального фона (подсветка)
        if c == 0:
            L = x
        dev.append(_pool(d))
        out.append(dev[-1])
        del x, d
    out.append(np.hypot(dev[1], dev[2]))                  # насыщенность отклонения
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(L, (0, 0), 1.0), cv2.CV_32F, ksize=3))
    out.append(_pool(lap)); del lap
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.magnitude(gx, gy); del gx, gy
    out.append(_pool(g)); del g
    m1 = cv2.blur(L, (5, 5))
    sd = np.sqrt(np.maximum(cv2.blur(L * L, (5, 5)) - m1 * m1, 0)); del m1
    out.append(_pool(sd)); del sd
    dog = cv2.GaussianBlur(L, (0, 0), 2.0) - cv2.GaussianBlur(L, (0, 0), 4.0)
    out.append(_pool(np.abs(dog))); del dog
    h = min(o.shape[0] for o in out)
    w = min(o.shape[1] for o in out)
    return np.stack([o[:h, :w] for o in out], -1).astype(np.float32)


def load_backbone(path=None):
    """Нейросеть-извлекатель признаков (ONNX) через cv2.dnn, если файл есть; иначе None."""
    p = path or BACKBONE
    if not p or not os.path.exists(p):
        return None
    meta = p[:-5] + '.json'
    return dict(net=cv2.dnn.readNetFromONNX(p), cfg=json.load(open(meta)) if os.path.exists(meta) else {})


def _cnn_features(img, net, tile=256):
    """Признаки из нейросети по плиткам (не держим всю мозаику в сети). Выходы сети усредняются до сетки
    патчей и проецируются случайной фиксированной матрицей до 32 признаков (как в PaDiM)."""
    cfg, net = net['cfg'], net['net']
    mean = np.array(cfg.get('mean', [0.485, 0.456, 0.406]), np.float32)
    std = np.array(cfg.get('std', [0.229, 0.224, 0.225]), np.float32)
    H, W = img.shape[:2]
    Hp, Wp = H // PATCH, W // PATCH
    feats, proj = None, None
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            t = img[y0:y0 + tile, x0:x0 + tile]
            th, tw = t.shape[:2]
            if th < 16 or tw < 16:
                continue
            pad = np.zeros((tile, tile, 3), np.uint8)
            pad[:th, :tw] = t
            blob = (cv2.cvtColor(pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - mean) / std
            net.setInput(blob.transpose(2, 0, 1)[None])
            outs = net.forward(net.getUnconnectedOutLayersNames())
            ph, pw = tile // PATCH, tile // PATCH
            maps = [cv2.resize(o[0].transpose(1, 2, 0), (pw, ph), interpolation=cv2.INTER_LINEAR).reshape(ph, pw, -1)
                    for o in outs]
            f = np.concatenate(maps, -1)
            if proj is None:
                rng = np.random.default_rng(0)
                proj = (rng.standard_normal((f.shape[-1], min(32, f.shape[-1]))) / np.sqrt(f.shape[-1])).astype(np.float32)
                feats = np.zeros((Hp, Wp, proj.shape[1]), np.float32)
            f = f @ proj
            hh, ww = min(th // PATCH, Hp - y0 // PATCH), min(tw // PATCH, Wp - x0 // PATCH)
            feats[y0 // PATCH:y0 // PATCH + hh, x0 // PATCH:x0 // PATCH + ww] = f[:hh, :ww]
    return feats


# ------------------------------------------------------------------ гауссова модель «нормы»
class Normal:
    """«Норма» — смесь нескольких гауссиан признаков пустой подложки (у подложки бывает несколько видов
    участков: чистый, с точкой разметки, с краем креста). Расстояние патча — до ближайшей гауссианы
    (Махаланобис), в единицах разброса нормы."""

    def __init__(self, means=None, icovs=None, scale=None, kind='hand'):
        self.means, self.icovs, self.scale, self.kind = means, icovs, scale, kind

    @classmethod
    def fit(cls, X, k=3, trim=0.12, iters=1, kind='hand'):
        """Устойчиво: подгонка (EM, OpenCV), отбросить самые далёкие trim (там и есть «что-то»), снова."""
        X = np.asarray(X, np.float32)
        rng = np.random.default_rng(0)
        if len(X) > 12000:                                # для скорости и памяти — случайная выборка
            X = X[rng.choice(len(X), 12000, replace=False)]
        sd = X.std(0) + 1e-6                              # масштаб признаков — для устойчивой подгонки
        keep = np.ones(len(X), bool)
        for _ in range(iters + 1):
            means, icovs = _gmm(X[keep] / sd, k)
            mdl = cls(means * sd, [ic / np.outer(sd, sd) for ic in icovs], (0.0, 1.0), kind)
            d = mdl._raw(X)
            keep = d <= np.quantile(d, 1 - trim)
        d = d[keep]
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med))) * 1.4826 + 1e-6
        mdl.scale = (med, mad)
        return mdl

    def _raw(self, X):
        d = None
        for mu, ic in zip(self.means, self.icovs):
            di = _maha(X, mu.astype(np.float32), ic.astype(np.float32))
            d = di if d is None else np.minimum(d, di)
        return d

    def distance(self, F):
        """Расстояние до нормы для карты признаков (Hp, Wp, D)."""
        h, w, D = F.shape
        flat = F.reshape(-1, D)
        d = np.empty(len(flat), np.float32)
        for i in range(0, len(flat), 65536):                # кусками: мало памяти
            d[i:i + 65536] = self._raw(flat[i:i + 65536])
        med, mad = self.scale
        return ((d - med) / mad).reshape(h, w)

    def save(self, path):
        np.savez(path, means=np.array(self.means), icovs=np.array(self.icovs), scale=np.array(self.scale),
                 kind=self.kind)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(list(z['means']), list(z['icovs']), tuple(z['scale']), str(z['kind']))


def _gmm(X, k):
    """Смесь k гауссиан (OpenCV EM); если не сошлось — одна гауссиана. Возвращает (средние, обратные ковариации)."""
    D = X.shape[1]
    reg = np.eye(D) * 1e-3
    try:
        em = cv2.ml.EM_create()
        em.setClustersNumber(k)
        em.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_GENERIC)
        em.setTermCriteria((cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 1e-3))
        em.trainEM(X.astype(np.float64))
        w = em.getWeights().ravel()
        means = [m for m, wi in zip(em.getMeans(), w) if wi > 0.02]
        covs = [c for c, wi in zip(em.getCovs(), w) if wi > 0.02]
        if not means:
            raise ValueError
    except Exception:
        means, covs = [X.mean(0)], [np.cov(X, rowvar=False)]
    return [np.asarray(m, np.float32) for m in means], [np.linalg.inv(c + reg).astype(np.float32) for c in covs]


def _maha(X, mu, ic):
    Y = X - mu
    return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', Y, ic, Y), 0))


# ------------------------------------------------------------------ карта аномалий
THRESH = 8.0                 # порог патча: столько «разбросов нормы» от середины


def anomaly_mask(F, model, valid, thresh=THRESH, dust=True, min_patches=2):
    """Маска аномальных патчей. valid — патчи над сканом и не у крестов. dust=True: маленькие пятна
    с тёмной бесцветной серединкой (пыль, точки) не считаются. Мелочь меньше min_patches патчей отбрасывается.
    Возвращает (маска, расстояния z)."""
    z = model.distance(F)
    v = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT,
                  borderValue=0) > 0                                   # края скана и у крестов — нет
    m = ((z > thresh) & v).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    area = st[:, cv2.CC_STAT_AREA]
    ok = np.zeros(n, bool)
    ok[1:] = area[1:] >= min_patches
    if dust and model.kind == 'hand' and n > 1:
        dL, chroma = F[..., 0], F[..., 3]
        core = (dL < -6) & (chroma < 0.4 * np.abs(dL))                    # тёмное бесцветное
        has_core = np.bincount(lab[core], minlength=n) > 0
        bright = np.bincount(lab.ravel(), weights=(dL > 6).ravel().astype(float), minlength=n)
        small = area < 40
        ok &= ~(has_core & small & (bright < 0.25 * area))                # пыль: тёмное ядро, мало, не светлое
    return ok[lab], z


# ------------------------------------------------------------------ модель пустой подложки
def default_model(objective=None):
    """Файл модели пустой подложки для этого объектива (если его сделали mc.train_hints(..., default=True))."""
    p = os.path.join(os.path.expanduser('~'), '.microcontrol', f"hint_model_{objective}x.npz")
    return p if objective and os.path.exists(p) else None


def export_backbone(path=None):
    """Сделать файл нейросети для подсказок (настоящий PaDiM): ResNet18, обученная на ImageNet, до layer2.
    Нужны torch и torchvision и интернет — один раз на любом ПК; файл (~11 МБ) скопировать на ПК микроскопа
    в ~/.microcontrol/backbone.onnx. На самом ПК микроскопа torch не нужен: сеть запускает OpenCV."""
    import torch
    import torchvision
    path = path or BACKBONE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    net = torchvision.models.resnet18(weights='DEFAULT').eval()

    class Head(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            m = self.m
            f1 = m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x)))))
            return f1, m.layer2(f1)
    torch.onnx.export(Head(net), torch.zeros(1, 3, 256, 256), path, output_names=['layer1', 'layer2'], opset_version=13)
    json.dump(dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), open(path[:-5] + '.json', 'w'))
    return path
