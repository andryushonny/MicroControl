"""Проекты, образцы, выделения, съёмки — объекты поверх папок (всё хранится на диске, объекты лёгкие).

    Проект (Project)        работа: несколько образцов, описание
      Образец (Sample)      одна подложка на столике: свои выделения и координаты, описание
        Выделение (Selection)   область скана с объективом (или draft); список плоский: порядок = слои
          Съёмка (Scan)     каждая съёмка выделения — новая версия (кадры, сшивка, сетка)

    <папка>/projects/<Проект>/project.json
    <папка>/projects/<Проект>/samples/<Образец>/sample.json          выделения образца (по порядку слоёв)
    <папка>/projects/<Проект>/samples/<Образец>/scans/<дата>_S03_50x/   съёмки (все — плоско)
    <папка>/projects/<Проект>/samples/<Образец>/export/              экспорт (чертёж, снимки, mixed.png)
    <папка>/projects/<Проект>/samples/<Образец>/thumbs/<uid>/        картинки углов выделений
    <папка>/projects/<Проект>/samples/<Образец>/photos/              отдельные снимки (Пробел)

    p = mc.Project("Graphene", "flakes on SiO2")    # открыть или создать; p.uid — короткий id ("k3f9x")
    p = mc.Project("k3f9x")                         # открыть имеющийся проект по id
    s = p.sample("Chip 1", "285 nm oxide")          # образец проекта: открыть или создать
    m = mc.Microscope(objective=10, sample=s)        # или потом: m.sample = s
    s.selections, s.scans(), s.selections[0].latest.view()
"""
import os
import re
import json
import glob
from datetime import datetime

import cv2
import numpy as np

ROOT = 'projects'


def safe(name, default='Untitled'):
    s = re.sub(r'[^\w\- ]+', '_', str(name or '')).strip(' _')
    return s or default


def root(base='.'):
    return os.path.join(os.path.abspath(base), ROOT)


def _now():
    return datetime.now().isoformat(timespec='seconds')


class _Node:
    """Папка с json-описанием (имя, описание, когда создана)."""
    FILE = 'node.json'

    def __init__(self, folder, name, description=None, create=True):
        self.folder = os.path.abspath(folder)
        self.file = os.path.join(self.folder, self.FILE)
        if os.path.exists(self.file):
            self.data = json.load(open(self.file, encoding='utf-8'))
        elif create:
            self.data = dict(name=name, description=description or '', created=_now())
            os.makedirs(self.folder, exist_ok=True)
            self.save()
        else:
            raise FileNotFoundError(f"No {type(self).__name__.lower()} '{name}' in {os.path.dirname(self.folder)}")
        if description is not None and description != self.data.get('description'):
            self.description = description

    name = property(lambda s: s.data.get('name') or os.path.basename(s.folder))

    @property
    def description(self):
        return self.data.get('description', '')

    @description.setter
    def description(self, text):
        self.data['description'] = str(text or '')
        self.save()

    def save(self):
        self.data['updated'] = _now()
        tmp = self.file + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=1, ensure_ascii=False, default=float)
        os.replace(tmp, self.file)

    def __eq__(self, o):
        return isinstance(o, _Node) and o.folder == self.folder

    def __hash__(self):
        return hash(self.folder)

    def __repr__(self):
        return f"{type(self).__name__} '{self.name}'" + (f" — {self.description}" if self.description else "")


UID_ABC = '23456789abcdefghjkmnpqrstuvwxyz'                          # без 0 1 i l o: не спутать


def by_uid(uid, folder='.'):
    """Папка проекта с коротким id uid (без учёта регистра) или None."""
    u = str(uid or '').strip().lower()
    if not re.fullmatch(r'[0-9a-z]{4,5}', u):
        return None
    for f in glob.glob(os.path.join(root(folder), '*', 'project.json')):
        try:
            if json.load(open(f, encoding='utf-8')).get('uid', '').lower() == u:
                return os.path.dirname(f)
        except (OSError, ValueError):
            pass
    return None


def _new_uid(folder):
    import secrets
    while True:
        u = ''.join(secrets.choice(UID_ABC) for _ in range(5))
        if by_uid(u, folder) is None:
            return u


class Project(_Node):
    """Проект: Project("Имя", "описание") — открыть или создать в <folder>/projects/.
    Project("k3f9x") — открыть уже имеющийся проект по его короткому id (p.uid, 5 символов)."""
    FILE = 'project.json'

    def __init__(self, name, description=None, folder='.', create=True):
        f_ = by_uid(name, folder) if not os.path.exists(os.path.join(root(folder), safe(name), self.FILE)) else None
        super().__init__(f_ or os.path.join(root(folder), safe(name)), name, description, create)
        self.base = os.path.abspath(folder)
        if not self.data.get('uid'):                                      # короткий id — один раз, навсегда
            self.data['uid'] = _new_uid(folder)
            self.save()

    uid = property(lambda s: s.data.get('uid'))

    @property
    def samples(self):
        """Образцы проекта (по времени создания)."""
        out = []
        for f in sorted(glob.glob(os.path.join(self.folder, 'samples', '*', Sample.FILE))):
            out.append(Sample(self, os.path.basename(os.path.dirname(f))))
        return sorted(out, key=lambda s: s.data.get('created', ''))

    def sample(self, name, description=None, create=True):
        """Образец проекта: открыть или создать."""
        return Sample(self, name, description, create)

    def __repr__(self):
        n = len(self.samples)
        return f"Project '{self.name}' [{self.uid}]" + (f" — {self.description}" if self.description else "") + \
            f"  ({n} sample{'s' if n != 1 else ''})"


class Sample(_Node):
    """Образец (подложка) проекта: свои выделения (sample.json), съёмки, снимки."""
    FILE = 'sample.json'

    def __init__(self, project, name, description=None, create=True):
        if isinstance(project, str):
            project = Project(project)
        self.project = project
        super().__init__(os.path.join(project.folder, 'samples', safe(name)), name, description, create)
        self.data.setdefault('selections', [])
        self.data.setdefault('seq', 0)

    photos = property(lambda s: os.path.join(s.folder, 'photos'))
    title = property(lambda s: f"{s.project.name} / {s.name}")

    @property
    def selections(self):
        """Выделения образца (как записаны: номер, объектив, углы, точки фокуса)."""
        return [Selection(self, d) for d in self.data.get('selections', [])]

    def selection(self, num):
        return next((s for s in self.selections if s.num == num), None)

    def scans(self, latest=False):
        """Съёмки образца: все версии (latest=True — последняя каждого выделения)."""
        return [Scan(f) for f in (latest_scans(self.folder) if latest else all_scans(self.folder))]

    def __repr__(self):
        return f"Sample '{self.title}'" + (f" — {self.description}" if self.description else "") + \
            f"  ({len(self.data.get('selections', []))} selections, {len(all_scans(self.folder))} scans)"


class Selection:
    """Выделение образца: номер, объектив, прямоугольник (настоящие мкм столика), съёмки (версии)."""

    def __init__(self, sample, data):
        self.sample, self.data = sample, data

    num = property(lambda s: s.data.get('num'))
    objective = property(lambda s: s.data.get('obj'))
    uid = property(lambda s: s.data.get('uid'))
    folder = property(lambda s: sel_folder(s.sample.folder, s.data))

    @property
    def scans(self):
        """Все съёмки выделения (старые -> новые)."""
        return [Scan(f) for f in scans_of(self.sample.folder, self.uid)]

    @property
    def latest(self):
        """Основная съёмка: выбранная (Scans -> Make main), иначе последняя."""
        s = self.scans
        mn = self.data.get('main')
        if mn:
            for x in s:
                if os.path.basename(x.folder) == mn:
                    return x
        return s[-1] if s else None

    def view(self):
        """Открыть последнюю съёмку в просмотре."""
        if self.latest is None:
            raise FileNotFoundError(f"Selection {self.num} is not scanned yet.")
        return self.latest.view()

    def __repr__(self):
        return f"Selection {self.num} ({self.objective}x, {len(self.scans)} scans)"


class Scan:
    """Съёмка выделения (папка с кадрами): stitch(), view(), export()."""

    def __init__(self, folder):
        self.folder = os.path.abspath(folder)

    @property
    def meta(self):
        return json.load(open(os.path.join(self.folder, 'meta.json'), encoding='utf-8'))

    def stitch(self, **kw):
        from .stitch import stitch
        return stitch(self.folder, **kw)

    def view(self, **kw):
        from .viewer import view
        return view(self.folder, **kw)

    def export(self, **kw):
        from .mask import export
        return export(self.folder, **kw)

    def __fspath__(self):
        return self.folder

    def __repr__(self):
        return f"Scan {os.path.basename(self.folder)}"


def _date(iso):
    """'2026-11-22T12:43:10' -> '2026-11-22 12:43'."""
    return (iso or '')[:16].replace('T', ' ') or '-'


def _table(cols, rows):
    """Таблица строками: cols — заголовки, rows — списки значений; ширина колонок по содержимому."""
    w = [max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows else len(str(c)) for i, c in enumerate(cols)]
    line = lambda r: "  ".join(str(v).ljust(w[i]) if i < len(r) - 1 else str(v) for i, v in enumerate(r))
    return line(cols), [line(r) for r in rows]


class _Listing(list):
    """Список проектов / образцов, который печатается таблицей (и в Jupyter — HTML-таблицей)."""
    COLS = ()

    def rows(self):
        return []

    def text(self):
        head, rows = _table(self.COLS, self.rows())
        return head, rows

    def __repr__(self):
        if not self:
            return "(none yet)"
        head, rows = self.text()
        return "\n".join([head, "-" * len(head)] + rows)

    def _repr_html_(self):
        import html
        cells = lambda r, tag: "".join(f"<{tag} style='text-align:left;padding:2px 10px'>{html.escape(str(v))}</{tag}>"
                                       for v in r)
        return ("<table><tr>" + cells(self.COLS, 'th') + "</tr>" +
                "".join("<tr>" + cells(r, 'td') + "</tr>" for r in self.rows()) + "</table>")


class ProjectList(_Listing):
    COLS = ("#", "Id", "Project", "Samples", "Selections", "Scans", "Created", "Updated", "Description")

    def rows(self):
        out = []
        for i, p in enumerate(self):
            ss = p.samples
            out.append([i + 1, p.uid, p.name, len(ss), sum(len(x.data.get('selections', [])) for x in ss),
                        sum(len(all_scans(x.folder)) for x in ss), _date(p.data.get('created')),
                        _date(max([p.data.get('updated', '')] + [x.data.get('updated', '') for x in ss])),
                        p.description])
        return out


class SampleList(_Listing):
    COLS = ("#", "Id", "Project / sample", "Selections", "Scans", "Created", "Updated", "Description")

    def rows(self):
        return [[i + 1, s.project.uid, s.title, len(s.data.get('selections', [])), len(all_scans(s.folder)),
                 _date(s.data.get('created')), _date(s.data.get('updated')), s.description]
                for i, s in enumerate(self)]


def projects(folder='.'):
    """Все проекты, новые сверху — таблицей: id, имя, образцы, выделения, съёмки, создан, изменён, описание."""
    out = [Project(os.path.basename(os.path.dirname(f)), folder=folder)
           for f in glob.glob(os.path.join(root(folder), '*', Project.FILE))]
    return ProjectList(sorted(out, key=lambda p: p.data.get('updated', ''), reverse=True))


def all_samples(folder='.'):
    """Все образцы всех проектов, недавние сверху — таблицей (как projects())."""
    out = [s for p in projects(folder) for s in p.samples]
    return SampleList(sorted(out, key=lambda s: s.data.get('updated', ''), reverse=True))


def sample_table(samples):
    """Для диалога Open: (заголовок, строки) — образцы с датами."""
    sl = samples if isinstance(samples, SampleList) else SampleList(samples)
    head, rows = _table(sl.COLS[1:], [r[1:] for r in sl.rows()])
    return head, rows


def last(folder='.'):
    """(проект, образец) последнего открытого образца или None."""
    try:
        pj, sm = open(os.path.join(root(folder), '.last'), encoding='utf-8').read().strip().split('\n')[:2]
        if os.path.exists(os.path.join(root(folder), pj, 'samples', sm, Sample.FILE)):
            return pj, sm
    except (OSError, ValueError):
        pass
    return None


def remember(sample):
    os.makedirs(root(sample.project.base), exist_ok=True)
    with open(os.path.join(root(sample.project.base), '.last'), 'w', encoding='utf-8') as f:
        f.write(f"{os.path.basename(sample.project.folder)}\n{os.path.basename(sample.folder)}\n")


# ------------------------------------------------------------------ выделения микроскопа <-> sample.json
def sel_folder(sample_folder, reg):
    """Папка мелочей выделения (картинки углов): thumbs/<uid>. Съёмки — в scans/ образца, плоско."""
    return os.path.join(sample_folder, 'thumbs', str(reg.get('uid') or f"S{reg.get('num', 0):02d}"))


def scan_uid(folder):
    """Выделение, которому принадлежит съёмка (uid из meta.json) или None."""
    try:
        return json.load(open(os.path.join(folder, 'meta.json'), encoding='utf-8')).get('sample_uid')
    except (OSError, ValueError):
        return None


def scan_label(folder):
    """«S03_50x» по имени папки съёмки (<дата>_S03_50x[_имя])."""
    m_ = re.search(r'_(S\d+_\d+x)', os.path.basename(folder))
    return m_.group(1) if m_ else os.path.basename(folder)


_UIDS = {}


def scans_of(sample_folder, uid):
    """Папки съёмок выделения uid (старые -> новые)."""
    return [f for f in all_scans(sample_folder) if _uid_cached(f) == uid]


def _uid_cached(f):
    mp = os.path.join(f, 'meta.json')
    try:
        mt = os.path.getmtime(mp)
    except OSError:
        return None
    c = _UIDS.get(f)
    if c is None or c[0] != mt:
        c = _UIDS[f] = (mt, scan_uid(f))
    return c[1]


_KEYS = ('num', 'uid', 'obj', 'fw', 'fh', 'A', 'B', 'pts', 'rank', 'shape', 'parent', 'kind', 'src', 'cells', 'factors', 'sub',
         'name', 'hidden', 'export')


def save_regions(m):
    """Выделения микроскопа -> sample.json открытого образца (и картинки углов). Пишет, только если изменилось."""
    sm = getattr(m, '_sample', None)
    if sm is None or getattr(m, '_regions', None) is None:
        return
    sels = [{k: r[k] for k in _KEYS if k in r} for r in m._regions]
    pend = {k: m._pending[k] for k in _KEYS if k in m._pending} if getattr(m, '_pending', None) else None
    new = json.loads(json.dumps(dict(selections=sels, pending=pend, seq=getattr(m, '_area_seq', 0)), default=float))
    if all(sm.data.get(k) == v for k, v in new.items()):
        return
    sm.data.update(new)
    sm.save()
    for r in m._regions:                                   # картинки углов — один раз
        th = r.get('thumbs') or []
        d = os.path.join(sel_folder(sm.folder, r), 'thumbs')
        if th and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            for i, t in enumerate(th):
                ok, buf = cv2.imencode('.jpg', t, [cv2.IMWRITE_JPEG_QUALITY, 85])
                buf.tofile(os.path.join(d, f"{i}.jpg"))


def load_regions(m):
    """sample.json -> выделения микроскопа (с картинками углов)."""
    sm = m._sample

    def fix(r):
        r = dict(r)
        r['A'] = tuple(r['A'])
        if 'B' in r:
            r['B'] = tuple(r['B'])
        d = os.path.join(sel_folder(sm.folder, r), 'thumbs')
        r['thumbs'] = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
                       for p in sorted(glob.glob(os.path.join(d, '*.jpg')))]
        return r
    m._regions = [fix(r) for r in sm.data.get('selections', [])]
    p = sm.data.get('pending')
    m._pending = fix(p) if p else None
    m._area_seq = int(sm.data.get('seq', 0))
    m._sel = len(m._regions) - 1 if m._regions else None


# ------------------------------------------------------------------ съёмки на диске
def is_scan(f):
    return os.path.exists(os.path.join(f, 'meta.json'))


def sample_of(folder):
    """Папка образца, к которому относится папка (сам образец, выделение или съёмка), или None."""
    f = os.path.abspath(folder)
    for _ in range(6):
        if os.path.exists(os.path.join(f, Sample.FILE)):
            return f
        f2 = os.path.dirname(f)
        if f2 == f:
            break
        f = f2
    return None


def sample_obj(folder):
    """Объект Sample по папке внутри образца (или None)."""
    sf = sample_of(folder)
    if not sf:
        return None
    pf = os.path.dirname(os.path.dirname(sf))
    return Sample(Project(os.path.basename(pf), folder=os.path.dirname(os.path.dirname(pf))), os.path.basename(sf))


def all_scans(sample_folder):
    """Все съёмки образца (scans/<дата>_Sxx_NNx), по времени."""
    return [s for s in sorted(glob.glob(os.path.join(sample_folder, 'scans', '*'))) if is_scan(s)]


def latest_scans(sample_folder, stitched=False):
    """Основная (Make main) или последняя съёмка каждого выделения из sample.json — в порядке списка (сверху вниз:
    верхний слой первым). Удалённые выделения не показываются (папки остаются: Ctrl+Z вернёт)."""
    sj = os.path.join(sample_folder, 'sample.json')
    try:
        regs = json.load(open(sj, encoding='utf-8')).get('selections') or []
    except (OSError, ValueError):
        regs = []
    by = {}
    for f in all_scans(sample_folder):
        by.setdefault(_uid_cached(f), []).append(f)
    out = []
    rk = lambda r: r.get('rank') if r.get('rank') is not None else -1e9
    for r in sorted(regs, key=lambda r: (rk(r), -r.get('num', 0))):
        v = by.get(r.get('uid')) or []
        if not v:
            continue
        mn = r.get('main')
        pick = [f for f in v if os.path.basename(f) == mn] if mn else []
        if stitched:                                       # показать: основная или самая новая СШИТАЯ (есть картинка)
            has = lambda f: os.path.exists(os.path.join(f, 'join', 'preview_all.jpg'))
            pick = [f for f in pick if has(f)] or [f for f in reversed(v) if has(f)][:1]
            if not pick:
                continue
        out.append(pick[0] if pick else v[-1])
    return out


def scan_folder(sample_folder, reg, name=''):
    """Новая папка съёмки выделения: scans/<дата>_<время>_S03_50x[_имя] (все съёмки образца — в одной папке)."""
    t = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    nm = re.sub(r'[^\w\-]+', '_', name or '').strip('_')
    base = os.path.join(sample_folder, 'scans')
    tag = f"S{reg['num']:02d}_{reg.get('obj', 0)}x"
    f = os.path.join(base, f"{t}_{tag}" + (f"_{nm}" if nm else ""))
    k = 1
    while os.path.exists(f):
        k += 1
        f = os.path.join(base, f"{t}_{tag}_{k}")
    return f
