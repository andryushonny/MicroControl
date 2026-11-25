"""Простые окна для меню (tkinter): новый проект, открыть проект, описание проекта, выбрать съёмку.
Каждое окно модальное и поверх остальных; закрыли — None."""


BG, BG2, FG, DIM_, LINE_ = '#1f1b1a', '#2b2a30', '#eeeeee', '#a5a5a5', '#58504e'   # как в окнах (ui.PANEL, ui.TEXT ...)
ACC, ACC_FG = '#ffcd00', '#1a1a1a'                       # жёлтый акцент (кнопка по умолчанию, выбранная строка)
_FONT_LOADED = []


def _family():
    """JetBrains Mono (microcontrol/fonts) для окон tkinter: в Windows — подгрузить в процесс; иначе — что есть."""
    import os
    if not _FONT_LOADED:
        _FONT_LOADED.append(True)
        if os.name == 'nt':
            try:
                import ctypes
                d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
                for f in ('JetBrainsMono-Regular.ttf', 'JetBrainsMono-Bold.ttf'):
                    ctypes.windll.gdi32.AddFontResourceExW(os.path.join(d, f), 0x10, 0)   # FR_PRIVATE
            except Exception:
                pass
    try:
        import tkinter.font as tkf
        fams = set(tkf.families())
    except Exception:
        fams = set()
    for f in ('JetBrains Mono', 'Consolas', 'DejaVu Sans Mono', 'Courier New'):
        if f in fams:
            return f
    return 'TkFixedFont'


def _root():
    try:
        import tkinter as tk
    except ImportError:
        raise RuntimeError("tkinter is not installed (it comes with the usual Python for Windows). Without it use "
                           "the code: m.sample = mc.Project('name').sample('name')") from None
    r = tk.Tk()
    r.withdraw()
    try:
        r.attributes('-topmost', True)
    except Exception:
        pass
    fam = _family()                                      # тёмная тема и шрифт — как в окнах программы
    r._fam = fam
    for k, v in (('*Background', BG), ('*Foreground', FG), ('*Font', (fam, 10)), ('*highlightThickness', 0),
                 ('*Entry.Background', BG2), ('*Entry.insertBackground', FG), ('*Entry.relief', 'flat'),
                 ('*Text.Background', BG2), ('*Text.insertBackground', FG), ('*Text.relief', 'flat'),
                 ('*Listbox.Background', BG2), ('*Listbox.selectBackground', ACC), ('*Listbox.selectForeground', ACC_FG),
                 ('*Listbox.relief', 'flat'), ('*Listbox.borderWidth', 0),
                 ('*Checkbutton.selectColor', BG2), ('*Checkbutton.activeBackground', BG),
                 ('*Checkbutton.activeForeground', FG), ('*Scrollbar.Background', BG2),
                 ('*Scrollbar.troughColor', BG), ('*Scrollbar.relief', 'flat')):
        r.option_add(k, v)
    return r


def _win(root, title):
    import tkinter as tk
    w = tk.Toplevel(root, bg=BG, padx=8, pady=6)
    w.title(title)
    try:
        w.attributes('-topmost', True)
    except Exception:
        pass
    try:
        import os
        w.iconbitmap(os.path.join(os.path.dirname(__file__), 'icon.ico'))
    except Exception:
        pass
    w.resizable(True, True)
    return w


def _btn(parent, text, command, primary=False):
    """Кнопка как в окнах: плоская, тёмная; главная — жёлтая. Наведение — светлее."""
    import tkinter as tk
    bg, fg = (ACC, ACC_FG) if primary else ('#3a3a42', FG)
    hv = '#ffd84d' if primary else '#4a4a55'
    b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg, activebackground=hv, activeforeground=fg,
                  relief='flat', bd=0, padx=16, pady=5, cursor='hand2', highlightthickness=0)
    b.bind('<Enter>', lambda *_: b.config(bg=hv))
    b.bind('<Leave>', lambda *_: b.config(bg=bg))
    return b


def _label(parent, text, dim=False, bold=False, **kw):
    import tkinter as tk
    fam = getattr(parent.winfo_toplevel().master, '_fam', 'TkFixedFont')
    return tk.Label(parent, text=text, fg=DIM_ if dim else FG, font=(fam, 10, 'bold') if bold else (fam, 10), **kw)


def new_project(title="New project", name_label="Name", desc_label="Short description"):
    """Имя и описание (проекта или образца) -> (имя, описание) или None."""
    import tkinter as tk
    root = _root()
    w = _win(root, title)
    out = {}
    _label(w, name_label, dim=True).grid(row=0, column=0, sticky='w', padx=10, pady=(10, 2))
    e1 = tk.Entry(w, width=48, bd=6, relief='flat')
    e1.grid(row=1, column=0, padx=10, sticky='we')
    _label(w, desc_label, dim=True).grid(row=2, column=0, sticky='w', padx=10, pady=(10, 2))
    t = tk.Text(w, width=48, height=4, wrap='word', bd=6)
    t.grid(row=3, column=0, padx=10, sticky='we')

    def ok(*_):
        if e1.get().strip():
            out['r'] = (e1.get().strip(), t.get('1.0', 'end').strip())
            root.quit()
    b = tk.Frame(w)
    b.grid(row=4, column=0, pady=10)
    _btn(b, "Create", ok, primary=True).pack(side='left', padx=5)
    _btn(b, "Cancel", root.quit).pack(side='left', padx=5)
    w.bind('<Return>', ok)
    w.protocol("WM_DELETE_WINDOW", root.quit)
    e1.focus_set()
    root.mainloop()
    root.destroy()
    return out.get('r')


def ask(title, label, value=''):
    """Одна строка текста -> строка или None."""
    import tkinter as tk
    root = _root()
    w = _win(root, title)
    out = {}
    _label(w, label, dim=True).pack(anchor='w', padx=10, pady=(10, 2))
    e = tk.Entry(w, width=32, bd=6, relief='flat')
    e.insert(0, value)
    e.pack(padx=10, fill='x')

    def ok(*_):
        if e.get().strip():
            out['r'] = e.get().strip()
            root.quit()
    b = tk.Frame(w)
    b.pack(pady=(12, 8))
    _btn(b, "Open", ok, primary=True).pack(side='left', padx=5)
    _btn(b, "Cancel", root.quit).pack(side='left', padx=5)
    w.bind('<Return>', ok)
    w.protocol("WM_DELETE_WINDOW", root.quit)
    e.focus_set()
    root.mainloop()
    root.destroy()
    return out.get('r')


def checklist(title, rows, checked, header=None, button="Export", kids=None):
    """Список с галочками -> [True / False] по строкам или None (отмена). kids — {строка: [строки внутри неё]}:
    галочка строки ставит / снимает и все вложенные (потом их можно поменять по одной)."""
    import tkinter as tk
    root = _root()
    w = _win(root, title)
    out = {}
    if header:
        tk.Label(w, text=header, font=(root._fam, 10, 'bold'), anchor='w', justify='left').pack(fill='x', padx=12,
                                                                                                pady=(10, 0))
    fr = tk.Frame(w)
    fr.pack(fill='both', expand=True, padx=10, pady=6)
    vs = []

    def toggled(i):
        for j in (kids or {}).get(i, ()):
            vs[j].set(vs[i].get())
    for i, (r, c) in enumerate(zip(rows, checked)):
        v = tk.BooleanVar(value=bool(c))
        tk.Checkbutton(fr, text=r, variable=v, font=(root._fam, 10), anchor='w', bd=0, pady=2,
                       command=lambda i=i: toggled(i)).pack(fill='x')
        vs.append(v)

    def ok(*_):
        out['r'] = [v.get() for v in vs]
        root.quit()
    b = tk.Frame(w)
    b.pack(pady=(10, 8))
    _btn(b, "All", lambda: [v.set(True) for v in vs]).pack(side='left', padx=5)
    _btn(b, "None", lambda: [v.set(False) for v in vs]).pack(side='left', padx=5)
    _btn(b, button, ok, primary=True).pack(side='left', padx=5)
    _btn(b, "Cancel", root.quit).pack(side='left', padx=5)
    w.bind('<Return>', ok)
    w.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()
    root.destroy()
    return out.get('r')


def choose(title, rows, current=None, buttons=("Open",), extra=None, header=None):
    """Список строк -> (кнопка, индекс) или None. extra — подпись сверху, header — строка заголовков колонок
    (тем же моноширинным шрифтом, что и список: таблица)."""
    import tkinter as tk
    root = _root()
    w = _win(root, title)
    out = {}
    if extra:
        _label(w, extra, dim=True, justify='left').pack(anchor='w', padx=10, pady=(10, 0))
    if header:
        tk.Label(w, text=header, font=(root._fam, 10, 'bold'), anchor='w', justify='left').pack(
            fill='x', padx=12, pady=(10, 0))
    fr = tk.Frame(w)
    fr.pack(fill='both', expand=True, padx=10, pady=(2 if header else 10, 10))
    sb = tk.Scrollbar(fr)
    sb.pack(side='right', fill='y')
    lb = tk.Listbox(fr, width=min(170, max(100, *(len(r) + 2 for r in rows))) if rows else 100, height=min(20, max(5, len(rows))), yscrollcommand=sb.set,
                    font=(root._fam, 10), activestyle='none')
    for r in rows:
        lb.insert('end', r)
    lb.pack(side='left', fill='both', expand=True)
    sb.config(command=lb.yview)
    if current is not None and 0 <= current < len(rows):
        lb.selection_set(current)
        lb.see(current)

    def press(name):
        sel = lb.curselection()
        if not name.startswith("New") and not sel:
            return
        out['r'] = (name, sel[0] if sel else None)
        root.quit()
    b = tk.Frame(w)
    b.pack(pady=(10, 8))
    for i_, name in enumerate(buttons):
        _btn(b, name, lambda n=name: press(n), primary=i_ == 0).pack(side='left', padx=5)
    _btn(b, "Cancel", root.quit).pack(side='left', padx=5)
    lb.bind('<Double-Button-1>', lambda *_: press(buttons[0]))
    w.bind('<Return>', lambda *_: press(buttons[0]))
    w.protocol("WM_DELETE_WINDOW", root.quit)
    lb.focus_set()
    root.mainloop()
    root.destroy()
    return out.get('r')


def describe(title, lines, fields):
    """Сведения (строки) и поля описаний [(подпись, текст)] -> [новые тексты] (Save) или None."""
    import tkinter as tk
    root = _root()
    w = _win(root, title)
    out = {}
    tk.Label(w, text="\n".join(lines), justify='left', font=(root._fam, 10)).pack(anchor='w', padx=10, pady=10)
    boxes = []
    for lab, txt in fields:
        _label(w, lab, dim=True).pack(anchor='w', padx=10)
        t = tk.Text(w, width=70, height=3, wrap='word', bd=6)
        t.insert('1.0', txt or '')
        t.pack(fill='both', expand=True, padx=10, pady=(0, 8))
        boxes.append(t)

    def save():
        out['r'] = [t.get('1.0', 'end').strip() for t in boxes]
        root.quit()
    b = tk.Frame(w)
    b.pack(pady=(12, 8))
    _btn(b, "Save", save, primary=True).pack(side='left', padx=5)
    _btn(b, "Close", root.quit).pack(side='left', padx=5)
    w.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()
    root.destroy()
    return out.get('r')
