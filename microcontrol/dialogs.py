"""Простые окна для меню (tkinter): новый проект, открыть проект, описание проекта, выбрать съёмку.
Каждое окно модальное и поверх остальных; закрыли — None."""


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
    return r


def _win(root, title):
    import tkinter as tk
    w = tk.Toplevel(root)
    w.title(title)
    try:
        w.attributes('-topmost', True)
    except Exception:
        pass
    w.resizable(True, True)
    return w


def new_project(title="New project", name_label="Name", desc_label="Short description"):
    """Имя и описание (проекта или образца) -> (имя, описание) или None."""
    import tkinter as tk
    root = _root()
    w = _win(root, title)
    out = {}
    tk.Label(w, text=name_label).grid(row=0, column=0, sticky='w', padx=10, pady=(10, 2))
    e1 = tk.Entry(w, width=48)
    e1.grid(row=1, column=0, padx=10, sticky='we')
    tk.Label(w, text=desc_label).grid(row=2, column=0, sticky='w', padx=10, pady=(10, 2))
    t = tk.Text(w, width=48, height=4, wrap='word')
    t.grid(row=3, column=0, padx=10, sticky='we')

    def ok(*_):
        if e1.get().strip():
            out['r'] = (e1.get().strip(), t.get('1.0', 'end').strip())
            root.quit()
    b = tk.Frame(w)
    b.grid(row=4, column=0, pady=10)
    tk.Button(b, text="Create", width=12, command=ok).pack(side='left', padx=5)
    tk.Button(b, text="Cancel", width=12, command=root.quit).pack(side='left', padx=5)
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
    tk.Label(w, text=label).pack(anchor='w', padx=10, pady=(10, 2))
    e = tk.Entry(w, width=32)
    e.insert(0, value)
    e.pack(padx=10, fill='x')

    def ok(*_):
        if e.get().strip():
            out['r'] = e.get().strip()
            root.quit()
    b = tk.Frame(w)
    b.pack(pady=10)
    tk.Button(b, text="Open", width=12, command=ok).pack(side='left', padx=5)
    tk.Button(b, text="Cancel", width=12, command=root.quit).pack(side='left', padx=5)
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
        tk.Label(w, text=header, font=('Consolas', 10, 'bold'), anchor='w', justify='left').pack(fill='x', padx=12,
                                                                                                pady=(10, 0))
    fr = tk.Frame(w)
    fr.pack(fill='both', expand=True, padx=10, pady=6)
    vs = []

    def toggled(i):
        for j in (kids or {}).get(i, ()):
            vs[j].set(vs[i].get())
    for i, (r, c) in enumerate(zip(rows, checked)):
        v = tk.BooleanVar(value=bool(c))
        tk.Checkbutton(fr, text=r, variable=v, font=('Consolas', 10), anchor='w',
                       command=lambda i=i: toggled(i)).pack(fill='x')
        vs.append(v)

    def ok(*_):
        out['r'] = [v.get() for v in vs]
        root.quit()
    b = tk.Frame(w)
    b.pack(pady=(0, 10))
    tk.Button(b, text="All", width=8, command=lambda: [v.set(True) for v in vs]).pack(side='left', padx=5)
    tk.Button(b, text="None", width=8, command=lambda: [v.set(False) for v in vs]).pack(side='left', padx=5)
    tk.Button(b, text=button, width=12, command=ok).pack(side='left', padx=5)
    tk.Button(b, text="Cancel", width=12, command=root.quit).pack(side='left', padx=5)
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
        tk.Label(w, text=extra, justify='left').pack(anchor='w', padx=10, pady=(10, 0))
    if header:
        tk.Label(w, text=header, font=('Consolas', 10, 'bold'), anchor='w', justify='left').pack(
            fill='x', padx=12, pady=(10, 0))
    fr = tk.Frame(w)
    fr.pack(fill='both', expand=True, padx=10, pady=(2 if header else 10, 10))
    sb = tk.Scrollbar(fr)
    sb.pack(side='right', fill='y')
    lb = tk.Listbox(fr, width=min(170, max(100, *(len(r) + 2 for r in rows))) if rows else 100, height=min(20, max(5, len(rows))), yscrollcommand=sb.set,
                    font=('Consolas', 10), activestyle='dotbox')
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
    b.pack(pady=(0, 10))
    for name in buttons:
        tk.Button(b, text=name, width=12, command=lambda n=name: press(n)).pack(side='left', padx=5)
    tk.Button(b, text="Cancel", width=12, command=root.quit).pack(side='left', padx=5)
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
    tk.Label(w, text="\n".join(lines), justify='left', font=('Consolas', 10)).pack(anchor='w', padx=10, pady=10)
    boxes = []
    for lab, txt in fields:
        tk.Label(w, text=lab).pack(anchor='w', padx=10)
        t = tk.Text(w, width=70, height=3, wrap='word')
        t.insert('1.0', txt or '')
        t.pack(fill='both', expand=True, padx=10, pady=(0, 8))
        boxes.append(t)

    def save():
        out['r'] = [t.get('1.0', 'end').strip() for t in boxes]
        root.quit()
    b = tk.Frame(w)
    b.pack(pady=10)
    tk.Button(b, text="Save", width=12, command=save).pack(side='left', padx=5)
    tk.Button(b, text="Close", width=12, command=root.quit).pack(side='left', padx=5)
    w.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()
    root.destroy()
    return out.get('r')
