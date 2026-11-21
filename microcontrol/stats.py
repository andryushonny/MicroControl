"""Где проходит время: каждое действие (движение, ожидание, снимок, фокус, запись, сшивка ...) меряется
таймером, вложенные действия вычитаются из внешних (у каждого — своё, «чистое» время), отчёт — m.info().

    with timed("Stage: travel"):
        ...
"""
import time
from contextlib import contextmanager

_T0 = time.monotonic()
_DATA = {}                  # имя -> [вызовов, своё время, всего, самый долгий]
_STACK = []                 # открытые действия: [имя, начало, время вложенных]


def reset():
    global _T0
    _DATA.clear()
    _T0 = time.monotonic()


@contextmanager
def timed(name):
    """Измерить действие name. Вложенные timed() считаются отдельно и из этого вычитаются."""
    fr = [name, time.monotonic(), 0.0]
    _STACK.append(fr)
    try:
        yield
    finally:
        dt = time.monotonic() - fr[1]
        if _STACK and _STACK[-1] is fr:
            _STACK.pop()
        else:                                           # на случай исключения внутри вложенного
            try:
                _STACK.remove(fr)
            except ValueError:
                pass
        if _STACK:
            _STACK[-1][2] += dt
        own = max(0.0, dt - fr[2])
        d = _DATA.setdefault(name, [0, 0.0, 0.0, 0.0])
        d[0] += 1
        d[1] += own
        d[2] += dt
        d[3] = max(d[3], dt)


def add(name, seconds, calls=1):
    """Добавить уже измеренное время (без вложенности)."""
    d = _DATA.setdefault(name, [0, 0.0, 0.0, 0.0])
    d[0] += calls
    d[1] += seconds
    d[2] += seconds
    d[3] = max(d[3], seconds / max(1, calls))
    if _STACK:
        _STACK[-1][2] += seconds


def _t(s):
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{s % 60:04.1f}" if s >= 3600 else \
        f"{int(s // 60):02d}:{s % 60:04.1f}"


def report(top=20):
    """Таблица: действие, вызовов, своё время, доля от всего измеренного, в среднем, самое долгое."""
    wall = time.monotonic() - _T0
    tot = sum(d[1] for d in _DATA.values())
    rows = sorted(_DATA.items(), key=lambda kv: -kv[1][1])[:top]
    if not rows:
        return "Nothing measured yet."
    w = max(28, max(len(k) for k, _ in rows) + 2)
    head = (f"Where the time goes: measured {_t(tot)} of {_t(wall)} since "
            f"{'the start' if wall > 0 else 'now'} (the rest is you, Python, Jupyter)\n")
    lines = [f"  {'Action':<{w}}{'Calls':>8}{'Own time':>12}{'Share':>9}{'With inner':>12}{'Mean':>10}{'Longest':>10}"]
    for k, (n, own, al, mx) in rows:
        lines.append(f"  {k:<{w}}{n:>8}{_t(own):>12}{100 * own / max(tot, 1e-9):>8.1f}%{_t(al):>12}"
                     f"{al / max(n, 1):>9.2f}s{mx:>9.2f}s")
    bar = rows[0]
    tip = f"\n  Most time: {bar[0]} ({100 * bar[1][1] / max(tot, 1e-9):.0f} %)."
    return (head + "\n".join(lines) + tip + "\n  Own time: without the actions inside it (they have their own lines); "
            "with inner: all of it.\n  m.info(reset=True): count from zero.")


def data():
    """{действие: dict(calls, seconds, total_seconds, longest)} — для агента и своих отчётов."""
    return {k: dict(calls=n, seconds=round(own, 3), total_seconds=round(a, 3), longest=round(mx, 3))
            for k, (n, own, a, mx) in _DATA.items()}


def timed_fn(name):
    """Декоратор: вся функция — одно действие name."""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            with timed(name):
                return fn(*a, **kw)
        return wrap
    return deco
