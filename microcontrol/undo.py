"""История действий в окнах (Ctrl+Z — отменить, Ctrl+Y или Ctrl+Shift+Z — вернуть).

Окно после каждого действия сравнивает своё состояние с прежним; изменилось — в историю кладётся пара
(было, стало) с подписью. Отмена восстанавливает «было», возврат — «стало». Новое действие после отмены
стирает ветку возвратов, как в любом редакторе.
"""
LIMIT = 1000


class History:
    def __init__(self):
        self.done, self.undone = [], []

    def push(self, label, before, after, **extra):
        self.done.append(dict(label=label or "action", before=before, after=after, **extra))
        del self.done[:-LIMIT]
        self.undone.clear()

    def undo(self):
        """Последнее действие (dict) — его надо откатить к ['before']; None — нечего отменять."""
        if not self.done:
            return None
        e = self.done.pop()
        self.undone.append(e)
        return e

    def redo(self):
        if not self.undone:
            return None
        e = self.undone.pop()
        self.done.append(e)
        return e

    def __len__(self):
        return len(self.done)


def undo_key(key):
    """'undo' (Ctrl+Z), 'redo' (Ctrl+Y, Ctrl+Shift+Z) или None. Windows присылает управляющие коды 26 / 25;
    GTK — букву с модификаторами в старших битах (Ctrl = 0x04, Shift = 0x01)."""
    if key is None or key < 0:
        return None
    low, mods = key & 0xFF, (key >> 16) & 0xFF
    if low == 26 and not mods:
        return 'undo'
    if low == 25 and not mods:
        return 'redo'
    ctrl = bool(mods & 0x04)
    shift = bool(mods & 0x01)
    if not ctrl:
        try:                                                # Windows: Ctrl+буква могла прийти буквой
            import ctypes
            ctrl = bool(ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000)
            shift = shift or bool(ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000)
        except Exception:
            ctrl = False
    if not ctrl:
        return None
    if low in (ord('z'), ord('Z')):
        return 'redo' if shift or low == ord('Z') else 'undo'
    if low in (ord('y'), ord('Y')):
        return 'redo'
    return None
