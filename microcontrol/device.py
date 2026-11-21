"""Память прибора: у каждого микроскопа (по модели и серийному номеру) свой файл.

В файле хранится:
  home     — «дом»: точка, от которой считаются настоящие координаты (m.set_home()), и фото этого места;
  last     — последнее положение столика (шаги мотора от дома) и было ли движение прервано.

Контроллер не знает, где столик: положение считает программа. Поэтому оно записывается после
каждого движения, а при подключении восстанавливается — но как «неподтверждённое», пока человек
не скажет m.confirm_position() (между сеансами столик могли сдвинуть рукой или другой программой).

Файлы: ~/.microcontrol/devices/<модель>_<серийный>.json и ..._home.png (для симулятора — в папке работы).
"""
import json, os, re, tempfile
from datetime import datetime

DIR = os.path.join(os.path.expanduser("~"), ".microcontrol", "devices")


class DeviceMemory:
    def __init__(self, model, serial, folder=None):
        name = re.sub(r'[^A-Za-z0-9._-]+', '-', f"{model}_{serial}").strip('-') or 'device'
        self.dir = folder or DIR
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, name + ".json")
        self.photo_path = os.path.join(self.dir, name + "_home.png")
        self.data = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding='utf-8') as f:
                    self.data = json.load(f)
            except (OSError, ValueError):
                self.data = {}                   # испорченный файл — как будто памяти нет
        self.data.update(model=model, serial=serial)

    @property
    def home(self):
        return self.data.get('home')

    @property
    def last(self):
        return self.data.get('last')

    def save(self):
        """Атомарная запись: сначала во временный файл, потом замена (файл не бывает наполовину записан)."""
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix='.tmp')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def position(self, p, moving, **extra):
        """Запомнить положение (шаги от дома). moving=True — движение началось и ещё не закончено."""
        self.data['last'] = dict(p=dict(p), moving=bool(moving), time=_now(), **extra)
        self.save()

    def set_home(self, **info):
        self.data['home'] = dict(time=_now(), **info)
        self.save()


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
