"""Очередь задач: съёмки выделений одна за другой (прибор один) и сшивки в фоне на нескольких ядрах.

    q = m.queue                         # очередь этого микроскопа (окно m.manual() пользуется ей же)
    q.scan(uid)                         # снять выделение (прибор), затем quick-сшивка (фон, отдельный процесс)
    q.stitch(folder, 'quality')         # сшивка в фоне (без пересъёмки)
    q.quality(folder)                   # точная сшивка: сшить (фон) -> проверить -> доснять (прибор) -> ...
    q.stop(job) / q.stop_all()          # остановить идущую или убрать ждущую
    q.wait()                            # дождаться всех

Две «дорожки»: прибор (inst) — одна задача за раз, в потоке этого процесса (столик, камера); ядра (cpu) —
до q.workers сшивок сразу, каждая в своём процессе python -m microcontrol.jobs (не мешает окну, её можно
остановить мгновенно). Задача — цепочка шагов (снять -> сшить, сшить -> доснять -> сшить): между шагами она
отпускает дорожку, и на приборе уже снимается следующее выделение.
"""
import os, sys, json, time, threading, subprocess

STATES_ACTIVE = ('queued', 'running', 'waiting')
_RATES_DEF = {'quick': 0.06, 'quality': 0.6}               # с на кадр, пока не измерено на этом компьютере


def _rates_path():
    return os.path.join(os.path.expanduser('~'), '.microcontrol', 'job_rates.json')


def _rates(kind=None, sec_per_frame=None):
    """Сколько секунд на кадр уходит у сшивки этого вида (по прошлым разам; среднее с весом новому 0.3)."""
    p = _rates_path()
    try:
        d = json.load(open(p, encoding='utf-8'))
    except (OSError, ValueError):
        d = {}
    if sec_per_frame is not None and kind:
        d[kind] = sec_per_frame if kind not in d else 0.7 * d[kind] + 0.3 * sec_per_frame
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            json.dump(d, open(p, 'w', encoding='utf-8'))
        except OSError:
            pass
    return d.get(kind, _RATES_DEF.get(kind, 0.3)) if kind else d


def n_frames(folder):
    """Кадров в съёмке (строки index.csv)."""
    if not folder:
        return 0
    try:
        with open(os.path.join(folder, 'index.csv'), encoding='utf-8') as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def tm(sec):
    sec = max(0, int(round(sec or 0)))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60:02d}:{sec % 60:02d}"


class Job:
    """Одна задача очереди. state: queued (ждёт), running (шаг идёт), waiting (между шагами: ждёт дорожку),
    done, failed, stopped. lane — дорожка текущего шага: 'inst' (прибор) или 'cpu' (ядра)."""
    _n = 0

    def __init__(self, kind, title, steps, est=0.0, uid=None, folder=None, obj=None):
        Job._n += 1
        self.id, self.kind, self.title, self.uid, self.folder, self.obj = Job._n, kind, title, uid, folder, obj
        self.state, self.lane, self.step_title = 'queued', None, ''
        self.frac, self.text, self.left = 0.0, '', None
        self.est, self.t_add, self.t0, self.t_end = float(est or 0.0), time.monotonic(), None, None
        self.result, self.error, self.note = None, None, ''
        self._gen, self._send, self._stop, self._halt = steps, None, False, None
        self._proc = None
        self.paused = False                             # пауза: ждёт, пока не возобновят (сканер — после кадра)
        self.step_t0, self.step_est = None, 0.0

    @property
    def active(self):
        return self.state in STATES_ACTIVE

    def elapsed(self):
        return 0.0 if self.t0 is None else (self.t_end or time.monotonic()) - self.t0

    def remaining(self):
        """Сколько ещё (оценка): идёт — по полосе прогресса шага или по оценке; ждёт — вся оценка."""
        if not self.active:
            return 0.0
        nx = getattr(self, '_next', None)
        if nx is None:
            return self.est
        after = nx[5] if len(nx) > 5 else 0.0
        if self.state == 'running':
            el = time.monotonic() - (self.step_t0 or time.monotonic())
            return (self.left if self.left is not None else max(0.0, self.step_est - el)) + after
        return (nx[4] or 0.0) + after

    def line(self):
        s = {'queued': 'waiting', 'waiting': 'waiting', 'running': f"{100 * (self.frac or 0):.0f}%"}.get(self.state, self.state)
        return f"{self.title}  {s}"


_ALL = []                                                  # очереди этого процесса (просмотр показывает их)


def summary_all():
    """Строка о всех очередях процесса (для просмотра: там видно, что в фоне что-то идёт) или ''."""
    return "; ".join(q.summary() for q in _ALL if q.active())


class JobQueue:
    def __init__(self, m=None, workers=None):
        self.m = m
        self.workers = workers or max(1, min(4, (os.cpu_count() or 2) // 2))
        self.jobs = []
        self.events = []                                # (текст, job) — окну: что закончилось
        self._lock = threading.RLock()
        self._inst = None                               # (job, поток) — шаг на приборе
        self._cpu = []                                  # задачи, чей шаг идёт на ядрах
        self._wake = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True, name='mc-queue')
        self._th.start()
        _ALL.append(self)

    # ---------------------------------------------------------------- задачи
    def add(self, job):
        with self._lock:
            self.jobs.append(job)
        self._wake.set()
        return job

    def scan(self, uid, title=None, est=None, batch=None):
        """Снять выделение uid (текущим объективом), затем quick-сшивка на ядрах. batch — общий знак съёмок,
        поставленных вместе: m.last_scans — их папки (как после m.scan() нескольких выделений)."""
        m = self.m
        rg = next((r for r in m._regions if r.get('uid') == uid), None)
        if rg is None:                                  # выделение сделали в просмотре: взять с диска
            from .layers import refresh_from_disk
            refresh_from_disk(m)
            rg = next((r for r in m._regions if r.get('uid') == uid), None)
        from . import samples as SM
        k = m._regions.index(rg) if rg is not None else None
        title = title or (f"Scan {SM.title(m, k)}" if k is not None else "Scan")
        if est is None and k is not None:
            from . import timing as TM
            import numpy as np
            nf = int(np.prod(SM.frames_of(m, k)))
            est = TM.estimate(m, nf, ('points' if m._use_fp else 'off') if rg.get('pts') else 'off')[0] \
                + nf * _rates('quick')
        o_ = SM.obj(m, rg) if rg is not None else m.objective
        j = Job('scan', title, None, est=est, uid=uid, obj=o_)
        j.batch = batch if batch is not None else j.id
        j.hold = o_ != m.objective                     # другой объектив: ждёт, пока его поставят (и S)
        j._gen = self._scan_steps(j)
        return self.add(j)

    def stitch(self, folder, mode='quality', title=None):
        nf = n_frames(folder)
        j = Job('stitch', title or f"{mode.capitalize()} stitch", None, est=nf * _rates(mode), folder=folder)
        j._gen = self._stitch_steps(j, mode)
        return self.add(j)

    def quality(self, folder, title=None, rounds=2, reshoot=True):
        """Точная сшивка: сшить на ядрах, проверить, доснять на приборе (тем же объективом), снова сшить."""
        nf = n_frames(folder)
        j = Job('quality', title or "Quality stitch", None, est=nf * _rates('quality'), folder=folder,
                obj=getattr(self.m, 'objective', None))
        j._gen = self._quality_steps(j, rounds, reshoot)
        return self.add(j)

    # ---------------------------------------------------------------- шаги задач (генераторы)
    def _scan_steps(self, j):
        m = self.m

        def shoot():
            from .util import no_stop
            from .layers import refresh_from_disk
            refresh_from_disk(m)                        # просмотр мог поменять выделения: не затереть их
            if not any(r.get('uid') == j.uid for r in m._regions):
                raise RuntimeError("the selection is gone (deleted)")
            if j.obj != m.objective:
                raise RuntimeError(f"the objective is {m.objective}x now, the scan was queued for {j.obj}x")
            m._abort = False
            from . import storage as ST_
            from .live import Runner
            run_ = {}

            def on_new(f_):                             # кадры сшиваются на ходу (свой процесс, на ядрах)
                if 'r' not in run_:
                    run_['r'] = Runner(f_)
            ST_.ON_CREATE.append(on_new)
            f = None
            try:
                with no_stop():                         # скан останавливается сам, после кадра (снятое — цело)
                    f = m.scan(only=[j.uid], stitch=None)
            finally:
                ST_.ON_CREATE.remove(on_new)
                if 'r' in run_:
                    run_['r'].stop()                    # дошить последние кадры: quick возьмёт швы готовыми
                stopped_ = bool(m._abort or j._stop)
                m._abort = False
                fp_ = f or (run_['r'].folder if 'r' in run_ else None)
                if fp_ and (stopped_ or f is None) and n_frames(fp_) > 0:   # прервали (или ошибка): снятое — сшить
                    from .project import scan_label as _sl  # отдельной задачей на ядрах: прибор и очередь — дальше
                    self.stitch(fp_, 'quick', title=f"Quick stitch {_sl(fp_)} (stopped)")
            return f, stopped_

        def halt():
            m._abort = True
        q_ = n_frames_est(j, m) * _rates('quick')
        folder, stopped_ = yield ('inst', "Scanning", shoot, halt, max(1.0, j.est - q_), q_)
        j.folder = folder
        if folder:
            if getattr(m, '_q_batch', None) != j.batch:      # m.last_scans — съёмки этой партии
                m._q_batch, m._q_scans = j.batch, []
            m._q_scans.append(folder)
            m.last_scans, m.last_scan = list(m._q_scans), folder
        if stopped_:
            j.note = f"stopped: {n_frames(folder)} frames so far are kept (quick stitch queued)"
            return
        yield ('cpu', "Quick stitch", ('stitch', folder, {'mode': 'quick'}), None, n_frames(folder) * _rates('quick'), 0.0)
        j.note = "scanned, quick stitched"

    def _stitch_steps(self, j, mode):
        yield ('cpu', f"{mode.capitalize()} stitch", ('stitch', j.folder, {'mode': mode}), None, j.est, 0.0)
        j.note = f"{mode} stitched"

    def _quality_steps(self, j, rounds, reshoot):
        from . import quality as Q
        m = self.m
        total = 0
        for k in range(rounds + 1):
            yield ('cpu', "Quality stitch" + (f" (round {k + 1})" if k else ""), ('stitch', j.folder, {'mode': 'quality'}),
                   None, n_frames(j.folder) * _rates('quality'), 0.0)
            rep = Q.info(j.folder) or {}
            if k == rounds or rep.get('ok') or not reshoot:
                break
            if m is None or getattr(m, 'objective', None) != _folder_obj(j.folder):
                j.note = "not re-shot: another objective"
                break

            def again(rep=rep):
                return Q.reshoot(m, j.folder, rep, verbose=True)

            def halt():
                m._abort = True
            n = yield ('inst', "Re-shooting", again, halt, 30.0, n_frames(j.folder) * _rates('quality'))
            if not n:
                break
            total += n
            Q._meta_add(j.folder, quality_rounds=k + 1)
        rep = Q.info(j.folder) or {}
        j.result = rep
        j.note = ("quality: " + (f"{rep['size_um'][0]:.1f} x {rep['size_um'][1]:.1f} um" if rep.get('size_um') else "done")
                  + (f", re-shot {total}" if total else "")
                  + ("" if rep.get('ok', True) else "; " + "; ".join(rep.get('problems', []))))

    # ---------------------------------------------------------------- управление
    def stop(self, job):
        """Остановить задачу: идущую — на ближайшем шаге (скан — после кадра, сшивка — сразу), ждущую — убрать."""
        if isinstance(job, int):
            job = next((j for j in self.jobs if j.id == job), None)
        if job is None or not job.active:
            return False
        if job.paused:                                  # на паузе: сначала отпустить, иначе не остановится
            self.resume(job)
        with self._lock:
            job._stop = True
            if job.state in ('queued', 'waiting') and not (self._inst and self._inst[0] is job) and job not in self._cpu:
                self._finish(job, 'stopped', note=job.note or "removed from the queue")
            else:
                if job._halt is not None:
                    try:
                        job._halt()
                    except Exception:
                        pass
                if job._proc is not None:
                    try:
                        job._proc.stdin.write("stop\n")
                        job._proc.stdin.flush()
                    except (OSError, ValueError):
                        pass
                    job._t_stop = time.monotonic()
                tid = getattr(job, '_tid', None)
                if tid is not None:
                    from .util import job_stop
                    job_stop(tid)
        self._wake.set()
        return True

    def pause(self, job):
        """Пауза задачи (до resume): скан — после текущего кадра (прибор свободен), сшивка — процесс замирает,
        ждущая — не начнётся. Остальные задачи идут. -> True / False."""
        if isinstance(job, int):
            job = next((j for j in self.jobs if j.id == job), None)
        if job is None or not job.active or job.paused:
            return False
        with self._lock:
            job.paused = True
            job._t_pause = time.monotonic()
            if self._inst is not None and self._inst[0] is job and self.m is not None:
                self.m._pause = True
            if job._proc is not None and job._proc.poll() is None:
                _suspend(job._proc.pid, True)
        self._wake.set()
        return True

    def resume(self, job):
        if isinstance(job, int):
            job = next((j for j in self.jobs if j.id == job), None)
        if job is None or not job.paused:
            return False
        with self._lock:
            job.paused = False
            dt_ = time.monotonic() - getattr(job, '_t_pause', time.monotonic())
            if job.step_t0 is not None:                 # пауза — не в счёт времени шага
                job.step_t0 += dt_
            if self._inst is not None and self._inst[0] is job and self.m is not None:
                self.m._pause = False
            if job._proc is not None and job._proc.poll() is None:
                _suspend(job._proc.pid, False)
        self._wake.set()
        return True

    def stop_all(self):
        for j in list(self.jobs):
            self.stop(j)

    def active(self):
        return [j for j in self.jobs if j.active]

    def inst_busy(self):
        """Прибор занят очередью: идёт или вот-вот пойдёт шаг на приборе."""
        with self._lock:
            if self._inst is not None:
                return True
            return any(j.active and (j.lane == 'inst' or (j.kind == 'scan' and j.state == 'queued' and not self.held(j)))
                       for j in self.jobs)

    def held(self, j):
        """Съёмка ждёт свой объектив (его ещё не поставили или не сказали S)."""
        return j.kind == 'scan' and j.state == 'queued' and not j.paused and j.obj != getattr(self.m, 'objective', j.obj)   # стоит — пошла

    def waiting_for(self):
        """{объектив: сколько съёмок его ждут}."""
        out = {}
        for j in self.jobs:
            if j.active and self.held(j):
                out[j.obj] = out.get(j.obj, 0) + 1
        return out

    def release(self, obj):
        """Объектив obj поставили: ждущие его съёмки — в работу. -> сколько."""
        n = 0
        with self._lock:
            for j in self.jobs:
                if j.active and j.kind == 'scan' and j.state == 'queued' and j.obj == obj and getattr(j, 'hold', False):
                    j.hold = False
                    n += 1
        self._wake.set()
        return n

    def total_left(self):
        """Сколько ещё всей очереди (оценка): прибор — подряд, ядра — параллельно на workers."""
        inst = sum(j.remaining() for j in self.jobs if j.active and j.lane in ('inst', None) and not self.held(j))
        cpu = sum(j.remaining() for j in self.jobs if j.active and j.lane == 'cpu')
        return max(inst, cpu / self.workers)

    def summary(self):
        a = self.active()
        if not a:
            return ""
        pz_ = sum(1 for j in a if j.paused)
        run_ = [j for j in a if j.state == 'running' and not j.paused]
        wf_ = self.waiting_for()
        held_ = sum(wf_.values())
        wait_ = len(a) - len(run_) - held_ - pz_
        return (f"Queue: {len(run_)} running" + (f", {pz_} paused" if pz_ else "") + (f", {wait_} waiting" if wait_ else "")
                + "".join(f", {n} wait for {o}x" for o, n in sorted(wf_.items()))
                + (f"  -  all done in ~{tm(self.total_left())}" if len(a) > held_ else ""))

    def recent(self, sec=8.0):
        now = time.monotonic()
        return [j for j in self.jobs if j.active or (j.t_end and now - j.t_end < sec)]

    def wait(self, timeout=None):
        """Дождаться, пока все задачи закончатся. -> True, если закончились."""
        t0 = time.monotonic()
        while self.active():
            if timeout is not None and time.monotonic() - t0 > timeout:
                return False
            time.sleep(0.05)
        return True

    def pop_events(self):
        with self._lock:
            ev, self.events = self.events, []
        return ev

    # ---------------------------------------------------------------- планировщик
    def _finish(self, j, state, note=None, error=None):
        j.state, j.lane, j.t_end = state, None, time.monotonic()
        if note:
            j.note = note
        if error is not None:
            j.error = error
        if j.t0 is None:
            j.t0 = j.t_end
        txt = {'done': f"Done: {j.title}" + (f" ({j.note})" if j.note else ""),
               'failed': f"Failed: {j.title}: {j.error}",
               'stopped': f"Stopped: {j.title}" + (f" ({j.note})" if j.note else "")}[state]
        self.events.append((txt, j))
        try:
            from . import log as L
            L.event('JOB-001', txt + f"  [{tm(j.elapsed())}]")
        except Exception:
            pass

    def _advance(self, j, value=None, exc=None):
        """Следующий шаг задачи (из генератора). -> шаг или None (задача закончилась)."""
        try:
            if exc is not None:
                st = j._gen.throw(exc)
            elif j._send is None and j.state == 'queued':
                st = next(j._gen)
            else:
                st = j._gen.send(value)
        except StopIteration:
            self._finish(j, 'stopped' if j._stop else 'done')
            return None
        except Exception as e:
            self._finish(j, 'failed', error=e)
            return None
        j._send = True
        j._next = st
        j.state = 'waiting'
        j.lane = st[0]
        if j.t0 is None:
            j.t0 = time.monotonic()
        return st

    def _loop(self):
        while True:
            self._wake.wait(0.1)
            self._wake.clear()
            try:
                self._tick()
            except Exception as e:                       # планировщик не падает
                try:
                    from . import log as L
                    L.debug('JOB-002', f"queue tick: {e}")
                except Exception:
                    pass

    def _tick(self):
        with self._lock:
            for j in self.jobs:                          # новые задачи: первый шаг
                if j.state == 'queued' and not j._stop and not j.paused:
                    if j.kind == 'scan' and j.obj != getattr(self.m, 'objective', j.obj):
                        continue                         # ждёт свой объектив
                    self._advance(j)
            if self._inst is None:                       # прибор свободен: первая ждущая на приборе
                j = next((j for j in self.jobs if j.state == 'waiting' and j.lane == 'inst' and not j.paused), None)
                if j is not None:
                    self._start_inst(j)
            for j in [j for j in self._cpu if j._proc is not None and j._proc.poll() is not None
                      and getattr(j, '_reader_done', False)]:
                self._cpu.remove(j)
                self._end_cpu(j)
            for j in self._cpu:                          # остановили, а процесс не выходит: силой через 5 с
                if j._stop and getattr(j, '_t_stop', None) and time.monotonic() - j._t_stop > 5 and j._proc.poll() is None:
                    j._proc.kill()
            while len(self._cpu) < self.workers:
                j = next((j for j in self.jobs if j.state == 'waiting' and j.lane == 'cpu' and not j.paused), None)
                if j is None:
                    break
                self._start_cpu(j)

    def _start_inst(self, j):
        lane, title, fn, halt, est = j._next[:5]
        j.state, j.step_title, j.frac, j.text, j.left = 'running', title, 0.0, '', None
        j.step_t0, j.step_est, j._halt = time.monotonic(), est or 0.0, halt
        if j._stop:
            self._inst = None
            self._finish(j, 'stopped')
            return
        out = {}

        def run():
            from .util import Progress, job_begin, job_end
            from .errors import Stopped
            tid = threading.get_ident()
            j._tid = tid
            job_begin(tid)
            Progress.hooks[tid] = lambda t, f, txt, el, eta: self._hook(j, t, f, txt, eta)
            try:
                out['r'] = fn()
            except Stopped as e:
                out['stop'] = e
            except BaseException as e:
                out['e'] = e
            finally:
                Progress.hooks.pop(tid, None)
                job_end(tid)
                j._tid = None
                with self._lock:
                    self._inst = None
                    j._halt = None
                    if 'e' in out:
                        self._advance(j, exc=out['e'])
                    elif 'r' in out and j._stop:          # остановили, а шаг вернул снятое: задача его запишет
                        self._advance(j, out['r'])
                        if j.active:
                            self._finish(j, 'stopped', note=j.note or "stopped")
                            try:
                                j._gen.close()
                            except Exception:
                                pass
                    elif 'stop' in out or j._stop:
                        j._stop = True
                        self._finish(j, 'stopped', note=j.note or "stopped")
                        try:
                            j._gen.close()
                        except Exception:
                            pass
                    else:
                        self._advance(j, out.get('r'))
                self._wake.set()
        th = threading.Thread(target=run, daemon=True, name=f'mc-job-{j.id}')
        self._inst = (j, th)
        th.start()

    def _hook(self, j, t, f, txt, eta):
        j.frac, j.text, j.left = f, (txt or t), eta

    def _start_cpu(self, j):
        lane, title, spec, _h, est = j._next[:5]
        j.state, j.step_title, j.frac, j.text, j.left = 'running', title, 0.0, '', None
        j.step_t0, j.step_est = time.monotonic(), est or 0.0
        if j._stop:
            self._finish(j, 'stopped')
            return
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([here, os.environ.get('PYTHONPATH', '')]),
                   PYTHONIOENCODING='utf-8')
        kw = {}
        if os.name == 'nt':
            kw['creationflags'] = 0x08000000 | 0x00004000          # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY
        j._proc = subprocess.Popen([sys.executable, '-m', 'microcontrol.jobs', json.dumps(spec)], env=env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True, encoding='utf-8', **kw)
        j._reader_done, j._res = False, None
        self._cpu.append(j)

        def read():
            try:
                for ln in j._proc.stdout:
                    try:
                        msg = json.loads(ln)
                    except ValueError:
                        continue
                    if msg[0] == 'p':
                        self._hook(j, msg[1], msg[2], msg[3], msg[4])
                    else:
                        j._res = msg
            finally:
                j._reader_done = True
                self._wake.set()
        threading.Thread(target=read, daemon=True).start()

    def _end_cpu(self, j):
        res, spec = j._res or ['err', f"process ended ({j._proc.returncode})"], j._next[2]
        j._proc = None
        if res[0] == 'ok' and spec[0] == 'stitch':
            nf = n_frames(spec[1])
            if nf:
                _rates(spec[2].get('mode', 'quality'), (time.monotonic() - j.step_t0) / nf)
        if j._stop or res[0] == 'stopped':
            j._stop = True
            self._finish(j, 'stopped', note=j.note or "stopped")
            try:
                j._gen.close()
            except Exception:
                pass
        elif res[0] == 'ok':
            self._advance(j, res[1])
        else:
            self._advance(j, exc=RuntimeError(res[1]))


def _suspend(pid, on):
    """Заморозить (on) / разморозить процесс сшивки (пауза задачи на ядрах)."""
    try:
        import psutil
        p = psutil.Process(pid)
        p.suspend() if on else p.resume()
        return True
    except ImportError:
        pass
    except Exception:
        return False
    if os.name == 'nt':
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x0800, False, int(pid))     # PROCESS_SUSPEND_RESUME
        if not h:
            return False
        try:
            f = ctypes.windll.ntdll.NtSuspendProcess if on else ctypes.windll.ntdll.NtResumeProcess
            return f(h) == 0
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    import signal
    try:
        os.kill(pid, signal.SIGSTOP if on else signal.SIGCONT)
        return True
    except OSError:
        return False


def quick_partial(folder):
    """Прерванная съёмка: то, что снято, — quick-сшить сразу (швы живой сшивки готовы — секунды)."""
    try:
        from .stitch import stitch
        stitch(folder, mode='quick', verbose=False)
        return True
    except Exception as e:
        try:
            from . import log as L
            L.debug('JOB-030', f"quick of an interrupted scan {folder}: {type(e).__name__}: {e}")
        except Exception:
            pass
        return False


def unstitched(sample_folder):
    """Съёмки образца с кадрами, но без сшивки (окно упало посреди съёмки): их quick-сшить при запуске."""
    from .project import all_scans
    from .live import running
    out = []
    for f in all_scans(sample_folder):
        if n_frames(f) > 0 and os.path.exists(os.path.join(f, 'join', 'live.json')) \
                and not os.path.exists(os.path.join(f, 'join', 'stitch.json')) and not running(f):   # из очереди, не досшита
            out.append(f)
    return out


def n_frames_est(j, m):
    """Кадров у выделения задачи (для оценки сшивки до съёмки)."""
    if j.folder:
        return n_frames(j.folder)
    try:
        from . import samples as SM
        import numpy as np
        k = next(i for i, r in enumerate(m._regions) if r.get('uid') == j.uid)
        return int(np.prod(SM.frames_of(m, k)))
    except Exception:
        return 0


def _folder_obj(folder):
    try:
        return json.load(open(os.path.join(folder, 'meta.json'), encoding='utf-8')).get('objective')
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- процесс сшивки (python -m microcontrol.jobs spec)
def _child(spec):
    """Шаг на ядрах в своём процессе: прогресс и итог — строками JSON в stdout, «stop» в stdin — остановить."""
    proto = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    kind, folder, kw = spec
    try:
        log = open(os.path.join(folder, f'{kind}_log.txt'), 'w', encoding='utf-8')
    except OSError:
        log = open(os.devnull, 'w')
    sys.stdout = sys.stderr = log
    try:
        os.nice(5)
    except (AttributeError, OSError):
        pass
    from .util import Progress, stop_begin, stop_request
    from .errors import Stopped
    stop_begin()

    def watch():
        for ln in sys.stdin:
            if ln.strip() == 'stop':
                stop_request()
                return
    threading.Thread(target=watch, daemon=True).start()
    last = [0.0]

    def hook(t, f, txt, el, eta):
        now = time.monotonic()
        if now - last[0] < 0.2 and f < 1:
            return
        last[0] = now
        proto.write(json.dumps(['p', t, f, txt, eta]) + "\n")
    Progress.hook = hook
    try:
        if kind == 'stitch':
            from .stitch import stitch
            stitch(folder, verbose=True, **kw)
        else:
            raise ValueError(f"unknown job {kind}")
        proto.write(json.dumps(['ok', None]) + "\n")
    except Stopped:
        proto.write(json.dumps(['stopped', None]) + "\n")
    except BaseException as e:
        proto.write(json.dumps(['err', f"{type(e).__name__}: {e}"]) + "\n")
    finally:
        log.flush()
        proto.flush()


if __name__ == '__main__':
    _child(json.loads(sys.argv[1]))
