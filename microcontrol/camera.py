"""Камера MindVision MV-SUA502C через mvsdk.py (MVCAMSDK_X64.dll)."""
import os
import platform
import numpy as np
from .errors import CameraError

SDK_DIRS = [   # где искать DLL, если SDK не прописан в PATH
    r"C:\Program Files\MindVision\SDK\X64",
    r"C:\Program Files (x86)\MindVision\SDK\X64",
    os.path.join(os.path.dirname(__file__), "SDK", "X64"),
]

SDK_HINT = ("Download and install the MindVision camera SDK (Windows x64) from the camera maker, mindvision.com.cn "
            "(Downloads -> SDK), run the installer as administrator, restart the PC. It is needed only on the PC "
            "the camera is plugged into; from another PC use Microscope('net://KEY@auto').")


def _load_sdk(sdk_folder=None):
    for d in ([sdk_folder] if sdk_folder else []) + SDK_DIRS:
        if d and os.path.isfile(os.path.join(d, "MVCAMSDK_X64.dll")):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
            break
    from . import mvsdk
    return mvsdk


class MVCamera:
    """Обёртка: открыть камеру, взять кадр, выставить экспозицию/усиление/баланс белого."""
    def __init__(self, sdk_folder=None):
        try:
            self.sdk = mv = _load_sdk(sdk_folder)
        except OSError as e:
            raise CameraError('E502', SDK_HINT + " Or pass sdk_folder=<SDK\\X64 folder>.") from e
        try:
            self._open(mv)
        except CameraError:
            raise
        except OSError as e:       # SDK есть, но не установлен драйвер/регистрация: падает внутри DLL
            raise CameraError('E502', "the SDK crashed while opening the camera (" + str(e) + "). " + SDK_HINT) from e
        except Exception as e:
            code = getattr(e, 'error_code', None)
            if code == -45:
                raise CameraError('E504', "close the Optosky software, other MindVision programs and other Python "
                                          "windows that hold it (Task Manager), or run the program through "
                                          "net://... if the camera is on another PC.") from e
            if code in (-44, -5, -2):
                raise CameraError('E502', "the SDK is not installed correctly (code %s). " % code + SDK_HINT) from e
            raise

    def _open(self, mv):
        devs = mv.CameraEnumerateDevice()
        if not devs:
            raise CameraError('E501', "check the USB cable, close the Optosky software; if the camera was never "
                                      "used on this PC, install the MindVision SDK (see E502)")
        self.name = devs[0].GetFriendlyName()
        self.h = mv.CameraInit(devs[0], -1, -1)
        cap = mv.CameraGetCapability(self.h)
        self.mono = cap.sIspCapacity.bMonoSensor != 0
        mv.CameraSetIspOutFormat(self.h, mv.CAMERA_MEDIA_TYPE_MONO8 if self.mono
                                 else mv.CAMERA_MEDIA_TYPE_BGR8)
        # Отражение и поворот кадра в самой камере: другие программы (штатная Optosky) могут оставить их
        # включёнными, и картинка придёт перевёрнутой. Всегда сбрасываем — ориентацию задаёт только MicroControl
        # (turn_180, в ручном режиме клавиша T). Что было — в self.was (для журнала).
        self.was = {}
        for d_, name_ in ((0, 'mirror_h'), (1, 'mirror_v')):
            try:
                self.was[name_] = bool(mv.CameraGetMirror(self.h, d_))
                mv.CameraSetMirror(self.h, d_, 0)
            except Exception:
                pass
        try:
            self.was['rotate'] = int(mv.CameraGetRotate(self.h))
            mv.CameraSetRotate(self.h, 0)
        except Exception:
            pass
        mv.CameraSetTriggerMode(self.h, 0)            # непрерывный поток
        mv.CameraSetSharpness(self.h, 0)              # без программной резкости (важно для автофокуса)
        mv.CameraSetAeState(self.h, 0)                # ручная экспозиция
        try:
            mv.CameraSetWbMode(self.h, 0)             # ручной баланс белого: цвет кадров скана не плывёт
        except Exception:
            pass
        mv.CameraSetExposureTime(self.h, 20 * 1000)   # 20 мс, как в штатном ПО
        mv.CameraPlay(self.h)
        size = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if self.mono else 3)
        self.buf = mv.CameraAlignMalloc(size, 16)

    def grab(self, flush=2):
        """Свежий кадр numpy (BGR или GRAY). flush — сколько старых кадров пропустить."""
        mv = self.sdk
        for _ in range(flush + 1):
            raw, head = mv.CameraGetImageBuffer(self.h, 3000)
            mv.CameraImageProcess(self.h, raw, self.buf, head)
            mv.CameraReleaseImageBuffer(self.h, raw)
        if platform.system() == "Windows":
            mv.CameraFlipFrameBuffer(self.buf, head, 1)
        data = (mv.c_ubyte * head.uBytes).from_address(self.buf)
        shape = (head.iHeight, head.iWidth) if self.mono else (head.iHeight, head.iWidth, 3)
        return np.frombuffer(data, np.uint8).reshape(shape).copy()

    def set(self, exposure_ms=None, gain=None, white_balance=False):
        """Экспозиция (мс или 'auto'), аналоговое усиление, однократный баланс белого."""
        mv = self.sdk
        if exposure_ms == 'auto':
            mv.CameraSetAeState(self.h, 1)
        elif exposure_ms is not None:
            if not 0.05 <= float(exposure_ms) <= 1000:
                raise CameraError('E504', f"{exposure_ms} ms, allowed 0.05…1000")
            mv.CameraSetAeState(self.h, 0)
            mv.CameraSetExposureTime(self.h, float(exposure_ms) * 1000)
        if gain is not None:
            mv.CameraSetAnalogGain(self.h, int(gain))
        if white_balance and not self.mono:
            mv.CameraSetOnceWB(self.h)
        return self.info()

    def freeze(self):
        """Зафиксировать текущую автоэкспозицию (для одинаковой яркости тайлов карты)."""
        mv = self.sdk
        if mv.CameraGetAeState(self.h):
            t = mv.CameraGetExposureTime(self.h)
            mv.CameraSetAeState(self.h, 0)
            mv.CameraSetExposureTime(self.h, t)
        try:
            mv.CameraSetWbMode(self.h, 0)                 # баланс белого — тоже вручную на весь скан
        except Exception:
            pass

    def info(self):
        mv = self.sdk
        return dict(model=self.name, auto=bool(mv.CameraGetAeState(self.h)),
                    exposure_ms=mv.CameraGetExposureTime(self.h) / 1000,
                    gain=mv.CameraGetAnalogGain(self.h))

    def close(self):
        if getattr(self, "h", None):
            self.sdk.CameraUnInit(self.h)
            self.sdk.CameraAlignFree(self.buf)
            self.h = None
