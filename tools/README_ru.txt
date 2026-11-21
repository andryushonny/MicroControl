AtrSniffer.exe — полная запись обмена ПК <-> контроллер ATR8300 (USB, через USBPcap) и разбор в таблицу.
Нужен только .NET Framework 4.x (есть в Windows 10).

ЧТО НОВОГО
  Теперь в записи видны и служебные USB-запросы порта:
    SET_CONTROL_LINE_STATE DTR=.. RTS=..   — линии DTR/RTS (их включает/выключает программа при открытии порта)
    LINE_CODING baud=115200 ...           — скорость и формат порта
  Это как раз то, что программы делают «до» первой команды.

ГЛАВНОЕ: запись должна начаться ДО запуска программы, иначе подключение не попадёт в файл
(в прошлых записях его не было: все они начинались с середины сеанса).

ОДИН РАЗ
  1) Установить Wireshark, в установщике отметить USBPcap. Перезагрузить ПК.

ЗАПИСЬ 1 — штатное ПО Optosky (эталон)
  1) Закрыть ПО Optosky и все ноутбуки MicroControl.
  2) Запустить от администратора: C:\Program Files\USBPcap\USBPcapCMD.exe
     - выбрать корневой хаб, где виден контроллер (STMicroelectronics Virtual COM Port, VID_0483 PID_5740);
     - устройства: оставить ТОЛЬКО контроллер (не камеру и не флешку);
     - файл: D:\Sniffer\optosky.pcap. Запись пошла.
  3) Запустить ПО Optosky, дождаться подключения и шума вентилятора.
  4) Сделать по одному шагу: X вправо, X влево, Y вперёд, Y назад, Z вверх, Z вниз. Свет выключить и включить.
  5) Закрыть ПО Optosky (через крестик, обычным образом).
  6) Ctrl+C в окне USBPcapCMD.

ЗАПИСЬ 2 — MicroControl (то же самое, для сравнения)
  1) Так же запустить USBPcapCMD, файл D:\Sniffer\microcontrol.pcap.
  2) В cmd:  cd /d D:\MicroControl  ->  .venv\Scripts\activate  ->  python -m microcontrol probe
  3) В ноутбуке: m = mc.Microscope(objective=10, camera=False); m.move(dx=200); m.move(dx=-200); m.close()
  4) Ctrl+C в окне USBPcapCMD.

РАЗБОР
      AtrSniffer.exe pcap D:\Sniffer\optosky.pcap
      AtrSniffer.exe pcap D:\Sniffer\microcontrol.pcap
  Результат — в папке logs рядом с exe: atr_frames_*.csv (по строке на кадр и на USB-запрос) и atr_raw_*.log.
  Если в записи несколько устройств, программа сама оставит то, где есть кадры AA 55; вручную: --device N.

ПРИСЛАТЬ: оба файла atr_frames_*.csv (и .pcap, если не очень большие).
По ним видно всё, что отличается: линии DTR/RTS, порядок и содержимое команд, ответы и паузы.

Другое:
   AtrSniffer.exe list       - COM-порты и порт прибора (0483:5740)
   AtrSniffer.exe selftest   - проверка разборщика кадров
