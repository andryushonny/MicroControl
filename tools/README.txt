AtrSniffer.exe - full recording of the PC <-> ATR8300 controller exchange (USB, via USBPcap) and its parsing into a table.
Only .NET Framework 4.x is needed (it comes with Windows 10). Source code: AtrSniffer.cs.

The recording shows the USB control requests of the port as well:
    SET_CONTROL_LINE_STATE DTR=.. RTS=..   - DTR/RTS lines (the program sets them when it opens the port)
    LINE_CODING baud=115200 ...           - port speed and format
That is what programs do "before" the first command.

IMPORTANT: start the recording BEFORE the program, otherwise the connection is not in the file.

ONCE
  1) Install Wireshark, tick USBPcap in the installer. Restart the PC.

RECORDING 1 - the stock Optosky software (reference)
  1) Close the Optosky software and all MicroControl notebooks.
  2) Run as administrator: C:\Program Files\USBPcap\USBPcapCMD.exe
     - choose the root hub where the controller is seen (STMicroelectronics Virtual COM Port, VID_0483 PID_5740);
     - devices: keep ONLY the controller (not the camera, not a USB stick);
     - file: D:\Sniffer\optosky.pcap. Recording starts.
  3) Start the Optosky software, wait for the connection and the fan noise.
  4) Make one step each: X right, X left, Y forward, Y back, Z up, Z down. Turn the light off and on.
  5) Close the Optosky software (with the window cross, the usual way).
  6) Ctrl+C in the USBPcapCMD window.

RECORDING 2 - MicroControl (the same, for comparison)
  1) Start USBPcapCMD the same way, file D:\Sniffer\microcontrol.pcap.
  2) In cmd:  cd /d D:\MicroControl  ->  .venv\Scripts\activate  ->  python -m microcontrol probe
  3) In the notebook: m = mc.Microscope(objective=10, camera=False); m.move(dx=200); m.move(dx=-200); m.close()
  4) Ctrl+C in the USBPcapCMD window.

PARSING
      AtrSniffer.exe pcap D:\Sniffer\optosky.pcap
      AtrSniffer.exe pcap D:\Sniffer\microcontrol.pcap
  The result is in the logs folder next to the exe: atr_frames_*.csv (one line per frame and per USB request) and atr_raw_*.log.
  If the recording has several devices, the program keeps the one with AA 55 frames; by hand: --device N.

Comparing the two atr_frames_*.csv files shows every difference: DTR/RTS lines, order and contents of the commands, replies and pauses.

Other:
   AtrSniffer.exe list       - COM ports and the instrument port (0483:5740)
   AtrSniffer.exe selftest   - frame parser self-test

Русская версия: README_ru.txt.
