// AtrSniffer.exe — прозрачный COM-прокси с журналом обмена ПО Optosky <-> прибор ATR8300.
// Работает на Windows 10 без установки Python и библиотек (нужен только .NET Framework 4.x, он есть в Win10).
//
//   AtrSniffer.exe list
//   AtrSniffer.exe proxy --virtual COM21 [--real COM5|auto] [--baud 115200] [--out папка]
//   AtrSniffer.exe selftest
//
// Схема: штатное ПО открывает COM20, виртуальная пара com0com соединяет COM20<->COM21,
// AtrSniffer держит COM21 и реальный порт прибора, пересылает байты в обе стороны и пишет журнал.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Ports;
using System.Text;
using System.Threading;
using System.Diagnostics;
using Microsoft.Win32;

static class Program
{
    const string HeaderLine = "t_ms;dir;cmd;cmd_name;len;checksum;data_hex;frame_hex";

    static readonly Dictionary<byte, string> CmdNames = new Dictionary<byte, string> {
        {0x03,"GET_PN"},{0x04,"GET_SN"},{0x13,"GET_TEC_TEMP"},{0x12,"SET_TEC_TEMP"},
        {0x20,"SET_LASER_POWER"},{0x16,"START_ACQUIRE"},{0x23,"START_ACQUIRE_DARK"},
        {0x17,"READ_OUT_CCD_DATA"},{0x31,"GET_CALIBRATION_DATA_INTEGER"},
        {0x30,"SET_CALIBRATION_DATA_INTEGER"},{0x2D,"SET_LIGHT_VOLTAGE"},
        {0x2E,"SET_X_MOTOR"},{0x2F,"SET_Y_MOTOR"},{0x2B,"SET_Z_MOTOR"},{0x24,"TURN_MOTOR"},
        {0x14,"SET_INTEGRATION_TIME"},{0x65,"SET_MAX_LASER_POWER"},{0x66,"GET_MAX_LASER_POWER"},
        {0x25,"SET_DEVICE_TYPE"},{0x26,"GET_DEVICE_TYPE"}
    };

    // ---------- разбор потока на кадры: AA 55 | LEN(2,BE) | CMD | DATA | SUM8 ; LEN = 4+len(DATA) ----------
    class FrameParser
    {
        readonly List<byte> buf = new List<byte>();
        public delegate void FrameCb(byte[] frame, bool checksumOk, bool isFrame);
        public void Feed(byte[] data, int n, FrameCb cb)
        {
            for (int i = 0; i < n; i++) buf.Add(data[i]);
            while (true)
            {
                // ищем заголовок
                int h = -1;
                for (int i = 0; i + 1 < buf.Count; i++)
                    if (buf[i] == 0xAA && buf[i + 1] == 0x55) { h = i; break; }
                if (h < 0)
                {
                    // всё, кроме последнего байта (может быть началом заголовка), — не кадр
                    int keep = (buf.Count > 0 && buf[buf.Count - 1] == 0xAA) ? 1 : 0;
                    int drop = buf.Count - keep;
                    if (drop > 0) { cb(buf.GetRange(0, drop).ToArray(), false, false); buf.RemoveRange(0, drop); }
                    return;
                }
                if (h > 0) { cb(buf.GetRange(0, h).ToArray(), false, false); buf.RemoveRange(0, h); }
                if (buf.Count < 4) return;
                int len = (buf[2] << 8) | buf[3];
                if (len < 4) { cb(new byte[] { buf[0] }, false, false); buf.RemoveAt(0); continue; }
                int total = len + 2;
                if (buf.Count < total) return;
                byte[] f = buf.GetRange(0, total).ToArray();
                int sum = 0;
                for (int i = 2; i < total - 1; i++) sum += f[i];
                bool ok = (byte)sum == f[total - 1];
                buf.RemoveRange(0, total);
                cb(f, ok, true);
            }
        }
    }

    static string Hex(byte[] b, int max)
    {
        var sb = new StringBuilder();
        int n = Math.Min(b.Length, max);
        for (int i = 0; i < n; i++) { if (i > 0) sb.Append(' '); sb.Append(b[i].ToString("X2")); }
        if (b.Length > n) sb.Append(" ...(" + b.Length + " bytes)");
        return sb.ToString();
    }

    class Logger
    {
        readonly object lk = new object();
        readonly StreamWriter csv, raw;
        readonly Stopwatch sw = Stopwatch.StartNew();
        public readonly string CsvPath, RawPath;
        public Logger(string dir)
        {
            Directory.CreateDirectory(dir);
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            CsvPath = Path.Combine(dir, "atr_frames_" + stamp + ".csv");
            RawPath = Path.Combine(dir, "atr_raw_" + stamp + ".log");
            csv = new StreamWriter(new FileStream(CsvPath, FileMode.Create, FileAccess.Write, FileShare.ReadWrite), new UTF8Encoding(true));
            raw = new StreamWriter(new FileStream(RawPath, FileMode.Create, FileAccess.Write, FileShare.ReadWrite), new UTF8Encoding(false));
            csv.WriteLine(HeaderLine); csv.Flush();
        }
        public double Now() { return sw.Elapsed.TotalMilliseconds; }
        public void Chunk(string dir, byte[] b, int n) { ChunkAt(Now(), dir, b, n); }
        public void ChunkAt(double t, string dir, byte[] b, int n)
        {
            var c = new byte[n]; Array.Copy(b, c, n);
            lock (lk) { raw.WriteLine(t.ToString("F3", CultureInfo.InvariantCulture) + " " + dir + " " + Hex(c, 256)); raw.Flush(); }
        }
        public void Frame(string dir, byte[] f, bool ok, bool isFrame) { FrameAt(Now(), dir, f, ok, isFrame); }
        public void FrameAt(double t, string dir, byte[] f, bool ok, bool isFrame)
        {
            string line;
            string ts = t.ToString("F3", CultureInfo.InvariantCulture);
            if (isFrame)
            {
                byte cmd = f[4];
                string name; if (!CmdNames.TryGetValue(cmd, out name)) name = "UNKNOWN";
                int len = (f[2] << 8) | f[3];
                byte[] data = new byte[Math.Max(0, f.Length - 6)];
                Array.Copy(f, 5, data, 0, data.Length);
                line = ts + ";" + dir + ";0x" + cmd.ToString("X2") + ";" + name + ";" + len + ";" + (ok ? "OK" : "BAD") +
                       ";" + Hex(data, 64) + ";" + Hex(f, 64);
            }
            else
                line = ts + ";" + dir + ";;NOT_A_FRAME;" + f.Length + ";;;" + Hex(f, 64);
            lock (lk) { csv.WriteLine(line); csv.Flush(); }
            Console.WriteLine(dir + " " + (isFrame ? ("0x" + f[4].ToString("X2") + " " + Hex(f, 24) + (ok ? "" : "  [BAD CHECKSUM]")) : ("raw " + Hex(f, 24))));
        }
        public void NoteAt(double t, string text)
        {
            string ts = t.ToString("F3", CultureInfo.InvariantCulture);
            lock (lk) { csv.WriteLine(ts + ";CONTROL;;" + text + ";;;;"); raw.WriteLine(ts + " CONTROL " + text); csv.Flush(); raw.Flush(); }
            Console.WriteLine("CONTROL " + text);
        }
        public void Close() { lock (lk) { csv.Close(); raw.Close(); } }
    }

    // ---------- поиск реального порта контроллера по USB ID (реестр, без доп. библиотек) ----------
    static List<string> FindPortsByVidPid(string vidpid)
    {
        var res = new List<string>();
        try
        {
            using (RegistryKey usb = Registry.LocalMachine.OpenSubKey(@"SYSTEM\CurrentControlSet\Enum\USB\" + vidpid))
            {
                if (usb == null) return res;
                foreach (string inst in usb.GetSubKeyNames())
                    using (RegistryKey dp = usb.OpenSubKey(inst + @"\Device Parameters"))
                    {
                        if (dp == null) continue;
                        object pn = dp.GetValue("PortName");
                        if (pn != null) res.Add(pn.ToString());
                    }
            }
        }
        catch { }
        return res;
    }

    static string FindRealPort()
    {
        var present = new HashSet<string>(SerialPort.GetPortNames(), StringComparer.OrdinalIgnoreCase);
        foreach (string p in FindPortsByVidPid("VID_0483&PID_5740"))
            if (present.Contains(p)) return p;
        return null;
    }

    // ---------- прокси ----------
    static volatile bool stop;

    static void Pump(string dir, Stream src, Stream dst, Logger log)
    {
        var parser = new FrameParser();
        var buf = new byte[8192];
        while (!stop)
        {
            int n = 0;
            try { n = src.Read(buf, 0, buf.Length); }
            catch (TimeoutException) { continue; }
            catch (Exception e) { if (!stop) Console.WriteLine("[" + dir + "] read error: " + e.Message); Thread.Sleep(50); continue; }
            if (n <= 0) { Thread.Sleep(1); continue; }
            // сначала пересылаем (минимальная задержка), потом логируем
            try { dst.Write(buf, 0, n); dst.Flush(); }
            catch (Exception e) { Console.WriteLine("[" + dir + "] write error: " + e.Message); }
            log.Chunk(dir, buf, n);
            parser.Feed(buf, n, (f, ok, isF) => log.Frame(dir, f, ok, isF));
        }
    }

    static SerialPort Open(string name, int baud)
    {
        var p = new SerialPort(name, baud, Parity.None, 8, StopBits.One);
        p.ReadTimeout = 200; p.WriteTimeout = 2000;
        p.ReadBufferSize = 1 << 16; p.WriteBufferSize = 1 << 16;
        p.Handshake = Handshake.None; p.DtrEnable = true; p.RtsEnable = true;
        p.Open();
        return p;
    }

    static int RunProxy(Dictionary<string, string> a)
    {
        string virt; if (!a.TryGetValue("virtual", out virt)) { Console.WriteLine("Нужен --virtual COMxx (второй конец пары com0com)"); return 2; }
        string real; if (!a.TryGetValue("real", out real)) real = "auto";
        if (real.ToLower() == "auto") real = FindRealPort();
        if (real == null) { Console.WriteLine("Реальный порт не найден (STM32 Virtual COM 0483:5740). Укажите --real COMxx. Порты: " + string.Join(", ", SerialPort.GetPortNames())); return 2; }
        int baud = 115200; string bs; if (a.TryGetValue("baud", out bs)) baud = int.Parse(bs);
        string outDir; if (!a.TryGetValue("out", out outDir)) outDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "logs");

        var log = new Logger(outDir);
        Console.WriteLine("Реальный порт: " + real + " @" + baud + "   виртуальный: " + virt);
        Console.WriteLine("Журналы: " + log.CsvPath + " , " + log.RawPath);
        SerialPort pv = null, pr = null;
        Stream sv = null, sr = null;
        try
        {
            if (a.ContainsKey("filetest")) // отладочный режим: обычные файлы/pty вместо COM (для тестов на Linux)
            {
                sr = new FileStream(real, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite, 1, false);
                sv = new FileStream(virt, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite, 1, false);
            }
            else { pr = Open(real, baud); pv = Open(virt, baud); sr = pr.BaseStream; sv = pv.BaseStream; }
        }
        catch (Exception e)
        {
            Console.WriteLine("Не удалось открыть порт: " + e.Message + "\nЗакройте штатное ПО (оно держит реальный порт) и проверьте имя виртуального порта.");
            return 3;
        }
        Console.CancelKeyPress += (s, e) => { e.Cancel = true; stop = true; };
        var t1 = new Thread(() => Pump("APP->DEV", sv, sr, log)) { IsBackground = true };
        var t2 = new Thread(() => Pump("DEV->APP", sr, sv, log)) { IsBackground = true };
        t1.Start(); t2.Start();
        Console.WriteLine("Работаю. Запустите штатное ПО, выберите в нём виртуальный порт. Ctrl+C — остановить.");
        while (!stop) Thread.Sleep(100);
        t1.Join(1000); t2.Join(1000);
        try { if (pv != null) pv.Close(); if (pr != null) pr.Close(); } catch { }
        log.Close();
        Console.WriteLine("Остановлено. Файлы журнала сохранены.");
        return 0;
    }


    // ---------- разбор записи USBPcap (Wireshark .pcapng или USBPcapCMD .pcap) ----------
    class UsbPacket { public double T; public int Bus, Dev, Ep; public bool In; public byte[] Data; public string Note; }

    static uint U32(byte[] b, int o) { return (uint)(b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)); }
    static int U16(byte[] b, int o) { return b[o] | (b[o + 1] << 8); }

    static void AddUsb(List<UsbPacket> res, double t, byte[] d, int off, int len)
    {
        if (len < 27) return;
        int hl = U16(d, off);
        if (hl < 27 || off + hl > d.Length) return;
        int info = d[off + 16], bus = U16(d, off + 17), dev = U16(d, off + 19), ep = d[off + 21], tr = d[off + 22];
        int dl = (int)U32(d, off + 23);
        if (tr == 2 && (info & 1) == 0 && hl >= 28 && dl > 0)   // control-запрос хоста: линии DTR/RTS, скорость порта
        {
            int stage = d[off + 27], st = off + hl;
            if (st + Math.Min(dl, 8) > off + len) return;
            string note = null;
            if (stage == 0 && dl >= 8)
            {
                int rt = d[st], rq = d[st + 1], wv = U16(d, st + 2), wi = U16(d, st + 4), wl = U16(d, st + 6);
                if (rt == 0x21 && rq == 0x22) note = "SET_CONTROL_LINE_STATE DTR=" + (wv & 1) + " RTS=" + ((wv >> 1) & 1);
                else if (rt == 0x21 && rq == 0x20) note = "SET_LINE_CODING (port settings follow)";
                else if (rt == 0x21 && rq == 0x23) note = "SEND_BREAK " + wv + " ms";
                else if (rt == 0xA1 && rq == 0x21) note = "GET_LINE_CODING";
                else if (rt == 0x80 && rq == 0x06) note = "GET_DESCRIPTOR type " + (wv >> 8);
                else if (rt == 0x00 && rq == 0x09) note = "SET_CONFIGURATION " + wv;
                else note = string.Format("CONTROL bmRequestType=0x{0:X2} bRequest=0x{1:X2} wValue=0x{2:X4} wIndex={3} wLength={4}", rt, rq, wv, wi, wl);
            }
            else if (stage == 1 && dl == 7)                       // данные SET_LINE_CODING
                note = "LINE_CODING baud=" + U32(d, st) + " stop=" + new[] { "1", "1.5", "2" }[Math.Min(2, (int)d[st + 4])] +
                       " parity=" + new[] { "none", "odd", "even", "mark", "space" }[Math.Min(4, (int)d[st + 5])] + " bits=" + d[st + 6];
            if (note != null)
                res.Add(new UsbPacket { T = t, Bus = bus, Dev = dev, Ep = 0, In = false, Data = new byte[0], Note = note });
            return;
        }
        if (tr != 3 || dl <= 0) return;                       // дальше — только bulk с данными (CDC)
        bool isIn = (ep & 0x80) != 0, completion = (info & 1) != 0;
        if (isIn != completion) return;                        // IN: данные в завершении, OUT: в запросе
        int avail = Math.Min(dl, off + len - (off + hl));
        if (avail <= 0) return;
        var p = new byte[avail]; Array.Copy(d, off + hl, p, 0, avail);
        res.Add(new UsbPacket { T = t, Bus = bus, Dev = dev, Ep = ep & 0x7F, In = isIn, Data = p });
    }

    static List<UsbPacket> ReadCapture(byte[] f)
    {
        var res = new List<UsbPacket>();
        if (f.Length < 24) return res;
        uint magic = U32(f, 0);
        if (magic == 0xA1B2C3D4 || magic == 0xA1B23C4D)         // классический pcap, little-endian
        {
            double div = magic == 0xA1B23C4D ? 1e9 : 1e6;
            int link = (int)U32(f, 20);
            if (link != 249) Console.WriteLine("Предупреждение: linktype=" + link + " (ожидался 249 = USBPcap)");
            int o = 24;
            while (o + 16 <= f.Length)
            {
                double t = U32(f, o) + U32(f, o + 4) / div;
                int incl = (int)U32(f, o + 8);
                if (o + 16 + incl > f.Length) break;
                AddUsb(res, t, f, o + 16, incl);
                o += 16 + incl;
            }
        }
        else if (magic == 0x0A0D0D0A)                            // pcapng
        {
            var res_div = new List<double>(); var links = new List<int>();
            int o = 0;
            while (o + 12 <= f.Length)
            {
                uint type = U32(f, o); int blen = (int)U32(f, o + 4);
                if (blen < 12 || o + blen > f.Length) break;
                if (type == 1)                                   // Interface Description
                {
                    links.Add(U16(f, o + 8)); double div = 1e6;
                    int p = o + 16;                              // опции
                    while (p + 4 <= o + blen - 4)
                    {
                        int code = U16(f, p), ol = U16(f, p + 2);
                        if (code == 0) break;
                        if (code == 9 && ol >= 1) { int v = f[p + 4]; div = (v & 0x80) != 0 ? Math.Pow(2, v & 0x7F) : Math.Pow(10, v); }
                        p += 4 + ((ol + 3) & ~3);
                    }
                    res_div.Add(div);
                }
                else if (type == 6)                              // Enhanced Packet
                {
                    int ifc = (int)U32(f, o + 8);
                    double div = ifc < res_div.Count ? res_div[ifc] : 1e6;
                    double t = (((ulong)U32(f, o + 12) << 32) | U32(f, o + 16)) / div;
                    int cap = (int)U32(f, o + 20);
                    if (ifc < links.Count && links[ifc] == 249) AddUsb(res, t, f, o + 28, cap);
                }
                o += (blen + 3) & ~3;
            }
        }
        else Console.WriteLine("Неизвестный формат файла (нужен .pcap/.pcapng из USBPcap/Wireshark)");
        return res;
    }

    static int RunPcap(string file, Dictionary<string, string> a)
    {
        byte[] data;
        try { data = File.ReadAllBytes(file); } catch (Exception e) { Console.WriteLine("Не удалось прочитать файл: " + e.Message); return 2; }
        var pk = ReadCapture(data);
        Console.WriteLine("Пакетов bulk с данными: " + pk.Count);
        if (pk.Count == 0) return 3;
        double t0 = pk[0].T;
        string outDir; if (!a.TryGetValue("out", out outDir)) outDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "logs");
        string want; a.TryGetValue("device", out want);

        // проход 1: какие USB-устройства дают валидные кадры AA 55
        var parsers = new Dictionary<string, FrameParser>();
        var okCount = new Dictionary<string, int>();
        var events = new List<object[]>();                       // t, key, dir, frame, ok, isFrame
        foreach (var p in pk)
        {
            string key = p.Bus + ":" + p.Dev;
            if (want != null && want != p.Dev.ToString() && want != key) continue;
            if (p.Note != null) { events.Add(new object[] { (p.T - t0) * 1000.0, key, "CONTROL", p.Note, true, false, false }); continue; }
            FrameParser fp; if (!parsers.TryGetValue(key + (p.In ? "i" : "o"), out fp)) { fp = new FrameParser(); parsers[key + (p.In ? "i" : "o")] = fp; }
            string dir = p.In ? "DEV->APP" : "APP->DEV";
            double tt = (p.T - t0) * 1000.0;
            events.Add(new object[] { tt, key, dir, p.Data, true, false, true }); // сырой кусок
            fp.Feed(p.Data, p.Data.Length, (fr, ok, isF) => {
                events.Add(new object[] { tt, key, dir, fr, ok, isF, false });
                if (isF && ok) { int c; okCount.TryGetValue(key, out c); okCount[key] = c + 1; }
            });
        }
        var good = new HashSet<string>(okCount.Keys);
        foreach (var kv in okCount) Console.WriteLine("Устройство bus:addr " + kv.Key + " — кадров AA 55 с верной суммой: " + kv.Value);
        if (good.Count == 0) { Console.WriteLine("Кадры AA 55 не найдены. Укажите --device N (адрес USB-устройства) или проверьте запись."); return 4; }

        var log = new Logger(outDir);
        foreach (var e in events)
        {
            string key = (string)e[1];
            if (!good.Contains(key)) continue;
            if ((string)e[2] == "CONTROL") log.NoteAt((double)e[0], (string)e[3]);
            else if ((bool)e[6]) log.ChunkAt((double)e[0], (string)e[2], (byte[])e[3], ((byte[])e[3]).Length);
            else log.FrameAt((double)e[0], (string)e[2], (byte[])e[3], (bool)e[4], (bool)e[5]);
        }
        log.Close();
        Console.WriteLine("Готово: " + log.CsvPath + " , " + log.RawPath);
        return 0;
    }

    // ---------- самопроверка разборщика ----------
    static int SelfTest()
    {
        int fails = 0;
        var p = new FrameParser();
        var got = new List<string>();
        byte[] s = { 0x00, 0xAA, 0x55, 0x00, 0x05, 0x24, 0x01, 0x2A, 0xAA, 0x55, 0x00, 0x07, 0x2E, 0x00, 0x00, 0x12, 0x47, 0xAA, 0x55, 0x00, 0x05, 0x24, 0x01, 0xFF };
        // подаём по кусочкам, чтобы проверить склейку
        for (int i = 0; i < s.Length; i += 3)
        {
            int n = Math.Min(3, s.Length - i);
            var chunk = new byte[n]; Array.Copy(s, i, chunk, 0, n);
            p.Feed(chunk, n, (f, ok, isF) => got.Add((isF ? "F" : "R") + (ok ? "+" : "-") + f.Length));
        }
        string r = string.Join(",", got.ToArray());
        string exp = "R-1,F+7,F+9,F-7";
        Console.WriteLine("selftest: " + r + (r == exp ? "  OK" : "  FAIL (ожидалось " + exp + ")"));
        if (r != exp) fails++;
        return fails;
    }

    static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("AtrSniffer: list | proxy --virtual COMxx [--real COMyy|auto] [--baud N] [--out dir] | pcap файл [--device N] [--out dir] | selftest");
            return 1;
        }
        var a = new Dictionary<string, string>();
        if (args[0] == "pcap")
        {
            if (args.Length < 2) { Console.WriteLine("Использование: AtrSniffer.exe pcap файл.pcapng [--device N] [--out папка]"); return 1; }
            for (int i = 2; i < args.Length; i++)
                if (args[i].StartsWith("--") && i + 1 < args.Length) { a[args[i].Substring(2)] = args[i + 1]; i++; }
            return RunPcap(args[1], a);
        }
        for (int i = 1; i < args.Length; i++)
            if (args[i].StartsWith("--") && i + 1 < args.Length) { a[args[i].Substring(2)] = args[i + 1]; i++; }
        switch (args[0])
        {
            case "list":
                Console.WriteLine("COM-порты: " + string.Join(", ", SerialPort.GetPortNames()));
                string rp = FindRealPort();
                Console.WriteLine("Порт контроллера ATR (0483:5740): " + (rp ?? "не найден"));
                return 0;
            case "proxy": return RunProxy(a);
            case "selftest": return SelfTest();
            default: Console.WriteLine("Неизвестная команда"); return 1;
        }
    }
}
