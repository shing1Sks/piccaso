"""Original soundtrack and sound design for the research film, synthesised from scratch (no samples, nothing to license).

120 bpm with bars starting at 1.5 s, so most scene cuts land on a downbeat. Every sound effect is placed at the same time as the
animation event in index.html. Writes music.wav (44.1 kHz stereo, 106 s); see README for muxing it onto the render.

  python make_music.py
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, istft, sosfilt

SR = 44100
DUR = 106.0
N = int(SR * DUR)
BEAT, BAR0 = 0.5, 1.5
rng = np.random.default_rng(7)


def T(sec):
    return np.arange(int(sec * SR)) / SR


def filt(x, kind, f, order=2):
    return sosfilt(butter(order, f, kind, fs=SR, output="sos"), x, axis=-1)


def fade(x, a=0.003, r=0.02):
    n = x.shape[-1]
    e = np.ones(n)
    na, nr = min(n, int(a * SR)), min(n, int(r * SR))
    if na:
        e[:na] = np.linspace(0, 1, na)
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return x * e


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def noise(sec):
    return rng.standard_normal(int(sec * SR))


class Bus:
    def __init__(self):
        self.x = np.zeros((2, N))

    def add(self, sig, t, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if sig.ndim == 1:
            p = np.asarray(pan, dtype=float)
            sig = np.stack([sig * np.cos((p + 1) * np.pi / 4), sig * np.sin((p + 1) * np.pi / 4)]) * np.sqrt(2)
        if i < 0:
            sig, i = sig[:, -i:], 0
        n = min(sig.shape[1], N - i)
        if n > 0:
            self.x[:, i:i + n] += gain * sig[:, :n]


# ---------------------------------------------------------------- instruments
def kick():
    t = T(0.5)
    f = 45 + 115 * np.exp(-t / 0.035)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.26)
    click = filt(noise(0.5), "highpass", 2500) * np.exp(-t / 0.004) * 0.25
    return fade(np.tanh(1.6 * (body + click)))


def clap():
    t = T(0.45)
    env = sum(np.where(t >= o, np.exp(-(t - o) / d), 0) for o, d in ((0, 0.006), (0.011, 0.006), (0.022, 0.14)))
    return fade(filt(noise(0.45), "bandpass", [900, 3200]) * env)


def hat(d=0.04):
    t = T(d * 6)
    return fade(filt(noise(d * 6), "highpass", 7500, 4) * np.exp(-t / d))


def pluck(m, dur=0.32, bright=1.0):
    t = T(dur)
    f = hz(m)
    s = np.zeros_like(t)
    for h in range(1, 16):
        if f * h > SR / 2.3:
            break
        s += np.sin(2 * np.pi * f * h * t + rng.uniform(0, 6.28)) / h * np.exp(-t * (4 + h * h * 0.8 / bright))
    return fade(s)


def lead(m, dur):
    t = T(dur + 0.15)
    f = hz(m) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, t / 0.25))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = sum(np.sin(h * ph) / h * np.exp(-t * h * 0.6) for h in (1, 2, 3, 5, 7)) * 0.8
    env = np.minimum(1, t / 0.006) * (0.55 + 0.45 * np.exp(-t / 0.12)) * np.where(t < dur, 1, np.exp(-(t - dur) / 0.05))
    return fade(s * env)


def saw_note(m, dur):
    t = T(dur + 0.7)
    s = sum(2 * ((hz(m) * d * t + rng.random()) % 1) - 1 for d in (0.992, 1.0, 1.008)) / 3
    env = np.minimum(1, t / 0.35) * np.where(t < dur, 1, np.exp(-(t - dur) / 0.2))
    return fade(s * env)


def bass_note(m, dur):
    t = T(dur + 0.05)
    f = hz(m)
    s = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.5 * np.sin(np.pi * f * t)
    env = np.minimum(1, t / 0.008) * np.where(t < dur, np.exp(-t / 0.6), np.exp(-dur / 0.6) * np.exp(-(t - dur) / 0.03))
    return fade(np.tanh(1.2 * s * env))


# ---------------------------------------------------------------- sound effects
def pop(m, dur=0.14):
    t = T(dur)
    f = hz(m) * (1 + 0.7 * np.exp(-t / 0.01))
    return fade(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.045))


def tick():
    t = T(0.02)
    return fade(filt(noise(0.02), "highpass", 3000) * np.exp(-t / 0.003), 0.0005, 0.002)


def paper():
    t = T(0.1)
    return fade(filt(noise(0.1), "bandpass", [1200, 6500]) * np.exp(-t / 0.02) + 0.4 * np.sin(2 * np.pi * 180 * t) * np.exp(-t / 0.015))


def thud(f0=130, f1=45, d=0.3):
    t = T(d * 3)
    f = f1 + (f0 - f1) * np.exp(-t / 0.03)
    return fade(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / d))


def crash(d=1.4):
    t = T(d * 2.5)
    return fade(filt(noise(d * 2.5), "highpass", 4000) * np.exp(-t / d))


def bell(m, d=1.6):
    t = T(d * 2)
    s = sum(a * np.sin(2 * np.pi * hz(m) * r * t) * np.exp(-t / (d / r ** 0.7)) for r, a in ((1, 1), (2.0, 0.45), (3.01, 0.25), (4.2, 0.15), (5.43, 0.08)))
    return fade(s * np.minimum(1, t / 0.002))


def whoosh(dur, f0, f1, f2, peak=0.5, width=0.45):
    """Band of noise whose centre glides f0 -> f1 (at `peak`) -> f2, loudest at `peak` (0..1 of dur)."""
    nper, hop = 1024, 256
    frames = int(dur * SR / hop) + 1
    freqs = np.fft.rfftfreq(nper, 1 / SR)[:, None]
    u = np.linspace(0, 1, frames)
    fc = np.where(u < peak, f0 * (f1 / f0) ** (u / peak), f1 * (f2 / f1) ** ((u - peak) / (1 - peak)))
    env = np.where(u < peak, (u / peak) ** 2, ((1 - u) / (1 - peak)) ** 1.5)
    mag = np.exp(-0.5 * (np.log(freqs + 1) - np.log(fc[None, :])) ** 2 / width ** 2) * env[None, :]
    _, x = istft(mag * np.exp(1j * rng.uniform(0, 2 * np.pi, mag.shape)), fs=SR, nperseg=nper, noverlap=nper - hop)
    return fade(x / (np.abs(x).max() + 1e-9))


def riser(dur):
    return crash(dur * 0.6)[: int(dur * SR)][::-1] * np.linspace(0.2, 1, int(dur * SR)) ** 2


def brush_bed(t0, t1, rate, bus, gain):
    """Soft bristle rustle while strokes are being painted: one short filtered-noise swish per stroke."""
    for t in np.sort(rng.uniform(t0, t1, int((t1 - t0) * rate))):
        d = rng.uniform(0.08, 0.18)
        s = filt(noise(d), "bandpass", [700, 3500]) * np.sin(np.pi * np.linspace(0, 1, int(d * SR)))
        bus.add(s, t, gain * rng.uniform(0.5, 1), rng.uniform(-0.6, 0.6))


# ---------------------------------------------------------------- harmony and form
CH = {"F": [53, 57, 60, 64, 67], "C": [55, 60, 62, 64, 71], "G": [55, 59, 62, 64, 69], "Am": [57, 60, 64, 67, 71]}
ROOT = {"F": 41, "C": 48, "G": 43, "Am": 45}
PROG = ["F", "C", "G", "Am"]
MEL = {"F": [(0, .5, 72), (.5, .5, 76), (1, 1, 77), (2.5, .5, 76), (3, 1, 72)],
       "C": [(0, .5, 74), (.5, .5, 76), (1, 1, 79), (2.5, .5, 76), (3, 1, 74)],
       "G": [(0, .5, 71), (.5, .5, 74), (1, 1.5, 79), (3, .5, 77), (3.5, .5, 76)],
       "Am": [(0, 1.5, 76), (1.5, .5, 72), (2, 1, 74), (3, 1, 72)]}


def chord(b):
    t = BAR0 + 2 * b
    if abs(t - 101.5) < 1e-6:
        return "F"
    if t >= 103.5 - 1e-6:
        return "C"
    return PROG[(b - 3) % 4]  # the drop at 7.5 s lands on F


def section(t):
    for end, name in ((7.5, "intro"), (31.5, "A"), (69.5, "B"), (77.5, "build"), (86.5, "peak"), (95.5, "break"), (97.5, "rebuild"), (103.5, "final")):
        if t < end - 1e-9:
            return name
    return "end"


GROOVE = {"A", "B", "build", "peak", "final"}
drums, bass, pad, arp, ld, sfx = Bus(), Bus(), Bus(), Bus(), Bus(), Bus()
duck = np.ones(N)
K, CL = kick(), clap()

for k in range(-3, 210):
    t = BAR0 + k * BEAT
    s, bb = section(t), k % 4
    if s in GROOVE or (s == "rebuild"):
        drums.add(K, t, 1.0 if s != "rebuild" else 0.7)
        i = int(t * SR)
        dk = 1 - 0.55 * np.exp(-T(0.4) / 0.11)
        duck[i:i + len(dk)] = np.minimum(duck[i:i + len(dk)], dk[: max(0, min(len(dk), N - i))])
    if s in GROOVE - {"A"} and bb in (1, 3):
        drums.add(CL, t, 0.42, 0.05)
    if s in GROOVE | {"rebuild"}:
        drums.add(hat(), t + 0.25, 0.22, 0.25)
    if (s in ("build", "peak", "final") or (s == "B" and t >= 51.5)):
        for o in (0.125, 0.375):
            drums.add(hat(0.025), t + o, 0.07 * rng.uniform(0.7, 1.1), -0.3)
    if s in ("peak", "final") and bb == 3:
        drums.add(hat(0.22), t + 0.25, 0.12, 0.3)
    if s in GROOVE:
        bass.add(bass_note(ROOT[chord(int(np.floor((t - BAR0) / 2)))], 0.2), t + 0.25, 0.55)

for t0 in (75.5, 95.5):  # snare rolls into the drops
    for j, t in enumerate(np.concatenate([np.arange(t0, t0 + 1, 0.25), np.arange(t0 + 1, t0 + 2, 0.125)])):
        drums.add(CL, t, 0.12 + 0.3 * j / 12, 0.05)

for b in range(-1, 53):
    t, c = BAR0 + 2 * b, chord(b)
    last = t >= 103.5 - 1e-6
    dur = 2.6 if last else 2.0
    for m in CH[c]:
        pad.add(np.stack([saw_note(m, dur), saw_note(m, dur)]), t, 0.11)
    if section(t) == "break":
        bass.add(bass_note(ROOT[c], 1.9), t, 0.35)
    if not last:
        tones = [m + 12 for m in CH[c]]
        for j in range(16):
            ts = t + j * 0.125
            if ts >= 103.5:
                break
            arp.add(pluck(tones[(0, 2, 4, 1, 3, 4, 2, 3)[j % 8]], 0.3), ts, 0.16 * (1.0 if j % 4 == 0 else 0.75), 0.35 if j % 2 else -0.35)
    if (51.5 <= t < 69.5) or (77.5 <= t < 86.5) or (97.5 <= t < 103.5):
        for o, d, m in MEL[c]:
            if t + o * BEAT < 86.5:
                ld.add(lead(m, d * BEAT * 0.9), t + o * BEAT, 0.2)
bass.add(bass_note(36, 2.4), 103.5, 0.55)

# brightness automation: dark intro opening up, darker breakdown, open for the ending
w = np.interp(np.arange(N) / SR, [0, 7.5, 86.5, 87.5, 95.5, 97.5, 106], [0.0, 1, 1, 0.25, 0.25, 1, 1])
pad_mix = filt(pad.x, "lowpass", 500) * (1 - w) + filt(pad.x, "lowpass", 2200) * w
arp_mix = filt(arp.x, "lowpass", 1200) * (1 - w) + arp.x * w
bass_mix = filt(bass.x, "lowpass", 700)

# ---------------------------------------------------------------- sound effects, timed to index.html
for b, pan in ((7.5, 0), (24.5, 0), (41.5, 0), (60.5, 0), (77.5, 0), (97.5, 0)):  # brush wipes
    wsh = whoosh(1.15, 250, 2600, 400, peak=0.45)
    sfx.add(wsh, b - 0.52, 0.55, np.linspace(-0.8, 0.8, len(wsh)))
for b in (13, 31.5, 51.5, 69.5, 86.5, 92.5):  # plain cuts
    sfx.add(whoosh(0.6, 500, 1800, 700, peak=0.4), b - 0.25, 0.22)
for t in (7.5, 77.5, 97.5):  # drops
    drums.add(riser(2.0), t - 2.0, 0.35)
    drums.add(crash(1.6), t, 0.3)
    drums.add(thud(110, 38, 0.6), t, 0.6)
sfx.add(bell(84, 2.0), 5.4, 0.12)  # "What if a model painted?"
sfx.add(bell(91, 2.0), 5.55, 0.07, 0.3)
for _ in range(36):  # pixels appearing
    sfx.add(pop(rng.choice([84, 86, 88, 91, 93, 96]), 0.06), rng.uniform(0.7, 3.9), 0.035, rng.uniform(-0.7, 0.7))
brush_bed(1.6, 4.6, 26, sfx, 0.05)
sfx.add(whoosh(0.8, 400, 3000, 1500, peak=0.6), 8.5, 0.15, 0.3)  # title underline
for i in range(4):
    sfx.add(pop([72, 76, 79, 84][i]), 9.9 + i * 0.18, 0.2)
for t0, t1 in ((13.8, 15.0), (15.3, 16.6), (16.9, 18.6), (18.9, 21.4)):  # strokes level by level
    brush_bed(t0, t1, 22, sfx, 0.05)
for i in range(6):
    sfx.add(tick(), 21.6 + i * 0.12, 0.25, -0.4 + i * 0.16)
for i in range(10):  # paper cards
    sfx.add(paper(), 25.0 + i * 0.22, 0.28, -0.6 + (i % 5) * 0.3)
for i in range(3):
    sfx.add(whoosh(0.35, 600, 2000, 900, peak=0.5), 31.9 + i * 0.25, 0.12, -0.5)
sfx.add(thud(160, 55, 0.25), 34.2, 0.35)  # "scribbles." stamp
sfx.add(paper(), 34.2, 0.25)
for k in range(4):  # slot grids
    sfx.add(pop([72, 76, 79, 84][k]), 36.6 + k * 0.85, 0.22, 0.4)
sfx.add(bell(84, 1.2), 40.0, 0.09, 0.4)
for i in range(3):
    sfx.add(whoosh(0.35, 600, 2000, 900, peak=0.5), 42.2 + i * 0.25, 0.1, -0.5)
    sfx.add(whoosh(0.35, 600, 2000, 900, peak=0.5), 43.4 + i * 0.25, 0.1, 0.5)
sfx.add(pop(60, 0.3), 47.2, 0.15, -0.5)  # arm A greys out
sfx.add(thud(140, 50, 0.4), 47.4, 0.5)  # "B wins"
sfx.add(bell(79, 1.6), 47.45, 0.08)
sfx.add(bell(84, 1.6), 47.6, 0.06)
for i, m in enumerate([91, 88, 86, 84, 81, 79, 76, 74]):  # lever bars, pitch falls with the gain
    sfx.add(pop(m, 0.18), 52.1 + i * 0.32, 0.16, 0.3)
for t in np.arange(61.0, 64.6, 1 / 17):  # caption typing
    sfx.add(tick(), t + rng.uniform(-0.01, 0.01), 0.16 * rng.uniform(0.6, 1), rng.uniform(-0.2, 0.2))
for i in range(4):
    sfx.add(whoosh(0.5, 500, 2200, 800, peak=0.5), 63.0 + i * 0.5, 0.1, 0.4)
    sfx.add(pop([79, 84, 88, 91][i]), 65.4 + i * 0.55, 0.15)
for i in range(4):  # training stats
    sfx.add(pop([72, 76, 79, 84][i]), 69.8 + i * 0.15, 0.18)
for t in (73.0, 74.8):  # bird evolves
    sfx.add(whoosh(0.6, 400, 2500, 1200, peak=0.5), t - 0.2, 0.18)
brush_bed(77.9, 83.4, 30, sfx, 0.04)  # wall of paintings
for i in range(3):
    sfx.add(pop([84, 88, 91][i]), 80.6 + i * 0.9, 0.2, 0.5)
sfx.add(whoosh(1.1, 120, 900, 600, peak=0.85), 87.1, 0.3, -0.3)  # 150M bar
sfx.add(pop(98, 0.05), 88.4, 0.18, -0.3)  # the tiny 232k bar
for i in range(5):  # next steps
    sfx.add(pop([72, 74, 76, 79, 84][i]), 93.0 + i * 0.45, 0.2, -0.3)
brush_bed(97.8, 102.6, 24, sfx, 0.04)  # end painting
for i, m in enumerate([72, 76, 79, 84]):  # thanks for watching
    sfx.add(bell(m, 2.2), 100.2 + i * 0.12, 0.09, -0.3 + i * 0.2)
sfx.add(crash(2.0), 103.5, 0.2)
sfx.add(thud(90, 36, 0.9), 103.5, 0.3)

# ---------------------------------------------------------------- mix
def pingpong(x, d=0.375, fb=0.38, taps=6):
    y, m = np.zeros_like(x), x.mean(axis=0)
    for k in range(1, taps + 1):
        s = int(d * k * SR)
        y[k % 2, s:] += fb ** k * m[: N - s]
    return filt(y, "lowpass", 3500)


def reverb(x, decay=0.5, length=2.4):
    t = T(length)
    ir = np.stack([filt(noise(length), "lowpass", 6000) * np.exp(-t / decay) for _ in range(2)])
    ir[:, : int(0.015 * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    return np.stack([fftconvolve(x[c], ir[c])[:N] for c in range(2)])


pad_mix, arp_mix, ld_x, drums_x = 2.0 * pad_mix, 1.6 * arp_mix, 2.2 * ld.x, 0.45 * drums.x
music = drums_x + duck * (pad_mix + 0.8 * bass_mix + arp_mix) + ld_x
music += 0.35 * pingpong(arp_mix + 1.5 * ld_x) + 0.18 * reverb(pad_mix + arp_mix + ld_x + 0.5 * drums_x)
fx = 1.4 * (sfx.x + 0.25 * reverb(sfx.x, 0.35, 1.5))
mix = music + fx
env = np.interp(np.arange(N) / SR, [0, 0.25, 104.6, 106], [0, 1, 1, 0])
mix = mix * env / np.quantile(np.abs(mix), 0.9995) * 0.8  # rare peaks above 0.8 go into the soft limiter below
mix = np.where(np.abs(mix) < 0.8, mix, np.sign(mix) * (0.8 + 0.18 * np.tanh((np.abs(mix) - 0.8) / 0.18)))

for name, x in (("drums", drums_x), ("bass", bass_mix), ("pad", pad_mix), ("arp", arp_mix), ("lead", ld_x), ("sfx", sfx.x), ("mix", mix)):
    print(f"{name:6s} rms {20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12):6.1f} dB  peak {np.abs(x).max():.2f}")
wavfile.write("music.wav", SR, (mix.T * 32767).astype(np.int16))
print("wrote music.wav")
