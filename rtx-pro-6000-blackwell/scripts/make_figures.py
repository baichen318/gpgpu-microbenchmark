#!/usr/bin/env python3
"""Generate dependency-free SVG figures and summary tables from raw CSV data."""
import csv
import math
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
FIG = os.path.join(ROOT, "report", "figures")
os.makedirs(FIG, exist_ok=True)

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
          "#17becf", "#e377c2"]


def read_csv(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def fmt_bytes(v):
    v = float(v)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(v) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{v:.0f} B"
            return f"{v:.0f} {unit}" if v >= 10 else f"{v:.1f} {unit}"
        v /= 1024.0
    return f"{v:.1f} TiB"


def fmt_num(v, unit=""):
    if abs(v) >= 1e12:
        return f"{v/1e12:.2f} T{unit}"
    if abs(v) >= 1e9:
        return f"{v/1e9:.2f} G{unit}"
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f} M{unit}"
    if abs(v) >= 1e3:
        return f"{v/1e3:.2f} k{unit}"
    return f"{v:.2f} {unit}".strip()


def xml_escape(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def nice_ticks(lo, hi, count=6):
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        return [lo]
    span = hi - lo
    raw = span / max(1, count - 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in [1, 2, 2.5, 5, 10]:
        step = m * mag
        if span / step <= count + 0.5:
            break
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            ticks.append(v)
        v += step
    return ticks


def chart(path, title, xlabel, ylabel, series, logx=False, logy=False,
          width=960, height=520, xfmt=None, yfmt=None, ymin=None, ymax=None,
          legend_pos="upper right", xticks=None):
    ml, mr, mt, mb = 100, 40, 58, 78
    pw, ph = width - ml - mr, height - mt - mb
    xs, ys = [], []
    for s in series:
        for p in s["points"]:
            x = p[0]
            if x > 0 or not logx:
                xs.append(math.log10(x) if logx else x)
                if s.get("band"):
                    ys.extend([p[1], p[2]])
                else:
                    ys.append(p[1])
    if not xs or not ys:
        return
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = (min(ys), max(ys)) if ymin is None or ymax is None else (ymin, ymax)
    if ymin is not None:
        ylo = ymin
    if ymax is not None:
        yhi = ymax
    if xlo == xhi:
        xlo -= 1
        xhi += 1
    if ylo == yhi:
        ylo -= 1
        yhi += 1
    pad = 0.04 * (yhi - ylo)
    ylo -= pad
    yhi += pad
    if ymin is not None:
        ylo = ymin
    if ymax is not None:
        yhi = ymax

    def sx(v):
        t = math.log10(v) if logx else v
        return ml + (t - xlo) / (xhi - xlo) * pw

    def sy(v):
        return mt + (1.0 - (v - ylo) / (yhi - ylo)) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
           f'height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{width/2}" y="32" text-anchor="middle" '
           f'font-family="Helvetica,Arial,sans-serif" font-size="20" '
           f'font-weight="bold" fill="#222">{xml_escape(title)}</text>']
    # grid and y ticks
    yticks = nice_ticks(ylo, yhi, 6)
    for t in yticks:
        if t < ylo - 1e-12 or t > yhi + 1e-12:
            continue
        y = sy(t)
        out.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{ml+pw}" y2="{y:.2f}" '
                   f'stroke="#e6e6e6" stroke-width="1"/>')
        label = yfmt(t) if yfmt else f"{t:.3g}"
        out.append(f'<text x="{ml-12}" y="{y+5:.2f}" text-anchor="end" '
                   f'font-family="Helvetica,Arial,sans-serif" font-size="13" '
                   f'fill="#333">{xml_escape(label)}</text>')
    # x ticks
    if xticks is None:
        tick_positions = nice_ticks(xlo, xhi, 7)
    else:
        tick_positions = [math.log10(v) if logx else v for v in xticks]
        tick_positions = [t for t in tick_positions
                          if xlo - 1e-9 <= t <= xhi + 1e-9]
    for t in tick_positions:
        if t < xlo - 1e-12 or t > xhi + 1e-12:
            continue
        x = ml + (t - xlo) / (xhi - xlo) * pw
        out.append(f'<line x1="{x:.2f}" y1="{mt}" x2="{x:.2f}" y2="{mt+ph}" '
                   f'stroke="#ededed" stroke-width="1"/>')
        value = 10 ** t if logx else t
        label = xfmt(value) if xfmt else f"{value:.3g}"
        out.append(f'<text x="{x:.2f}" y="{mt+ph+24}" text-anchor="middle" '
                   f'font-family="Helvetica,Arial,sans-serif" font-size="13" '
                   f'fill="#333">{xml_escape(label)}</text>')
    # axes
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" '
               f'stroke="#333" stroke-width="1.5"/>')
    out.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" '
               f'stroke="#333" stroke-width="1.5"/>')
    out.append(f'<text x="{ml+pw/2}" y="{height-22}" text-anchor="middle" '
               f'font-family="Helvetica,Arial,sans-serif" font-size="15" '
               f'fill="#222">{xml_escape(xlabel)}</text>')
    out.append(f'<text x="24" y="{mt+ph/2}" text-anchor="middle" '
               f'font-family="Helvetica,Arial,sans-serif" font-size="15" '
               f'fill="#222" transform="rotate(-90 24 {mt+ph/2})">'
               f'{xml_escape(ylabel)}</text>')
    # series
    for i, s in enumerate(series):
        color = s.get("color", COLORS[i % len(COLORS)])
        pts = sorted([p for p in s["points"] if p[0] > 0 or not logx])
        if not pts:
            continue
        if s.get("band"):
            lo_pts = [(x, sy(lo)) for x, lo, hi in pts]
            hi_pts = [(x, sy(hi)) for x, lo, hi in pts]
            poly = " ".join([f"{sx(x):.2f},{y:.2f}" for x, y in lo_pts] +
                            [f"{sx(x):.2f},{y:.2f}" for x, y in reversed(hi_pts)])
            out.append(f'<polygon points="{poly}" fill="{color}" fill-opacity="0.16" '
                       f'stroke="none"/>')
            continue
        else:
            line_pts = pts
        d = " ".join([f"{'M' if j == 0 else 'L'}{sx(x):.2f},{sy(y):.2f}"
                      for j, (x, y) in enumerate(line_pts)])
        out.append(f'<path d="{d}" fill="none" stroke="{color}" '
                   f'stroke-width="2.4" stroke-linejoin="round"/>')
        for x, y in line_pts:
            out.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.5" '
                       f'fill="{color}"/>')
    # legend
    lx = ml + pw - 230 if legend_pos == "upper right" else ml + 20
    ly = mt + 16
    legend_i = 0
    for i, s in enumerate(series):
        if s.get("band"):
            continue
        color = s.get("color", COLORS[i % len(COLORS)])
        out.append(f'<line x1="{lx}" y1="{ly+legend_i*22}" x2="{lx+28}" '
                   f'y2="{ly+legend_i*22}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text x="{lx+36}" y="{ly+legend_i*22+5}" '
                   f'font-family="Helvetica,Arial,sans-serif" font-size="13" '
                   f'fill="#222">{xml_escape(s["name"])}</text>')
        legend_i += 1
    out.append("</svg>")
    with open(path, "w") as fh:
        fh.write("\n".join(out))


def bar_chart(path, title, ylabel, categories, values, colors=None,
              width=960, height=520, yfmt=None, logy=False):
    ml, mr, mt, mb = 100, 40, 58, 96
    pw, ph = width - ml - mr, height - mt - mb
    if logy:
        vals = [math.log10(max(v, 1e-9)) for v in values]
        ylo, yhi = 0, max(vals) * 1.08
    else:
        vals = values
        ylo, yhi = 0, max(vals) * 1.12
    if yhi == ylo:
        yhi = ylo + 1

    def sy(v):
        return mt + (1.0 - (v - ylo) / (yhi - ylo)) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
           f'height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{width/2}" y="32" text-anchor="middle" '
           f'font-family="Helvetica,Arial,sans-serif" font-size="20" '
           f'font-weight="bold" fill="#222">{xml_escape(title)}</text>']
    yticks = nice_ticks(ylo, yhi, 6)
    for t in yticks:
        if t < ylo - 1e-12 or t > yhi + 1e-12:
            continue
        y = sy(t)
        out.append(f'<line x1="{ml}" y1="{y:.2f}" x2="{ml+pw}" y2="{y:.2f}" '
                   f'stroke="#e6e6e6" stroke-width="1"/>')
        label = (f"{10**t:.3g}" if logy else (yfmt(t) if yfmt else f"{t:.3g}"))
        out.append(f'<text x="{ml-12}" y="{y+5:.2f}" text-anchor="end" '
                   f'font-family="Helvetica,Arial,sans-serif" font-size="13" '
                   f'fill="#333">{xml_escape(label)}</text>')
    n = len(categories)
    slot = pw / max(1, n)
    bw = slot * 0.62
    for i, (cat, val) in enumerate(zip(categories, values)):
        x = ml + slot * (i + 0.5) - bw / 2
        y = sy(vals[i])
        color = (colors[i] if colors and i < len(colors)
                 else COLORS[i % len(COLORS)])
        out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bw:.2f}" '
                   f'height="{mt+ph-y:.2f}" fill="{color}" fill-opacity="0.88"/>')
        label = yfmt(val) if yfmt else f"{val:.3g}"
        out.append(f'<text x="{x+bw/2:.2f}" y="{y-8:.2f}" text-anchor="middle" '
                   f'font-family="Helvetica,Arial,sans-serif" font-size="13" '
                   f'fill="#222">{xml_escape(label)}</text>')
        out.append(f'<text x="{x+bw/2:.2f}" y="{mt+ph+26}" text-anchor="middle" '
                   f'font-family="Helvetica,Arial,sans-serif" font-size="12" '
                   f'fill="#333">{xml_escape(cat)}</text>')
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" '
               f'stroke="#333" stroke-width="1.5"/>')
    out.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" '
               f'stroke="#333" stroke-width="1.5"/>')
    out.append(f'<text x="24" y="{mt+ph/2}" text-anchor="middle" '
               f'font-family="Helvetica,Arial,sans-serif" font-size="15" '
               f'fill="#222" transform="rotate(-90 24 {mt+ph/2})">'
               f'{xml_escape(ylabel)}</text>')
    out.append("</svg>")
    with open(path, "w") as fh:
        fh.write("\n".join(out))


def main():
    # 1. Memory latency
    rows = [r for r in read_csv("memory_latency.csv")
            if r.get("stride_mode") == "half"]
    for mode, color in [("ca", COLORS[0]), ("cg", COLORS[1])]:
        pts = [(f(r, "size_bytes"), f(r, "latency_ns"))
               for r in rows if r.get("mode") == mode]
        chart(os.path.join(FIG, f"fig_memory_latency_{mode}.svg"),
              "Memory hierarchy latency" if mode == "ca"
              else "Memory latency with L1 bypass (cg)",
              "Working set", "Latency (ns)", [{"name": mode.upper(), "points": pts,
                                              "color": color}],
              logx=True, xfmt=fmt_bytes, yfmt=lambda v: f"{v:.0f}",
              xticks=[4 << 10, 64 << 10, 1 << 20, 16 << 20, 128 << 20, 1 << 30])

    # 2. HBM bandwidth
    rows = [r for r in read_csv("memory_bandwidth.csv")
            if r.get("state") == "warm" and f(r, "size_bytes") >= (256 << 20)]
    series = []
    for op, color in [("read", COLORS[0]), ("write", COLORS[1]),
                      ("copy", COLORS[2]), ("triad", COLORS[3])]:
        pts = [(f(r, "size_bytes"), f(r, "bandwidth_gbps"))
               for r in rows if r.get("operation") == op]
        series.append({"name": op, "points": pts, "color": color})
    chart(os.path.join(FIG, "fig_hbm_bandwidth.svg"),
          "HBM bandwidth vs working set (>= 256 MiB)", "Working set", "GB/s",
          series, logx=True, xfmt=fmt_bytes, yfmt=lambda v: f"{v:.0f}",
          xticks=[256 << 20, 512 << 20, 1 << 30, 2 << 30, 4 << 30, 8 << 30])

    # 3. Cache / shared bandwidth
    rows = read_csv("cache.csv")
    bw = {}
    for r in rows:
        test = r.get("test")
        if test in ("l1_read", "l2_read", "shared_read"):
            bw[test] = f(r, "bandwidth_gbps")
    if bw:
        bar_chart(os.path.join(FIG, "fig_cache_bandwidth.svg"),
                  "Cache and shared-memory read bandwidth",
                  "Aggregate GB/s", ["L1", "L2", "Shared"],
                  [bw.get("l1_read", 0), bw.get("l2_read", 0),
                   bw.get("shared_read", 0)],
                  colors=[COLORS[0], COLORS[1], COLORS[2]],
                  yfmt=lambda v: f"{v/1000:.1f}k")

    # 4. Tensor throughput
    rows = read_csv("tensor.csv")
    labels = {"fp16_wmma": "FP16", "bf16_wmma": "BF16", "fp8_mma": "FP8",
              "fp4_unscaled": "FP4 unscaled", "fp4_mxf4": "MXFP4",
              "fp4_nvfp4": "NVFP4"}
    cats, vals = [], []
    for r in rows:
        m = r.get("metric")
        if m in labels:
            cats.append(labels[m])
            vals.append(f(r, "value"))
    if cats:
        bar_chart(os.path.join(FIG, "fig_tensor.svg"),
                  "Tensor-core throughput", "TFLOPS", cats, vals,
                  colors=[COLORS[0], COLORS[1], COLORS[2], COLORS[3],
                          COLORS[4], COLORS[5]],
                  yfmt=lambda v: f"{v:.0f}")

    # 5. NoC bandwidth
    rows = read_csv("noc_bandwidth.csv")
    pts = [(f(r, "blocks"), f(r, "bandwidth_gbps")) for r in rows]
    pts_sm = [(f(r, "blocks"), f(r, "bandwidth_gbps") / max(1.0, f(r, "blocks")))
              for r in rows]
    if pts:
        chart(os.path.join(FIG, "fig_noc_bandwidth.svg"),
              "NoC / L2 aggregate bandwidth scaling", "Active SMs", "GB/s",
              [{"name": "aggregate", "points": pts, "color": COLORS[0]}],
              xfmt=lambda v: f"{v:.0f}", yfmt=lambda v: f"{v/1000:.1f}k")
        chart(os.path.join(FIG, "fig_noc_bandwidth_per_sm.svg"),
              "Per-SM NoC / L2 bandwidth", "Active SMs", "GB/s per SM",
              [{"name": "per SM", "points": pts_sm, "color": COLORS[2]}],
              xfmt=lambda v: f"{v:.0f}", yfmt=lambda v: f"{v:.0f}")

    # 6. NoC latency with min/max band
    rows = read_csv("noc_latency.csv")
    band = [(f(r, "active_blocks"), f(r, "min_roundtrip_ns"),
             f(r, "max_roundtrip_ns")) for r in rows]
    med = [(f(r, "active_blocks"), f(r, "median_roundtrip_ns")) for r in rows]
    if med:
        chart(os.path.join(FIG, "fig_noc_latency.svg"),
              "NoC ping-pong round-trip latency", "Active SMs", "ns",
              [{"name": "median (band: min-max)", "points": band,
                "color": COLORS[1], "band": True},
               {"name": "median", "points": med, "color": COLORS[0]}],
              xfmt=lambda v: f"{v:.0f}", yfmt=lambda v: f"{v:.0f}")

    # 7. NoC latency histogram
    rows = read_csv("noc_matrix.csv")
    vals = [f(r, "roundtrip_ns") for r in rows]
    if vals:
        lo, hi, nb = 900.0, 1500.0, 24
        bins = [0] * nb
        for v in vals:
            k = int((v - lo) / (hi - lo) * nb)
            k = max(0, min(nb - 1, k))
            bins[k] += 1
        cats = [f"{lo + (hi-lo)*i/nb:.0f}" for i in range(nb)]
        bar_chart(os.path.join(FIG, "fig_noc_hist.svg"),
                  "Distribution of all-to-all NoC round-trip latency",
                  "Pairs", cats, bins, yfmt=lambda v: f"{v:.0f}")

    # 8. NoC latency vs SM-ID distance (binned) + correlation
    if rows:
        pairs = []
        for r in rows:
            pairs.append((abs(int(r["smid_i"]) - int(r["smid_j"])),
                          f(r, "roundtrip_ns")))
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in pairs)
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        corr = cov / math.sqrt(vx * vy) if vx and vy else 0.0
        edges = [0, 1, 4, 8, 16, 32, 64, 128, 256]
        pts = []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = [y for x, y in pairs if a <= x < b]
            if sel:
                pts.append(((a + b - 1) / 2.0, statistics.mean(sel)))
        chart(os.path.join(FIG, "fig_noc_distance.svg"),
              f"NoC latency vs SM-ID distance (Pearson r={corr:.3f})",
              "SM-ID distance (bin centre)", "Mean round-trip (ns)",
              [{"name": "mean", "points": pts, "color": COLORS[4]}],
              xfmt=lambda v: f"{v:.0f}", yfmt=lambda v: f"{v:.0f}")
        with open(os.path.join(ROOT, "results", "raw", "noc_distance_bins.csv"),
                  "w") as fh:
            fh.write("distance_lo,distance_hi,count,mean_roundtrip_ns\n")
            for a, b in zip(edges[:-1], edges[1:]):
                sel = [y for x, y in pairs if a <= x < b]
                if sel:
                    fh.write(f"{a},{b},{len(sel)},{statistics.mean(sel):.3f}\n")
        print(f"NoC latency vs SM-ID distance: Pearson r = {corr:.4f}")

    # 9. PCIe
    rows = [r for r in read_csv("pcie.csv") if r.get("memory_type") == "pinned"]
    series = []
    for direction, color in [("H2D", COLORS[0]), ("D2H", COLORS[1])]:
        pts = [(f(r, "size_bytes"), f(r, "bandwidth_gbps"))
               for r in rows if r.get("direction") == direction]
        series.append({"name": direction, "points": pts, "color": color})
    if series:
        chart(os.path.join(FIG, "fig_pcie.svg"),
              "PCIe pinned-memory bandwidth", "Transfer size", "GB/s",
              series, logx=True, xfmt=fmt_bytes, yfmt=lambda v: f"{v:.0f}",
              xticks=[1 << 10, 1 << 16, 1 << 20, 1 << 24, 1 << 30])

    # 10. Power / temperature
    rows = read_csv("power.csv")
    if rows:
        pts = [(f(r, "elapsed_s"), f(r, "power_w")) for r in rows]
        chart(os.path.join(FIG, "fig_power.svg"),
              "Sustained FP32 power draw", "Time (s)", "Power (W)",
              [{"name": "power", "points": pts, "color": COLORS[1]}],
              xfmt=lambda v: f"{v:.0f}", yfmt=lambda v: f"{v:.0f}")
        pts = [(f(r, "elapsed_s"), f(r, "temp_c")) for r in rows]
        chart(os.path.join(FIG, "fig_temperature.svg"),
              "Sustained FP32 temperature", "Time (s)", "Temperature (C)",
              [{"name": "GPU temp", "points": pts, "color": COLORS[3]}],
              xfmt=lambda v: f"{v:.0f}", yfmt=lambda v: f"{v:.0f}")

    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
