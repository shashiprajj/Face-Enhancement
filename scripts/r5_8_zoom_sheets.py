"""R5-8 — stratified, zoomed DEV labelling sheets (2x input | output). Strata over l_hfc_growth_max of outputs that
PASS the global guard (the population the local guard must police); fixed before inspection:
  fc: all with g >= 8; random 12 (seed 0) from [4, 8), [2, 4), [0, 2)
  fi: all with g >= 30; random 12 (seed 0) from [15, 30), [10, 15), [6, 10), [3, 6), [0, 3)
→ experiments/phase5_rapid/outputs/r5_8/zoom_<mode>_<k>.png + data/manifests/phase5_rapid/r5_8/zoom_index.json
"""
import json, random, cv2, numpy as np, gzip, csv
R = list(csv.DictReader(gzip.open('data/manifests/phase5_rapid/r5_8/local_features.csv.gz', 'rt')))
STRATA = {"fc": [(8, 1e9, None), (4, 8, 12), (2, 4, 12), (0, 2, 12)],
          "fi": [(30, 1e9, None), (15, 30, 12), (10, 15, 12), (6, 10, 12), (3, 6, 12), (0, 3, 12)]}
index = {}
for mode, strata in STRATA.items():
    P = [r for r in R if r["mode"] == mode and r["guard_pass"] == "True"]
    items = []
    for lo, hi, k in strata:
        band = sorted((r for r in P if lo <= float(r["l_hfc_growth_max"]) < hi), key=lambda r: -float(r["l_hfc_growth_max"]))
        pick = band if k is None else sorted(random.Random(0).sample(band, min(k, len(band))),
                                              key=lambda r: -float(r["l_hfc_growth_max"]))
        items += [(r["name"], f"[{lo},{hi})", float(r["l_hfc_growth_max"])) for r in pick]
    index[mode] = [{"i": i, "name": n, "stratum": s, "g": round(g, 2)} for i, (n, s, g) in enumerate(items)]
    for s in range(0, len(items), 12):
        tiles = []
        for i, (n, st, g) in enumerate(items[s:s + 12], start=s):
            out = cv2.imread(f'experiments/phase5_rapid/outputs/r5_8/corpus/{mode}/{n}.png')
            inp = cv2.imread(f'Dataset-and-code/archive/input/val/{n}' if mode == 'fc' else f'experiments/phase5_rapid/outputs/r5_8/corpus/fi_chip/{n}.png')
            t = np.hstack([cv2.resize(inp, (256, 256), interpolation=cv2.INTER_NEAREST), cv2.resize(out, (256, 256), interpolation=cv2.INTER_NEAREST)])
            cv2.putText(t, f"{mode}#{i} g{g:.0f} {st}", (3, 14), 0, 0.45, (0, 0, 255), 1)
            tiles.append(t)
        while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(f'experiments/phase5_rapid/outputs/r5_8/zoom_{mode}_{s // 12:02d}.png', np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
json.dump(index, open('data/manifests/phase5_rapid/r5_8/zoom_index.json', 'w'), indent=0)
print({m: len(v) for m, v in index.items()})
