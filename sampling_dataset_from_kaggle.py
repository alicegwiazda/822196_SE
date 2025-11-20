#works with WikiArt dataset from Kaggle, from which we sample 4 images each from 13 atrists to create an MVP

import os, shutil, unicodedata, random, pandas as pd, glob

src_root = "/kaggle/input/wikiart"
sample_dir = "/kaggle/working/sample"
N = 4
SEED = 42
random.seed(SEED)

target_canonical_artists = [
    "picasso", "rembrandt", "goya", "monet", "manet", "frida-kahlo",
    "rubens", "kandinsky", "gauguin", "cezanne", "modigliani",
    "caravaggio", "da-vinci"
]

for p in [sample_dir,
          "/kaggle/working/wikiart_sample_with_metadata.zip",
          "/kaggle/working/wikiart_sample.zip"]:
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    elif os.path.isfile(p):
        try: os.remove(p)
        except: pass
os.makedirs(sample_dir, exist_ok=True)

def normalize(s: str) -> str:
    s = s.strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = s.replace(' ', '-').replace('_', '-')
    while '--' in s:
        s = s.replace('--', '-')
    return s

artist_aliases = {
    "picasso": ["pablo-picasso", "picasso"],
    "rembrandt": ["rembrandt", "rembrandt-van-rijn"],
    "goya": ["francisco-goya", "goya", "francisco-de-goya"],
    "monet": ["claude-monet", "monet"],
    "manet": ["edouard-manet", "manet", "édouard-manet"],
    "frida-kahlo": ["frida-kahlo", "kahlo"],
    "rubens": ["peter-paul-rubens", "rubens"],
    "kandinsky": ["wassily-kandinsky", "kandinsky", "vasily-kandinsky"],
    "gauguin": ["paul-gauguin", "gauguin"],
    "cezanne": ["paul-cezanne", "cezanne", "cézanne"],
    "modigliani": ["amedeo-modigliani", "modigliani", "amedeo-modígliani"],
    "caravaggio": ["caravaggio", "michelangelo-merisi-da-caravaggio"],
    "da-vinci": ["leonardo-da-vinci", "leonardo", "leonardo-di-ser-piero-da-vinci"],
}

artist_aliases = {k:v for k,v in artist_aliases.items()
                  if normalize(k) in {normalize(x) for x in target_canonical_artists}}

normalized_aliases = { normalize(k): { normalize(x) for x in v } for k, v in artist_aliases.items() }
alias_to_canonical = {}
for canon, aliases in normalized_aliases.items():
    for a in aliases:
        alias_to_canonical[a] = canon

alias_prefixes = { a: f"{a}_" for a in alias_to_canonical.keys() }

valid_ext = {".jpg", ".jpeg"}

found = { canon: [] for canon in normalized_aliases.keys() }
remaining = { canon: N for canon in normalized_aliases.keys() }

def all_done():
    return all(remaining[c] <= 0 for c in remaining)

for root, _, files in os.walk(src_root):
    if all_done():
        break
    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_ext:
            continue
        base = os.path.splitext(fname)[0].lower()

        matched_canon = None
        for alias, prefix in alias_prefixes.items():
            if base.startswith(prefix):
                matched_canon = alias_to_canonical[alias]
                break
        if not matched_canon:
            continue
        if remaining[matched_canon] <= 0:
            continue

        found[matched_canon].append(os.path.join(root, fname))
        remaining[matched_canon] -= 1

total = 0
for canon, files in found.items():
    if not files:
        print(f"[WARN] No files found for {canon}")
        continue
    outdir = os.path.join(sample_dir, canon)
    os.makedirs(outdir, exist_ok=True)
    for src in files:
        shutil.copy(src, outdir)
    print(f"[OK] {canon}: copied {len(files)} file(s)")
    total += len(files)
print(f"Copied total: {total}")

classes_csv_candidates = glob.glob(os.path.join(src_root, "**", "classes.csv"), recursive=True)
classes_csv_path = classes_csv_candidates[0]

def bn_noext(p): return os.path.splitext(os.path.basename(p))[0].lower()
sampled_paths = glob.glob(os.path.join(sample_dir, "**", "*"), recursive=True)
sampled_imgs = [p for p in sampled_paths if os.path.splitext(p)[1].lower() in valid_ext]
sampled_basenames = { bn_noext(p) for p in sampled_imgs }

df = pd.read_csv(classes_csv_path)

candidate_cols = [c for c in df.columns if any(k in c.lower() for k in
                   ["file", "image", "img", "path", "name", "url"])]
if not candidate_cols:
    candidate_cols = list(df.columns)

def norm_cell_to_basename(cell):
    s = str(cell).replace("\\", "/")
    s = os.path.basename(s)
    s = os.path.splitext(s)[0]
    return s.strip().lower()

best_col, best_hits = None, 0
for col in candidate_cols:
    try:
        hits = df[col].map(norm_cell_to_basename).isin(sampled_basenames).sum()
    except Exception:
        continue
    if hits > best_hits:
        best_col, best_hits = col, hits

if not best_col or best_hits == 0:
    raise RuntimeError("Couldn't match any rows from classes.csv to sampled images.")

df["_basename"] = df[best_col].map(norm_cell_to_basename)
subset = df[df["_basename"].isin(sampled_basenames)].copy()

bn2path = { bn_noext(p): p for p in sampled_imgs }
subset["sample_path"] = subset["_basename"].map(bn2path)

out_csv = os.path.join(sample_dir, "classes.csv")
subset.to_csv(out_csv, index=False)
print(f"[OK] Saved filtered metadata: {out_csv} ({len(subset)} rows)")

zip_base = "/kaggle/working/wikiart_sample_with_metadata"
shutil.make_archive(zip_base, "zip", sample_dir)
