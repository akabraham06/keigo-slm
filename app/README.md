# Live demo

A clean single-page app (`web/index.html`) that calls a FastAPI backend (`server.py`) running
your tuned model. Japanese in → English out, with the **detected source register** and whether
the output **preserved or flattened** it.

```
web/index.html   ── the SPA (open in any browser; no build step)
server.py        ── FastAPI backend that runs the v2 model on a GPU
```

## 1. Start the backend on Colab (GPU) and get a public URL

Run this as **one cell** in Colab (GPU runtime). It launches the API and opens a free public
tunnel — no account needed — then prints the URL to paste into the SPA.

```python
import os, re, json, time, subprocess, urllib.request
os.chdir("/content/keigo-slm"); os.makedirs(".", exist_ok=True)
!git pull
!pip install -q fastapi uvicorn
# free, no-signup tunnel
!wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared

# start the API (loads the v2 model — takes ~30-60s)
api = subprocess.Popen(["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"],
                       env={**os.environ, "PYTHONPATH": "src"})
for _ in range(150):
    try:
        if json.load(urllib.request.urlopen("http://localhost:8000/health"))["ok"]:
            print("✅ model loaded, API up"); break
    except Exception: time.sleep(3)

# open the public tunnel and print the URL
tun = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8000"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in tun.stdout:
    m = re.search(r"https://[-\w]+\.trycloudflare\.com", line)
    if m:
        print("\n🌐 PUBLIC URL (paste into the SPA):", m.group(0)); break
```

Keep that cell running for the whole demo — it's your live server.

## 2. Open the SPA and connect it

- Open `app/web/index.html` in your browser (double-click it, or serve the folder with
  `python -m http.server` and visit `localhost:8000`).
- Paste the `trycloudflare.com` URL into the **Backend URL** box and click **Check** — the dot
  turns green when connected.
- Type Japanese (or click a preset triple) and hit **Translate**.

## What the demo shows

Each translation returns three things the UI renders:

- **source register** — detected from the Japanese deterministically (verb morphology).
- **output register** — what band the English landed in (heuristic judge).
- **status** — `✓ preserved`, `⚠ flattened` (politeness lost — the forbidden failure), or
  `↑ slightly more formal`.

The strongest live moment: click the three **coffee** presets (casual → polite → formal). Same
request, and the model shifts the English register each time — "Want some coffee?" →
"Would you like some coffee?" → "Would you care to have some coffee?" — with no prompt change.

> Model is set by `KEIGO_MODEL` (default `akabraham06/keigo-slm-qwen3-1.7b-v2`). The tunnel URL
> changes every Colab session — just re-paste the new one into the SPA.
