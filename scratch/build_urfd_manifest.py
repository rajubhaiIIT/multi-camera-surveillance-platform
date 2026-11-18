import json
import re
import socket
import urllib.request
from pathlib import Path

# Force reliable IPv4 mapping for academic domain
_orig_getaddrinfo = socket.getaddrinfo
def _v4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if host == "fenix.ur.edu.pl":
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("62.93.43.105", port))]
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _v4_getaddrinfo

url = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="ignore")

# Find all mp4 files
mp4_links = re.findall(r'href=[\'"]([^\'"]+\.mp4)[\'"]', html)
print(f"Total MP4 links: {len(mp4_links)}")

clips = {"falls": {}, "adls": {}}

for link in mp4_links:
    clean = link.replace("./", "")
    filename = Path(clean).name  # e.g. fall-01-cam0.mp4 or adl-01-cam0.mp4
    full_url = f"https://fenix.ur.edu.pl/~mkepski/ds/{clean}"
    if filename.startswith("fall-"):
        seq_id = filename.split("-")[1]
        if seq_id not in clips["falls"] or "cam0" in filename:
            clips["falls"][seq_id] = {"filename": filename, "url": full_url}
    elif filename.startswith("adl-"):
        seq_id = filename.split("-")[1]
        if seq_id not in clips["adls"] or "cam0" in filename:
            clips["adls"][seq_id] = {"filename": filename, "url": full_url}

print(f"Mapped {len(clips['falls'])} Fall sequences and {len(clips['adls'])} ADL sequences.")
out_file = Path("data/raw/urfall/manifest.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(json.dumps(clips, indent=2))
print(f"Saved manifest -> {out_file}")
