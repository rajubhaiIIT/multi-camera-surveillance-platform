import urllib.request
import re

req = urllib.request.Request('https://fenix.ur.edu.pl/~mkepski/ds/uf.html', headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8', errors='ignore')

links = re.findall(r'href=[\'"]([^\'"]+)[\'"]', html)
print(f"Total links on uf.html: {len(links)}")
for l in links:
    if any(k in l.lower() for k in ['zip', 'csv', 'mp4', 'avi', 'fall', 'adl']):
        print("Dataset file:", l)
