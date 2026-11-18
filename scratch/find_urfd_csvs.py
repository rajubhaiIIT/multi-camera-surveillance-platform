import urllib.request
import re

req = urllib.request.Request('https://fenix.ur.edu.pl/~mkepski/ds/uf.html', headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8', errors='ignore')

links = re.findall(r'href=[\'"]([^\'"]+)[\'"]', html)
csvs = [l for l in links if 'csv' in l.lower()]
print(f"CSVs found ({len(csvs)}):")
for c in csvs[:15]:
    print(" ", c)
if any('urfall' in c.lower() for c in csvs):
    print("Found urfall csv!")
