<div align="center">

# Optimalizační algoritmy
</div>

Přehled výsledků vyhledávání minim jde najít v souboru `statistics_output.xlsx`,
kde je přehled iterací pro jednotlivé algoritmy, funkce a dimenzionalitu,
nalezená minima a vstupy, které k tomuto minimu vedly. Dále je pak pro
každý algoritmus, funkci a dimenzionalitu uveden průměr, medián, maximum,
minimum a standardní odchylka.

Vykreslené grafy dle zadání jde najít v adresáři `./plots`, jejich pojmenování
vždy udává testovací funkci, dimenze, typ grafu a typ algoritmu. Grafy byly
vykresleny pomocí knihovny `matplotlib`.

Implementace algoritmů je v souboru `optim.py`. Pokud máte nainstalovaný
Python 3.x, jde skript spustit následovně:

```
py -m pip install -r requirements.txt
py optim.py
```

Výstup skiptu se uloží do souboru `raw_output.xlsx`, a zároveň skript vykreslí
grafy pro jednotlivé vstupy.

### Hill climber

Pro funkci Schwefel váznul hill climber opakovaně v lokálních minimech,
rozhodl jsem se ho tedy spustit opakovaně v jedné iteraci, celkem 5x
(po 2000 voláních účelové funkce) a brát nejlepší výsledek. Toto opakované
spuštění je dobře patrné i na grafech, kde dochází ke zlomům a skoku na 
lepší výsledek.

Taktéž jsem zkoušel alternativní metodu určování okolí, v grafech se jedná
o grafy se sufixem `_alt`, a to použití 10% okolí od aktuálního bodu, ne 
10% z celého rozsahu. Tím se postupným snižováním hodnoty bodu při hledání
minima snižovala i velikost okolí. Především pro funkci Schwefel to vedlo
k celkem výraznému zlepšení dosahovaných výsledků.
