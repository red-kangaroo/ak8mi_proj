<div align="center">

# Matematická informatika
## Optimalizační algoritmy
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

Výstup skiptu se nalézá v souboru `raw_output.xlsx`, a zároveň skript vykreslí
grafy pro jednotlivé vstupy.
