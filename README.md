*Read this in [English](README_en.md).*

# PINN - Physics-Informed Neural Networks

En aquest repositori es troba el codi utilitzat per al meu TFG sobre les Physics-Informed Neural Networks i les seves aplicacions.

## Tipus de Problemes

* **Problemes Directes:** Trobar una aproximació de la solució real a partir de les condicions inicials i de frontera, i un conjunt de punts que han de complir les lleis físiques (les EDO o les EDP).
* **Problemes Indirectes (Inversos):** Aquí només s'ha estudiat per a les equacions de Meinhardt, amb l'objectiu d'observar la inestabilitat de Turing, però el procés per a una altra equació o sistema d'equacions és similar.

## Instal·lació

Per a instal·lar les dependències necessàries s'ha d'executar:

```
pip install -r requirements.txt
```

## Execució

Per a executar qualsevol de les EDOs/EDPs es fa amb la comanda:

```
python tipus_problema/EDO.py
```

Per exemple:

```
python directe/burguers.py
```

> **Nota:** A l'inici dels fitxers hi ha una sèrie de paràmetres per "jugar" amb les equacions i observar com canvien els resultats.