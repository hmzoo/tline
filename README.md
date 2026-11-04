# tline

Interface web minimaliste qui mesure la **vitesse du pointeur** et la trace en
temps réel sur une **fenêtre glissante de 3 secondes**.

- cadre **capture** : zone de survol (réticule + traînée du mouvement) ;
- cadre **vitesse · px/s** : courbe du module de la vitesse `|v|` ;
- cadre **dérivée de la vitesse · px/s²** : courbe de `dv/dt` (accélération
  signée, échelle symétrique autour de zéro) ;
- cadre **transformée (FFT) · 0–20 Hz** : spectre d'amplitude du signal de
  vitesse (résolution 0,33 Hz), barre de la fréquence dominante mise en avant.

Aucune dépendance : serveur `http.server` (stdlib) + HTML/CSS/JS sans build.

## Modes de mouvement

Le menu sous l'en-tête permet de comparer un geste **manuel** à des trajectoires
programmées, toutes injectées dans le même pipeline de mesure (traînée, courbes,
spectre) :

| Mode | Trajectoire | Lecture attendue |
| --- | --- | --- |
| `manuel` | la souris | dépend du geste |
| `constant` | gauche → droite à 420 px/s | vitesse plate, dérivée nulle, pas de pic spectral |
| `accélération` | gauche → droite, `v = 480·t` (plafond 1400 px/s) | vitesse en rampe, dérivée ≈ +480 px/s² |
| `cyclique` | aller-retour horizontal sinusoïdal à 1 Hz | vitesse en `\|cos\|`, donc pic spectral à 2 Hz |
| `cercle` | cercle à 1 Hz (rayon `min(w,h)/3.2`) | vitesse constante, dérivée nulle |
| `chaotique` | somme de 3 sinusoïdes incommensurables | vitesse irrégulière, spectre large |

Un changement de mode remet la mesure à zéro (échantillons, spectre, stats) pour
que la fenêtre affichée ne contienne qu'un seul comportement.

> Le spectre porte sur le **module** de la vitesse : pour un aller-retour à
> 1 Hz, la vitesse passe deux fois par un maximum par période, d'où un pic à
> 2 Hz. Une vitesse parfaitement constante n'a, elle, aucun pic (seuil relatif
> de 5 % de la vitesse moyenne).

## Apprentissage du geste

Deux boutons, sous le menu des modes :

| Bouton | Effet |
| --- | --- |
| **apprentissage** | démarre/arrête l'enregistrement. En mode `manuel`, chaque image (≈ 60 Hz) ajoute un couple `(vx, vy)` au jeu de données. À l'arrêt, le modèle est réentraîné et les nouvelles données sont envoyées au serveur. |
| **simuler** | rejoue une trajectoire générée par le modèle (mode `capture · modèle IA`). Désactivé tant qu'aucun modèle n'est entraîné. |
| **effacer** | vide le jeu de données en mémoire et sur le disque. |

La **barre d'espace** démarre et arrête l'apprentissage sans quitter le cadre de
capture (`preventDefault`, donc pas de défilement de page). Quand le bouton a
lui-même le focus, la touche est laissée au comportement natif pour éviter un
double basculement.

Une ligne de statut indique le nombre d'échantillons, l'état du modèle et son
écart moyen de reconstruction (px/s). Les données sont **persistées** dans
`data/dataset.jsonl` et rechargées au démarrage : le modèle survit aux
rechargements de page et s'enrichit à chaque session.

### Modèle

**Rejeu de segments recalé par plus proche voisin** :

1. chaque relevé `(vx, vy)` est stocké à ~60 Hz — vitesse réelle image par
   image, sans lissage (sinon le tremblement caractéristique disparaît) ;
2. un **état** n'est pas la vitesse instantanée mais un vecteur de `contexte`
   vitesses consécutives (curseur `contexte`, 1 → 12 points, soit jusqu'à
   ~200 ms d'histoire) ;
3. le déroulé **rejoue un fragment réel** de l'enregistrement pendant
   `segment` images (~400 ms, longueur randomisée de ± `segmentJitter`) ; au
   recalage, on retient les `ML_MATCH_K = 6` états les plus proches —
   historique proche **et** successeur cohérent avec la vitesse courante
   (pénalité `continuity`) — puis on en **tire un au hasard**, pondéré par
   `1/(score+1)` ;
4. un raccord de `blend` images amortit le saut à la jointure ;
5. le point de départ est tiré au hasard dans l'enregistrement.

Le tirage aléatoire (étape 3) et le départ aléatoire (étape 5) rendent chaque
simulation **différente**, tout en restant composée de morceaux réels du geste :
la signature est donc préservée. Mesures sur un geste enregistré (référence :
`v=232`, `p90=316`, `jerk=17 600`) — huit simulations indépendantes donnent
`v = 243 ± 21`, `p90 = 353 ± 27`, `jerk ≈ 17 100`, avec 8 points de départ
distincts et aucune dérive sur 3 600 images (60 s).

En simulation, la trajectoire n'est **pas bornée** : aucune réflexion sur les
bords. Une **caméra virtuelle** suit le point (poursuite amortie `CAM_GAIN`,
avec rattrapage si le point approche d'un bord) et la grille de fond défile —
la vitesse émise reste donc exactement celle du fragment rejoué.

Comme chaque vitesse émise provient de l'enregistrement, le déroulé reste par
construction dans la distribution du geste — contrairement à une régression
autorégressive, qui diverge ou s'éteint sur des vitesses aussi autocorrélées.

> Attention : environ deux images sur trois sont une copie exacte d'une vitesse
> enregistrée, le reste étant les raccords. Le modèle **rejoue** le geste, il ne
> l'invente pas.

L'**écart** affiché est l'erreur d'une prédiction 1-plus-proche-voisin sur un
échantillon de 120 situations : elle diminue quand le contexte augmente
(≈ 545 px/s à contexte 1 contre ≈ 360 px/s à contexte 8 sur un même geste).

### Signature du geste

Le tableau sous les commandes caractérise le mouvement sur la fenêtre de 3 s,
pour le geste (`manuel`) et pour le modèle (`simuler`) :

| Colonne | Sens |
| --- | --- |
| `v moy`, `p90` | niveau et dispersion de la vitesse (px/s) |
| `\|a\|` | accélération moyenne en valeur absolue (px/s²) |
| `jerk` | variation d'accélération — les « secousses » (px/s³) |
| `virages` | changements de direction par seconde |
| `arrêts` | part du temps sous 25 px/s |

C'est là que se lit ce qui distingue un geste : un geste nerveux a un `jerk` et
un `virages/s` élevés, un geste posé a un `p90` proche de `v moy`. Sur un geste
réaliste, le modèle reproduit ces valeurs à quelques pourcents près ; sur un
geste très tremblé, la recombination de fragments ajoute un peu de `jerk`.

### Modèles sauvegardés

La deuxième ligne de commandes gère des modèles nommés, stockés dans
`data/models/<nom>.json` :

| Élément | Rôle |
| --- | --- |
| liste déroulante | modèles existants ; la sélection charge le modèle (vitesses + hyperparamètres) |
| champ `nom du modèle` | nom du fichier ; vide → `modele-AAAAMMJJ-HHMM` |
| **sauvegarder** | écrit le modèle courant (Entrée dans le champ fait la même chose) |
| **supprimer** | supprime le modèle sélectionné (confirmation demandée) |

Format (JSON, un fichier par modèle) :

```json
{
  "version": 1,
  "name": "trackpad-2026-11-04",
  "createdAt": "2026-11-04T15:02:11Z",
  "samplingHz": 62.5,
  "window": { "context": 8, "segment": 24, "blend": 8, "continuity": 0.5 },
  "samples": 1598,
  "error": 16,
  "signature": { "v": 150, "p90": 212, "a": 317, "jerk": 11000, "turns": 6.5, "stops": 0 },
  "velocities": [[12.4, -3.1], [11.8, -4.0]]
}
```

Les trois hyperparamètres de rejeu et le contexte sont stockés avec les
donnnées : recharger un modèle restitue exactement le même comportement. Les
vitesses suffisent à reconstruire la table de transitions (< 5 ms), inutile de
la sérialiser. Ordre de grandeur : ~14 octets par point en JSON, soit ~220 Ko
pour 10 minutes de geste.

### API

| Route | Effet |
| --- | --- |
| `GET /api/dataset` | `{ count, samples }` (les 20 000 derniers couples) |
| `POST /api/dataset` | ajoute `{ samples: [[vx, vy], …] }` (50 000 max conservés) |
| `DELETE /api/dataset` | vide le jeu de données |
| `GET /api/models` | liste `[{ name, samples, createdAt, error, signature }]` |
| `GET /api/models/<nom>` | modèle complet (avec `velocities`) |
| `POST /api/models/<nom>` | écrit le modèle (nom `[A-Za-z0-9._-]`, 60 000 points max) |
| `DELETE /api/models/<nom>` | supprime le modèle |

## Lancer

```bash
python3 server.py          # http://127.0.0.1:8000
python3 server.py --open   # ouvre le navigateur
python3 server.py -p 9000  # autre port
```

Si le port demandé est occupé, le serveur essaie automatiquement les ports
suivants (jusqu'à 10) et affiche l'URL réellement utilisée.

Équivalent si vous préférez un serveur générique :

```bash
python3 -m http.server -d static 8000
```

## Structure

```
tline/
├── server.py         # serveur statique minimal (stdlib) + API /api/*
├── data/
│   ├── dataset.jsonl # couples (vx, vy) enregistrés (créé à l'usage)
│   └── models/       # modèles nommés, un JSON par modèle
└── static/
    ├── index.html    # structure de la page
    ├── style.css     # thème sombre minimaliste
    └── app.js        # capture, courbes, spectre, entraînement et simulation
```

## Détails de mesure

| Paramètre | Valeur | Rôle |
| --- | --- | --- |
| `WINDOW_MS` | 3000 | largeur de la fenêtre temporelle affichée |
| `SAMPLE_MS` | 16 | échantillonnage des courbes (~60 Hz) |
| `SMOOTH_MS` | 55 | lissage exponentiel de la vitesse instantanée |
| `ACCEL_SMOOTH_MS` | 70 | lissage exponentiel de la dérivée |
| `DECAY_MS` | 170 | décroissance de la vitesse quand le pointeur s'arrête |
| `TRAIL_MS` | 700 | longueur de la traînée dans le cadre de capture |
| `NFFT` | 256 | taille de la FFT (rééchantillonnage de la fenêtre de 3 s) |
| `SPECTRUM_MAX_HZ` | 20 | bande de fréquences affichée |
| `ML_K` | 6 | voisins utilisés pour tirer la transition suivante |
| `ML_TABLE_MAX` | 8000 | transitions conservées pour la recherche |

La vitesse est exprimée en **px/s** : `|Δp| / Δt`. Sa dérivée est calculée par
différence finie entre deux images `(v(t) − v(t−Δt)) / Δt`, en **px/s²**.
Le spectre est la FFT du signal de vitesse rééchantillonné sur 256 points
(fenêtre de Hann, moyenne retirée), avec `Δf = 1 / 3 s ≈ 0,33 Hz` ; l'amplitude
est lissée dans le temps pour éviter le scintillement. Le canvas est
redimensionné selon le `devicePixelRatio` pour rester net sur écran HiDPI.
