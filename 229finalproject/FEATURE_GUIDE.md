# Geometry Dash Levels Dataset Feature Guide

This guide maps each feature in the project’s level dataset to its in-game meaning and, where applicable, official Geometry Dash docs keys.

---

## 1) Raw dataset structure

Each level is stored as a pair of files:

- `levels/{N}stars/metadata/{LEVEL_ID}.json`
- `levels/{N}stars/data/{LEVEL_ID}.json`

Where `N` is the star-bucket directory (1 to 10 in this project layout).

---

## 2) `metadata/*.json` features (game + docs mapping)

| Feature | Game Meaning | Docs Mapping | Notes |
|---|---|---|---|
| `id` | Unique level ID | Server Level key `1` (`levelID`) | Primary level identifier |
| `name` | Level title shown in browser/search | Server Level key `2` (`levelName`) | Human-readable name |
| `author.id` | Creator player ID | Server Level key `6` (`playerID`) | Player-side identity |
| `author.accountID` | Creator account ID | Server User key `16` (`accountID`) | Account-side identity |
| `difficulty.requested` | Requested stars during rating flow | Server Level key `39` (`starsRequested`) | Not final awarded stars |
| `difficulty.stars` | Awarded stars | Server Level key `18` (`stars`) | Often used as model label |
| `difficulty.tier` | Difficulty label text (Easy/Harder/etc.) | API field `difficulty.level.pretty` | Human-readable tier |
| `stats.downloads` | Download count | Server Level key `10` (`downloads`) | Engagement signal |
| `stats.likes` | Net likes score | Server Level key `14` (`likes`) | Likes-dislikes semantics |
| `stats.objects` | Server object count (size proxy) | Server Level key `45` (`objects`) | Large-level signal |

---

## 3) `data/*.json` features (parsed level-string objects)

`data/*.json` contains parsed level objects (`data["data"]` list).  
Each object is one placed level element.

| Object Field | Game Meaning | Level-String Mapping | Notes |
|---|---|---|---|
| `id` | Exact GD object type ID | Object property key `1` | Identity of mechanic/build element |
| `x` | Horizontal position | Object property key `2` | Placement coordinate |
| `y` | Vertical position | Object property key `3` | Placement coordinate |
| `type` | Parser category (`trigger`, `portal`, `orb`, etc.) | Parser-derived grouping | Not a single official server key |

Common parser categories in this dataset include:
- `object`
- `trigger`
- `text`
- `pad`
- `portal`
- `orb`
- `pickup`

---

## 4) Processed feature sets used for modeling

## 4.1 `data/*.csv` (basic handcrafted features)

Built from raw `data` + `metadata`.

| Feature | Meaning in game/design terms |
|---|---|
| `id` | Level ID |
| `obj_count` | Total objects (build complexity/size) |
| `trigger_count` | Trigger count (logic/event intensity) |
| `portal_count` | Portal count (mode/speed/transition density) |
| `x_min`, `x_max` | Horizontal span of placed objects |
| `y_min`, `y_max` | Vertical span of placed objects |
| `stars` | Awarded star difficulty label |

---

## 4.2 `data_v2/*.csv` (object-ID histogram)

| Feature | Meaning |
|---|---|
| `id` | Level ID |
| `stars` | Awarded stars label |
| `length` | Number of parsed objects (`len(data["data"])`) |
| `obj_<ID>` | Count of object ID `<ID>` used in this level |

Interpretation: this captures **which exact object identities** are used and how frequently.

---

## 4.3 `data_v2_spatial/*.csv` (spatial bin counts)

| Feature | Meaning |
|---|---|
| `id` | Level ID |
| `stars` | Awarded stars label |
| `length` | Number of parsed objects |
| `grid_x{i}_y{j}` | Count of objects in normalized spatial cell `(i,j)` on a 100x20 grid |

Interpretation: this captures **where** gameplay/build density occurs across the level.

---

## 5) Train/val/test split behavior

The builders use a fixed random split pattern:

- 80% train
- 10% val
- 10% test

Implemented via two `train_test_split` calls with `random_state=42` (random, not explicitly stratified).

---

## 6) Important scope note

Features such as:

- `object_count__mean`
- `trigger_ratio__p75`
- `global_chunks`
- `global_objects_per_chunk`
- and similar `__mean/__std/__pXX` columns

(found under `experiments/subsets/...`) are **later engineered aggregate features**, not raw `levels/*stars/{metadata,data}` schema fields.

Keep these separate in reporting:

1. Raw schema (`metadata` + `data`)
2. Primary processed model tables (`data`, `data_v2`, `data_v2_spatial`)
3. Later engineered experiment tables

---

## 7) Documentation sources

- GD Docs landing: <https://wyliemaster.github.io/gddocs/#/>
- Server Level resource: <https://gddocs.omgrod.me/resources/server/level/>
- Server User resource: <https://omgrod.me/gddocs2/pt/resources/server/user/>
- Client level string object keys: <https://boomlings.dev/resources/client/level-components/level-string>

---

## 8) Validation statement (for reports)

Feature definitions are grounded in:

- the project’s collection/build scripts (`collect.js`, `build_dataset*.py`)
- Geometry Dash documentation for server keys and level-string object property keys

This avoids guess-based feature descriptions and keeps mappings reproducible.
