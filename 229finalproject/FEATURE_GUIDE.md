# Geometry Dash Difficulty Feature Guide

This guide explains the feature sets used in this project, why they matter for difficulty prediction, and how each metadata field maps to Geometry Dash documentation.

## Feature Sets Used In This Project

### 1) Basic geometry features (`data`)

These are compact summary statistics of a level's layout and object composition.

- `obj_count`: total object count from parsed level string.
- `trigger_count`: number of trigger objects.
- `portal_count`: number of portal objects.
- `x_min`, `x_max`: horizontal span of placed objects.
- `y_min`, `y_max`: vertical span of placed objects.

Why this helps:

- Captures coarse complexity and space usage.
- Fast to compute and easy to interpret.
- Good low-dimensional baseline, but may miss nuanced gameplay patterns.

### 2) ID histogram features (`data_v2`)

A high-dimensional vector of object-ID frequencies (counts by object type).

Why this helps:

- Preserves composition details (which object families are used).
- Strong signal for style and difficulty patterns that basic geometry misses.
- Works well with L1 selection because sparse weights can identify the most informative IDs.

### 3) Spatial grid features (`data_v2_spatial`)

Counts of objects arranged over position bins (grid cells), optionally separated by object classes.

Why this helps:

- Adds localization information (where complexity occurs in a level).
- Can represent choke points, dense transitions, and pacing changes.
- Useful for models that benefit from local structure (e.g., CNN-like processing).

## Metadata Feature Glossary (Docs-Aligned)

The dataset metadata schema contains exactly these leaf fields:

- `id`
- `name`
- `author.id`
- `author.accountID`
- `difficulty.requested`
- `difficulty.stars`
- `difficulty.tier`
- `stats.downloads`
- `stats.likes`
- `stats.objects`

Below, each field is mapped to a documented GD concept.

### `id`

- Meaning: unique level identifier.
- Docs mapping: **levelID** (Server Level key `1`).

### `name`

- Meaning: level title.
- Docs mapping: **levelName** (Server Level key `2`).

### `author.id`

- Meaning: creator's player ID.
- Docs mapping: **playerID** on a level (Server Level key `6`) and **userID** in user resources (Server User key `2`).

### `author.accountID`

- Meaning: creator's account ID (account-level identifier, separate from player ID).
- Docs mapping: **accountID** (Server User key `16`).

### `difficulty.requested`

- Meaning: stars requested by creator/mod flow before final rating.
- Docs mapping: **starsRequested** (Server Level key `39`).

### `difficulty.stars`

- Meaning: awarded stars for completing the level.
- Docs mapping: **stars** (Server Level key `18`).

### `difficulty.tier`

- Meaning: normalized difficulty tier in this dataset.
- Docs mapping:
  - Non-demon side aligns with documented difficulty buckets derived from **difficultyNumerator / difficultyDenominator** (Server Level keys `9` and `8`, where modern values correspond to unrated/easy/normal/hard/harder/insane).
  - Demon side aligns with **demon Difficulty** (Server Level key `43`: easy/medium/hard/insane/extreme demon buckets).

### `stats.downloads`

- Meaning: total download count.
- Docs mapping: **downloads** (Server Level key `10`).

### `stats.likes`

- Meaning: net likes (likes minus dislikes semantics in server data).
- Docs mapping: **likes** (Server Level key `14`).

### `stats.objects`

- Meaning: object count used by game/server logic as a size proxy.
- Docs mapping: **objects** (Server Level key `45`).

## Why Metadata Is Useful (and Limits)

Metadata adds strong context features:

- Popularity/engagement (`downloads`, `likes`) can correlate with polished level design and rating stability.
- Creator identity fields can capture creator-specific style priors.
- Difficulty/rating fields can be strong direct signals.

But for fair evaluation:

- Avoid leakage when predicting star difficulty from features that directly encode rating outcome.
- Report with and without direct rating-like metadata so performance claims remain interpretable.

## Documentation Sources

- GD Docs landing page: https://wyliemaster.github.io/gddocs/#/
- Server Level Resource: https://gddocs.omgrod.me/resources/server/level/
- Server User Resource: https://omgrod.me/gddocs2/pt/resources/server/user/
