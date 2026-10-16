# haryana-soi-landcover

Rule-based land-cover classification of Survey of India one-inch sheets (1914–15), Haryana, India.

Accompanies: Gerrits, P. J., Klehm, C., Cothren, J., Hames, H., Porter, J., Roberts, R., Sarmah, M., Theiss, H., Vidyarthi, V., Warn, S. & Petrie, C. A. *Hexagon KH-9 Imagery using Sunspot: Regional Monitoring of At-Risk Heritage and a Century of Land Development Pressure in Haryana, India (1915–2021).* (in review).

## Usage

```bash
pip install -r requirements.txt
jupyter lab reproduce_soi_classification.ipynb
```

The notebook reproduces Table 1 (calibration of the colour rules, leave-one-sheet-out) and the 1914–15 classification of Sirsa (Figure 8a; 17.7% built-up within 1 km, Table 3).

## Contents

| file | description |
|---|---|
| `soi_landcover.py` | classifier and calibration |
| `data/reference_polygons.gpkg` | 51 reference polygons, five sheets |
| `data/reference_pixels.npz` | RGB values of pixels inside the polygons |
| `data/sirsa_soi_sample.tif` | 6 × 6 km sheet excerpt, Sirsa (band 4: on-sheet mask) |
| `data/sirsa_pipeline_harmonised.tif` | published classification, Sirsa |
| `data/sirsa_zones.gpkg` | 0–1 and 1–2 km zones, site perimeter |
| `data/sirsa_grids.json` | analysis grids, for exact reprojection |

Polygon labels predate the paper's class names: `cultivated` is the pale wash (legend: *Grass lands*), reported as grassland; `bare/other` is paper without areal colour.

Sheet scans: British Library and Cambridge University Library.

## Licence

Code MIT; polygons and derived rasters CC BY 4.0.
