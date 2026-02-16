# Wolf Habitat Suitability Modeling in Northern Italy
## Environmental Data Acquisition and Processing

### Table of Contents
1. [Study Area Definition](#study-area-definition)
2. [NDVI Acquisition (MODIS)](#ndvi-acquisition)
3. [Elevation Data (SRTM)](#elevation-data)
4. [Human Footprint Variables](#human-footprint-variables)
5. [Terrain Derivatives](#terrain-derivatives)
6. [Climate Data](#climate-data)

---

## 1. Study Area Definition

### Setup and Libraries
```r
# Set working directory
setwd("/path/to/your/project")

# Load required libraries
libs <- c("geodata", "elevatr", "terra", "sf", "rgbif", "caret", "keras3", 
          "corrplot", "dplyr", "abind", "ggplot2", "tidyterra", "ggspatial", 
          "pROC", "imageRy")
lapply(libs, require, character.only = TRUE)

# Create directory for map data
dir.create("map_data", showWarnings = FALSE)
```

### Define Study Regions
```r
# Download administrative boundaries for Italy (GADM level 1 = regions)
italy <- gadm(country = "ITA", level = 1, path = "map_data")

# Select 9 northern Italian regions relevant for wolf distribution
regions <- italy[italy$NAME_1 %in% c(
  "Emilia-Romagna", "Toscana", "Lombardia", 
  "Veneto", "Piemonte", "Trento", "Umbria", 
  "Marche", "Liguria"
), ]

# Project to UTM Zone 32N (EPSG:32632) for metric calculations
regions_utm <- project(regions, "EPSG:32632")

# Convert to sf object for spatial operations
regions_sf <- st_as_sf(regions)
```

**Ecological Reasoning:**
- These 9 regions cover the core wolf range in the Northern Apennines
- UTM projection allows accurate distance and area calculations in meters and square grid cells

**Technical Details:**
- **Spatial extent:** ~120,000 km²
- **CRS:** EPSG:32632 (UTM Zone 32N)
- **Administrative level:** Regional (NUTS-2)

### Results
[INSERT: Map showing the 9 study regions]

**Key characteristics:**
- Total area: [Xxx] km²
- Includes both Alpine and Apennine mountain ranges
- High habitat diversity: forests, grasslands, agricultural areas

---

## 2. NDVI Acquisition (MODIS)

### Create Download Grid with Buffer
```r
# Union all regions into single polygon
regions_union <- st_union(regions_sf)

# Create 15km buffer around study area to avoid edge effects
# (15km = ~0.6° at this latitude, sufficient for MODIS processing)
regions_buffered <- st_buffer(regions_union, dist = 15000)

# Create grid of download points (0.25° spacing = ~27.5 km)
grid_points <- st_make_grid(
  regions_buffered, 
  cellsize = 0.25,  # Grid spacing in degrees
  what = "centers"   # Extract center points only
) %>% 
  st_as_sf()

# Filter points within buffered region
grid_points <- grid_points[
  st_intersects(grid_points, regions_buffered, sparse = FALSE), 
]

# Create dataframe for MODISTools batch download
download_df <- data.frame(
  site_name = paste0("patch_", seq_len(nrow(grid_points))),
  lat = st_coordinates(grid_points)[, 2],
  lon = st_coordinates(grid_points)[, 1]
)

# Visualize download strategy
plot(st_geometry(regions_sf), 
     main = "MODIS Download Grid", 
     lwd = 1.5, col = "lightgray", border = "black")
plot(st_geometry(regions_buffered), 
     add = TRUE, border = "blue", lty = 2, lwd = 3)
plot(st_geometry(grid_points), 
     add = TRUE, col = "red", pch = 20, cex = 0.5)
legend("bottomleft", 
       legend = c("Study Regions", "15km Buffer", 
                  paste(nrow(grid_points), "Download Points")),
       col = c("black", "blue", "red"),
       lty = c(1, 2, NA), lwd = c(1.5, 3, NA), pch = c(NA, NA, 20))
```

**Ecological Reasoning:**
- NDVI captures vegetation productivity, a proxy for prey availability
- June date captures peak vegetation season
- 250m resolution balances detail and computational feasibility

**Technical Details:**
- **Product:** MOD13Q1 (16-day composite, 250m)
- **Band:** 250m_16_days_NDVI
- **Date:** June 10, 2023 (peak growing season)
- **Patch size:** 17km × 17km per download point
- **Buffer:** 15km to ensure complete coverage
- **Grid spacing:** 0.25° (~27.5 km)

### Results
[INSERT: Map showing download grid and coverage]

**Download statistics:**
- Total download points: [X]
- Buffer size: 15 km
- Expected coverage: Complete study area + buffer

---

### Batch Download MODIS Data
```r
# Create directory for downloaded patches
dir.create("modis_patches", showWarnings = FALSE)

# Batch download using MODISTools
# Downloads 17×17 km patches centered on each grid point
mt_batch_subset(
  df = download_df,              # Grid points dataframe
  product = "MOD13Q1",           # MODIS Vegetation Indices product
  band = "250m_16_days_NDVI",    # NDVI band
  start = "2023-06-10",          # Single date (peak season)
  end = "2023-06-10",
  km_lr = 17,                    # 17 km left-right extent
  km_ab = 17,                    # 17 km above-below extent
  out_dir = "modis_patches",
  internal = FALSE               # Save as CSV files
)
```

### Process Downloaded Patches
```r
# Function to process individual MODIS patches
# Converts CSV to raster in MODIS Sinusoidal projection
process_patch_sinusoidal <- function(file) {
  tryCatch({
    # Read CSV file
    df <- read.csv(file)
    
    # Skip if too few data points
    if (nrow(df) < 10) return(NULL)
    
    # Extract metadata from first row
    meta <- df[1, ]
    
    # MODIS Sinusoidal projection definition
    modis_crs <- paste0(
      "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 ",
      "+a=6371007.181 +b=6371007.181 +units=m +no_defs"
    )
    
    # Calculate spatial extent in Sinusoidal coordinates
    ext_sin <- c(
      meta$xllcorner,  # Left
      meta$xllcorner + (meta$ncols * meta$cellsize),  # Right
      meta$yllcorner,  # Bottom
      meta$yllcorner + (meta$nrows * meta$cellsize)   # Top
    )
    
    # Determine value column name (varies in MODISTools versions)
    data_col <- if("value" %in% names(df)) "value" else "pixel_value"
    
    # Convert to matrix (row-wise filling)
    r_mat <- matrix(
      df[[data_col]], 
      nrow = meta$nrows, 
      ncol = meta$ncols, 
      byrow = TRUE
    )
    
    # Create raster object
    r <- rast(r_mat, extent = ext_sin, crs = modis_crs)
    
    return(r)
    
  }, error = function(e) {
    return(NULL)  # Skip problematic patches
  })
}
```

**Technical Notes:**
- MODIS uses Sinusoidal projection (equal-area)
- Each patch is ~17×17 km at 250m resolution
- CSV format requires conversion to spatial raster
- Error handling ensures robust processing

---

### Create NDVI Mosaic
```r
# List all downloaded patch files
patch_files <- list.files("modis_patches", full.names = TRUE, pattern = ".csv")

cat("Processing", length(patch_files), "patches...\n")

# Process all patches
patch_list_sin <- lapply(patch_files, process_patch_sinusoidal)

# Remove failed patches (NULL values)
patch_list_sin <- patch_list_sin[!sapply(patch_list_sin, is.null)]

cat("Successfully loaded", length(patch_list_sin), "patches\n")

# Mosaic all patches together
# fun = "mean" handles overlapping areas by averaging
ndvi_sinusoidal <- do.call(mosaic, c(patch_list_sin, fun = "mean"))
```

**Technical Details:**
- Mosaic combines overlapping patches
- Mean function resolves overlap conflicts
- Result is seamless coverage in Sinusoidal projection

---

### Project and Finalize NDVI
```r
# Project from Sinusoidal to UTM Zone 32N
cat("Projecting to UTM...\n")
ndvi_utm <- project(
  ndvi_sinusoidal, 
  regions_utm,           # Use regions as template
  method = "bilinear"    # Smooth interpolation for continuous data
)

# Apply MODIS scaling factor (stored as integers × 10000)
ndvi_utm <- ndvi_utm * 0.0001

# Mask to study region boundaries
ndvi_final <- mask(ndvi_utm, regions_utm)

# Save final product
writeRaster(ndvi_final, "ndvi_norditalien_final.tif", overwrite = TRUE)
```

**NDVI Interpretation:**
- **Range:** -1 to +1 (typically 0 to 1 for vegetation)
- Higher NDVI values indicate denser vegetation (potential wolf habitat)
- Low NDVI areas correspond to urban zones and agricultural lands
- Apennine mountain forests show highest NDVI values

---

---

## 3. Elevation Data (SRTM)

### Download High-Resolution Elevation
```r
# Download elevation data using elevatr package
# Source: SRTM (Shuttle Radar Topography Mission) at ~90m resolution
elevation_raw <- get_elev_raster(
  locations = regions_sf,
  z = 10,              # Zoom level 10 = high resolution
  clip = "bbox"        # Clip to bounding box
)

# Convert to terra SpatRaster format
elevation_terra <- rast(elevation_raw)
```

**Ecological Reasoning:**
- Elevation affects temperature, vegetation, and prey distribution
- High-resolution DEM captures fine-scale topographic features

**Technical Details:**
- **Source:** SRTM (Shuttle Radar Topography Mission)
- **Original resolution:** ~90m at this latitude


---

### Project to UTM and Resample to 100m
```r
# Project from WGS84 to UTM Zone 32N
elevation_utm_highres <- project(
  elevation_terra, 
  "EPSG:32632",        # UTM Zone 32N
  method = "bilinear"  # Smooth interpolation for continuous elevation
)

# Create 100m resolution template
template_100m <- rast(
  ext(regions_utm),    # Use study area extent
  res = 100,           # 100m pixel size
  crs = "EPSG:32632"   # UTM Zone 32N
)

# Resample to exactly 100m resolution to ensure alignment with all other environmental layers

# Calculate aggregation factor
current_res <- res(elevation_utm_highres)[1]
target_res <- 100
agg_factor <- target_res / current_res

cat("Aggregation factor:", round(agg_factor, 2), "\n")

# Aggregate to 100m using mean (preserves elevation patterns)
elevation_100m <- aggregate(
  elevation_utm_highres, 
  fact = agg_factor,     # Aggregation factor
  fun = "mean"           # Average elevation in each 100m cell
)

# Mask to study region boundaries
elevation_final <- mask(elevation_100m, regions_utm)

# Save elevation layer
writeRaster(elevation_final, "elevation_100m.tif", overwrite = TRUE)
```

**Why 100m Resolution?**
1. **Computational feasibility:** 100m balances detail and processing speed
2. **Alignment:** All variables use same grid (critical for CNN)
3. **Ecological relevance:** Captures landscape features at wolf movement scale

---

## 4. Terrain Derivatives

### Slope Calculation
```r
# Calculate slope from elevation using terra::terrain
slope <- terrain(
  elevation_final, 
  v = "slope",          # Calculate slope
  unit = "degrees"      # Output in degrees (0-90°)
)

# Mask to study area
slope_final <- mask(slope, regions_utm)

# Save
writeRaster(slope_final, "slope_100m.tif", overwrite = TRUE)

```

**Ecological Reasoning:**
- Slope affects wolf movement costs and hunting strategies

**Technical Details:**
- **Algorithm:** Horn's method (3×3 moving window)
- **Units:** Degrees (0° = flat, 90° = vertical)
- **Calculation:** arctan(√(dz/dx)² + (dz/dy)²)

---

### Roughness Calculation
```r
# Calculate terrain roughness using imageRy

# Roughness = standard deviation of elevation in 3×3 window
# Measures local topographic complexity
roughness_result <- im.kernel(
  elevation_final, 
  mw = 3,              # 3×3 moving window (300m × 300m)
  stat = "sd"          # Standard deviation
)

# Extract SD layer (roughness)
roughness <- roughness_result[["sd"]]

# Mask to study area
roughness_final <- mask(roughness, regions_utm)

# Save
writeRaster(roughness_final, "roughness_100m.tif", overwrite = TRUE)

```

**Ecological Reasoning:**
- Roughness quantifies terrain complexity beyond simple slope
- High roughness indicates broken, irregular terrain
- Smooth terrain (low roughness) often indicates agricultural/human modification

**Technical Details:**
- **Window size:** 3×3 cells (300m × 300m)
- **Metric:** Standard deviation of elevation
- **Units:** Meters of elevation variation
- **Interpretation:** Higher values = more topographically complex


**Terrain summary:**
- **Elevation:** Captures absolute height (prey distribution, climate)
- **Slope:** Captures steepness (movement costs, prey accessibility)  
- **Roughness:** Captures complexity (cover, den sites, refugia)

---

## 5. Climate Data (WorldClim)

### Download Temperature Data
```r
# Download WorldClim BioClim variables for Italy

# WorldClim provides 19 bioclimatic variables at ~1km resolution
# We use country-specific download for faster processing
worldclim <- worldclim_country(
  country = "ITA",       # Italy only
  var = "bio",           # Bioclimatic variables
  res = 0.5,             # 30 arc-seconds (~1km at equator)
  path = "climate_data"  # Save location
)

# Extract BIO1 = Annual Mean Temperature
temp_annual <- worldclim[[1]]

```

**Ecological Reasoning:**
- Temperature influences wolf physiology, prey availability, and habitat use
- Wolves are thermoregulatory generalists
- Temperature affects:
  - Snow depth and duration (hunting efficiency)
  - Prey distribution 
  - Vegetation productivity 
  - Den site selection (thermal refugia)

**WorldClim Variables Used:**
- **BIO1:** Annual Mean Temperature
  - Most stable climate metric
  - Captures broad thermal gradient
  - Less affected by extreme events

**Alternative variables considered:**
- BIO5 (Max temp warmest month): Summer stress
- BIO6 (Min temp coldest month): Winter harshness
- BIO7 (Temperature annual range): Thermal variability

---

### Process and Resample Temperature
```r
# Crop to study area 
temp_cropped <- crop(temp_annual, vect(regions_sf))

# Project to UTM Zone 32N
temp_utm <- project(
  temp_cropped, 
  "EPSG:32632", 
  method = "bilinear"  # Smooth interpolation for temperature
)

# Convert from integer (°C × 10) to actual °C
# WorldClim stores temperature as integers to save space
temp_utm <- temp_utm / 10

# Resample to 100m grid (matches all other variables)
temp_100m <- resample(
  temp_utm, 
  template_100m, 
  method = "bilinear"  # Interpolate temperature values
)

# Mask to study region
temp_final <- mask(temp_100m, regions_utm)

# Save
writeRaster(temp_final, "temperature_bio_100m.tif", overwrite = TRUE)

```

**Technical Notes:**
- **Original resolution:** ~1 km (30 arc-seconds)
- **Downsampled to:** 100m (for CNN compatibility)
- **Interpolation:** Bilinear (smooth temperature gradients)
- **Temporal coverage:** 1970-2000 climate normals

**Resampling justification:**
- Temperature varies smoothly across landscape
- 1km → 100m interpolation is valid (no sharp boundaries)
- Captures elevation-temperature relationship
- Maintains alignment with 100m grid

---


## 6. Human Footprint Variables

### Overview
Human infrastructure strongly influences wolf distribution through:
1. **Direct mortality:** Vehicle collisions, poaching
2. **Habitat fragmentation:** Roads/railways create barriers
3. **Disturbance:** Human activity reduces habitat quality
4. **Prey depletion:** Hunting pressure on ungulates

We quantify human footprint using OpenStreetMap (OSM) data:
- **Road density:** All road types (motorways to tracks)
- **Railway density:** Active rail lines
- **Population density:** Human population per 100m pixel

**Data source:** OpenStreetMap (OSM)
- **Advantages:** Free, detailed, regularly updated
- **Spatial extent:** Three OSM regions cover study area
  - Centro (Central Italy)
  - Nord-Est (Northeast Italy)
  - Nord-Ovest (Northwest Italy)

---

### 6.1 Road Density
```r
# Base path to OSM shapefiles (adjust to your directory)
base_path <- "/path/to/your/osm/data"

# Load road shapefiles from three OSM regions
cat("Loading road shapefiles...\n")

roads_centro <- st_read(
  file.path(base_path, "centro-260213-free/gis_osm_roads_free_1.shp")
)
roads_nordest <- st_read(
  file.path(base_path, "nord-est-260213-free/gis_osm_roads_free_1.shp")
)
roads_nordovest <- st_read(
  file.path(base_path, "nord-ovest-260213-free/gis_osm_roads_free_1.shp")
)

# Combine all road networks
roads_all <- rbind(roads_centro, roads_nordest, roads_nordovest)
```

**OSM Road Classification:**
- **Motorways:** High-speed highways (major barrier)
- **Primary/Secondary:** Major roads (moderate-high traffic)
- **Tertiary:** Minor roads (moderate traffic)
- **Residential:** Local streets (low-moderate traffic)
- **Track/Path:** Unpaved roads (minimal barrier)

---

### Process Roads: Fast Method
```r
# Transform to UTM for metric calculations
roads_utm <- st_transform(roads_all, crs = 32632)

# Fast crop using bounding box (much faster than st_intersection)
# st_crop clips to rectangular bounding box
roads_bbox <- st_crop(roads_utm, st_bbox(st_as_sf(regions_utm)))

# Rasterize roads to 100m grid
# Each cell gets value 1 if road present, 0 otherwise
roads_raster <- rasterize(
  vect(roads_bbox),      # Convert sf to SpatVector
  template_100m,         # 100m template grid
  field = 1,             # Assign value 1 to road cells
  background = 0         # Non-road cells = 0
)

# Calculate road density using focal window
# Focal window = proportion of cells with roads in 500m radius
road_density <- focal(
  roads_raster, 
  w = 5,                # 5×5 cell window = 500m × 500m
  fun = "mean",         # Mean of 0/1 = proportion with roads
  na.rm = TRUE
)

# Mask to study area boundaries
road_density_final <- mask(road_density, regions_utm)

# Save
writeRaster(road_density_final, "road_density_100m.tif", overwrite = TRUE)

```

**Focal Window Rationale:**
- **Window size:** 500m × 500m (5×5 cells at 100m resolution)
- **Ecological justification:** 
  - we assume that 500m approximates wolf road avoidance distance
  - Captures local road network density
- **Output values:** 0 to 1 (proportion of cells with roads)
  - 0 = no roads within 500m
  - 0.5 = 50% of cells have roads (moderate density)
  - 1.0 = complete road coverage (urban/highway interchange)

**Technical optimization:**
- `st_crop()` with bbox is 10-100× faster than `st_intersection()`

---

### 6.2 Railway Density
```r
# Load railway shapefiles from three regions

railways_centro <- st_read(
  file.path(base_path, "centro-260213-free/gis_osm_railways_free_1.shp")
)
railways_nordest <- st_read(
  file.path(base_path, "nord-est-260213-free/gis_osm_railways_free_1.shp")
)
railways_nordovest <- st_read(
  file.path(base_path, "nord-ovest-260213-free/gis_osm_railways_free_1.shp")
)

# Combine all railways
railways_all <- rbind(railways_centro, railways_nordest, railways_nordovest)

# Filter for active railways only
# Exclude abandoned, disused, and under-construction lines
railways_active <- railways_all %>%
  filter(!fclass %in% c("abandoned", "disused", "construction"))

```

**Railway Types:**
- **Rail:** Main passenger/freight lines (major barrier)
- **Light rail:** Trams, metros (urban areas)
- **Subway:** Underground (minimal surface impact)
- **Narrow gauge:** Historic/tourist lines (minor barrier)
- **Abandoned/Disused:** Excluded (not functional barriers)

**Ecological Impact:**
- Railways are effective barriers (fenced, fast trains)
- Less dense than roads but higher mortality risk per crossing
- Concentrate in valleys (follow terrain contours)
- Often parallel major roads (cumulative barrier effect)

---

### Process Railways
```r
# Transform and crop railways
railways_utm <- st_transform(railways_active, crs = 32632)
railways_cropped <- st_crop(railways_utm, st_bbox(st_as_sf(regions_utm)))

# Rasterize
railways_raster_100m <- rasterize(
  vect(railways_cropped), 
  template_100m, 
  field = 1,
  background = 0
)

# Calculate railway density (500m focal window)
cat("Calculating railway density (500m window)...\n")
railway_density_focal <- focal(
  railways_raster_100m, 
  w = 5,                # 500m × 500m window
  fun = "mean",         # Proportion with railways
  na.rm = TRUE
)

# Mask and save
railway_density_final <- mask(railway_density_focal, regions_utm)
writeRaster(railway_density_final, "railway_density_100m.tif", overwrite = TRUE)

```

---

### 6.4 Population Density (GHS-POP)
```r
# Population data from Global Human Settlement Layer (GHS-POP)
# Source: European Commission Joint Research Centre
# Resolution: 100m (perfect for our grid!)
# Year: 2025-2030 average

# GHS-POP data comes in global tiles
# Northern Italy requires 2 tiles (columns 19 and 20, row 5)
tile1 <- rast("GHS_POP_E2030_GLOBE_R2023A_4326_3ss_V1_0_R5_C19.tif")
tile2 <- rast("GHS_POP_E2030_GLOBE_R2023A_4326_3ss_V1_0_R5_C20.tif")

# Merge tiles into single raster
ghs_merged <- merge(tile1, tile2)

# Crop to study area (in WGS84 before projection)
regions_wgs84 <- project(regions, "EPSG:4326")
ghs_cropped <- crop(ghs_merged, vect(regions_wgs84))

```

**GHS-POP Dataset:**
- **Full name:** Global Human Settlement Layer - Population Grid
- **Provider:** European Commission JRC
- **Resolution:** 100m (3 arc-seconds)
- **Temporal coverage:** 1975-2030 (projections)
- **Methodology:** Disaggregates census data using built-up area
- **Units:** Persons per 100m × 100m cell

**Why GHS-POP over other datasets?**
1. **High resolution:** 100m (vs 1km for WorldPop)
2. **Recent:** 2030 projection captures current patterns
3. **Quality:** Based on Copernicus Sentinel imagery
4. **Free:** Open data, globally consistent

---

### Project and Resample Population Data
```r
# CRITICAL: Use "near" method, NOT "bilinear"
# Population is COUNT data - interpolation loses people!

ghs_utm <- project(
  ghs_cropped, 
  "EPSG:32632",
  method = "near"       # Nearest neighbor = no interpolation
)

# Resample to 100m template grid (using "near" again)
ghs_100m <- resample(
  ghs_utm, 
  template_100m, 
  method = "near"       # tries to preserve population counts
)

# Mask to study region
ghs_masked <- mask(ghs_100m, regions_utm)

# Replace NAs inside regions with 0 (no people vs. no data)
ghs_final <- ghs_masked
ghs_final[is.na(ghs_final)] <- 0
ghs_final <- mask(ghs_final, regions_utm)  # Re-apply mask for outside areas

# Save
writeRaster(ghs_final, "population_ghs_100m.tif", overwrite = TRUE)

```

---

**Ecological implications:**
- Population density is a negative predictor
- Zero-population areas are rare in Northern Italy

---


## 8. Distance to Water

### Load Water Features
```r
# Water data comes in two OSM layers:
# 1. water_a: Water bodies (polygons) - lakes, reservoirs
# 2. waterways: Rivers, streams (lines)

# Water bodies (polygons)
water_centro <- st_read(
  file.path(base_path, "centro-260213-free/gis_osm_water_a_free_1.shp")
)
water_nordest <- st_read(
  file.path(base_path, "nord-est-260213-free/gis_osm_water_a_free_1.shp")
)
water_nordovest <- st_read(
  file.path(base_path, "nord-ovest-260213-free/gis_osm_water_a_free_1.shp")
)

# Waterways (lines)
waterways_centro <- st_read(
  file.path(base_path, "centro-260213-free/gis_osm_waterways_free_1.shp")
)
waterways_nordest <- st_read(
  file.path(base_path, "nord-est-260213-free/gis_osm_waterways_free_1.shp")
)
waterways_nordovest <- st_read(
  file.path(base_path, "nord-ovest-260213-free/gis_osm_waterways_free_1.shp")
)

# Combine
water_all <- rbind(water_centro, water_nordest, water_nordovest)
waterways_all <- rbind(waterways_centro, waterways_nordest, waterways_nordovest)

```

**Ecological Importance of Water:**
- **Drinking:** Wolves need water daily (prey do too)
- **Prey concentration:** Ungulates visit water sources
- **Thermoregulation:** Cooling in summer heat
- **Den sites:** Often near water for pup-rearing
- **Corridors or barriers:**

---

### Filter Major Waterways
```r
# Filter for permanent, major waterways
# Exclude ephemeral streams and ditches
waterways_major <- waterways_all %>%
  filter(fclass %in% c("river", "stream", "canal"))

```

**Waterway Classification:**
- **River:** Large permanent flows (Po, Adige, Arno)
- **Stream:** Smaller permanent flows (mountain streams)
- **Canal:** Artificial waterways (irrigation, navigation)
- **Ditch:** Excluded (too small, often seasonal)
- **Drain:** Excluded (minor features)

---

### Calculate Distance to Water
```r
# Transform to UTM
water_utm <- st_transform(water_all, crs = 32632)
waterways_utm <- st_transform(waterways_major, crs = 32632)

# Crop to study area
water_cropped <- st_crop(water_utm, st_bbox(st_as_sf(regions_utm)))
waterways_cropped <- st_crop(waterways_utm, st_bbox(st_as_sf(regions_utm)))

# Rasterize both water types
water_raster <- rasterize(
  vect(water_cropped), 
  template_100m, 
  field = 1,          # Value doesn't matter (just presence/absence)
  background = NA
)

waterways_raster <- rasterize(
  vect(waterways_cropped), 
  template_100m, 
  field = 1,
  background = NA
)

# Combine water bodies and waterways
# cover() fills NAs in first raster with values from second
water_combined <- cover(water_raster, waterways_raster)

# Calculate Euclidean distance to nearest water
water_distance <- distance(water_combined)  # Returns distance in meters

# Mask to study area
water_distance_final <- mask(water_distance, regions_utm)

# Save
writeRaster(water_distance_final, "water_distance_100m.tif", overwrite = TRUE)

```

**Distance Calculation:**
- `distance()` computes Euclidean distance (straight-line)
- Distance in meters (UTM projection)
- Every cell gets distance to nearest water pixel
- Water pixels themselves have distance = 0

---



## 9. Variable Standardization and CNN Data Preparation

### Overview
Before training the Convolutional Neural Network (CNN), all environmental variables must be standardized to ensure:
1. **Equal contribution:** Variables on different scales (e.g., elevation in meters vs. NDVI 0-1) contribute equally
2. **Faster convergence:** Standardized inputs improve training speed
3. **Numerical stability:** Prevents gradient explosion/vanishing

We use **Z-score standardization** (mean = 0, standard deviation = 1):
```
Z = (X - μ) / σ
```

Where:
- X = original value
- μ = mean of variable
- σ = standard deviation
- Z = standardized value

---

### Load All Environmental Layers
```r
# =============================================================================
# LOAD ALL ENVIRONMENTAL VARIABLES (100m RESOLUTION)
# =============================================================================

library(terra)
library(dplyr)

cat("📥 Loading all environmental layers...\n")

# Load individual layers
elevation <- rast("elevation_100m.tif")
slope <- rast("slope_100m.tif")
roughness <- rast("roughness_100m.tif")
temperature <- rast("temperature_bio_100m.tif")
ndvi <- rast("ndvi_norditalien_final.tif")
population <- rast("population_ghs_100m.tif")
road_density <- rast("road_density_100m.tif")
railway_density <- rast("railway_density_100m.tif")
water_distance <- rast("water_distance_100m.tif")
landuse <- rast("landuse_wolf_habitat_100m.tif")

cat("✅ All layers loaded\n\n")
```

**Quality Control Check:**
```r
# Verify all layers have same properties
cat("Checking layer alignment...\n")

layers <- list(elevation, slope, roughness, temperature, ndvi, 
              population, road_density, railway_density, water_distance)

# Check if all have same extent and resolution
all_aligned <- all(sapply(layers[-1], function(x) {
  compareGeom(layers[[1]], x, stopOnError = FALSE)
}))

if (all_aligned) {
  cat("✅ All layers perfectly aligned\n")
} else {
  cat("⚠️  Layers not aligned - checking which ones...\n")
  for (i in 2:length(layers)) {
    if (!compareGeom(layers[[1]], layers[[i]], stopOnError = FALSE)) {
      cat("  Layer", i, "misaligned\n")
    }
  }
}
```

---

### Resample NDVI if Necessary
```r
# NDVI may have different resolution from other layers
# Resample to match 100m grid

if (!compareGeom(elevation, ndvi, stopOnError = FALSE)) {
  cat("⚠️  NDVI has different geometry - resampling to 100m grid...\n")
  
  ndvi <- resample(
    ndvi, 
    elevation,           # Use elevation as template
    method = "bilinear"  # Smooth interpolation for continuous data
  )
  
  cat("✅ NDVI resampled to match other layers\n\n")
} else {
  cat("✅ NDVI already aligned with other layers\n\n")
}
```

**Why resample NDVI separately?**
- NDVI may come from different source (MODIS at 250m originally)
- Other variables were all created at 100m resolution
- All inputs to CNN must have identical spatial properties

---

### Create Multi-Layer Stack
```r
# =============================================================================
# CREATE ENVIRONMENTAL STACK
# =============================================================================

cat("📚 Creating multi-layer environmental stack...\n")

# Stack all layers into single raster object
env_stack <- c(
  elevation, 
  slope, 
  roughness, 
  temperature, 
  ndvi,
  population, 
  road_density, 
  railway_density, 
  water_distance
)

# Assign meaningful names
names(env_stack) <- c(
  "Elevation", 
  "Slope", 
  "Roughness", 
  "Temperature", 
  "NDVI",
  "Population", 
  "Road_Density", 
  "Railway_Density", 
  "Water_Distance"
)

# Save unstandardized stack (for reference)
writeRaster(env_stack, "environmental_stack_100m.tif", overwrite = TRUE)

cat("✅ Environmental stack created:\n")
cat("   Layers:", nlyr(env_stack), "\n")
cat("   Resolution:", res(env_stack)[1], "m\n")
cat("   Total cells:", format(ncell(env_stack[[1]]), big.mark = ","), "\n")
cat("   File size:", round(file.size("environmental_stack_100m.tif") / 1e6, 1), "MB\n\n")
```

**Stack Properties:**
- **Dimensions:** [X] rows × [X] columns
- **Total pixels:** [X] million per layer
- **Total cells:** [X] million × 9 variables = [X] million values
- **Memory:** ~[X] GB in RAM when fully loaded

---

### Z-Score Standardization
```r
# =============================================================================
# Z-SCORE STANDARDIZATION FOR CNN
# =============================================================================

cat("📊 Performing Z-Score standardization...\n\n")

# Initialize empty stack for standardized layers
env_stack_scaled <- rast()

# Create dataframe to store scaling parameters
# CRITICAL: Save these for later de-standardization of predictions!
scaling_params <- data.frame(
  variable = character(),
  mean = numeric(),
  sd = numeric(),
  min_original = numeric(),
  max_original = numeric(),
  stringsAsFactors = FALSE
)

# Standardize each layer
for (i in 1:nlyr(env_stack)) {
  var_name <- names(env_stack)[i]
  layer <- env_stack[[i]]
  
  cat("  Processing:", var_name, "\n")
  
  # Calculate statistics
  mean_val <- global(layer, "mean", na.rm = TRUE)[[1]]
  sd_val <- global(layer, "sd", na.rm = TRUE)[[1]]
  min_val <- global(layer, "min", na.rm = TRUE)[[1]]
  max_val <- global(layer, "max", na.rm = TRUE)[[1]]
  
  # Apply Z-score transformation: (X - mean) / SD
  layer_scaled <- (layer - mean_val) / sd_val
  
  # Add to scaled stack
  if (i == 1) {
    env_stack_scaled <- layer_scaled
  } else {
    env_stack_scaled <- c(env_stack_scaled, layer_scaled)
  }
  
  # Store scaling parameters (needed for inverse transformation)
  scaling_params <- rbind(scaling_params, data.frame(
    variable = var_name,
    mean = mean_val,
    sd = sd_val,
    min_original = min_val,
    max_original = max_val
  ))
  
  # Report transformation
  cat("    Original range: [", round(min_val, 2), ", ", 
      round(max_val, 2), "]\n", sep = "")
  cat("    Standardized: mean =", 
      round(global(layer_scaled, "mean", na.rm = TRUE)[[1]], 6), 
      ", sd =", round(global(layer_scaled, "sd", na.rm = TRUE)[[1]], 6), "\n\n")
}

# Assign names to scaled stack
names(env_stack_scaled) <- names(env_stack)

# Save standardized stack
writeRaster(env_stack_scaled, "env_stack_scaled_100m.tif", overwrite = TRUE)

cat("✅ Standardization complete\n")
cat("   Saved: env_stack_scaled_100m.tif\n\n")
```

**Standardization Results:**

| Variable | Original Min | Original Max | Mean | SD | Scaled Mean | Scaled SD |
|----------|-------------|--------------|------|-----|-------------|-----------|
| Elevation | [X] m | [X] m | [X] | [X] | ~0.000 | ~1.000 |
| Slope | [X]° | [X]° | [X] | [X] | ~0.000 | ~1.000 |
| Roughness | [X] m | [X] m | [X] | [X] | ~0.000 | ~1.000 |
| Temperature | [X]°C | [X]°C | [X] | [X] | ~0.000 | ~1.000 |
| NDVI | [X] | [X] | [X] | [X] | ~0.000 | ~1.000 |
| Population | [X] | [X] | [X] | [X] | ~0.000 | ~1.000 |
| Road Density | [X] | [X] | [X] | [X] | ~0.000 | ~1.000 |
| Railway Density | [X] | [X] | [X] | [X] | ~0.000 | ~1.000 |
| Water Distance | [X] m | [X] m | [X] | [X] | ~0.000 | ~1.000 |

---

### Save Scaling Parameters
```r
# =============================================================================
# SAVE SCALING PARAMETERS (CRITICAL FOR LATER!)
# =============================================================================

cat("💾 Saving scaling parameters...\n")

# Save as CSV for easy inspection
write.csv(scaling_params, "scaling_parameters.csv", row.names = FALSE)

cat("✅ Saved: scaling_parameters.csv\n\n")

# Display scaling parameters
cat("Scaling parameters:\n")
print(scaling_params)
```

**Why save scaling parameters?**

These parameters are **essential** for:
1. **De-standardizing CNN predictions** back to original units
2. **Applying same transformation to new data** (e.g., validation from different year)
3. **Interpreting model results** in original units
4. **Reproducibility** of entire workflow

**Formula for de-standardization:**
```r
# To convert standardized value back to original:
original_value = (standardized_value × SD) + Mean

# Example for elevation:
# If standardized = 2.5, mean = 800m, sd = 400m
# Then: original = (2.5 × 400) + 800 = 1800m
```

---

### Visualization: Standardized vs Original
```r
# =============================================================================
# VISUALIZE STANDARDIZATION EFFECT
# =============================================================================

library(ggplot2)
library(tidyterra)
library(patchwork)

cat("📊 Creating comparison visualization...\n")

# Function to create side-by-side plots
plot_comparison <- function(original, standardized, var_name) {
  p1 <- ggplot() +
    geom_spatraster(data = original) +
    scale_fill_viridis_c(na.value = "gray90") +
    labs(title = paste0(var_name, " (Original)"), fill = "") +
    theme_minimal()
  
  p2 <- ggplot() +
    geom_spatraster(data = standardized) +
    scale_fill_viridis_c(na.value = "gray90") +
    labs(title = paste0(var_name, " (Standardized)"), fill = "Z-Score") +
    theme_minimal()
  
  return(p1 | p2)
}

# Example: Compare elevation
comparison_elev <- plot_comparison(
  env_stack[["Elevation"]], 
  env_stack_scaled[["Elevation"]], 
  "Elevation"
)

print(comparison_elev)
ggsave("plots/standardization_comparison_elevation.png", 
       width = 12, height = 5, dpi = 300)
```

### Results
[INSERT: Side-by-side comparison of original vs standardized elevation]

**Key observations:**
- Spatial patterns identical (only scale changes)
- Standardized values typically range from -3 to +3
- Most values within ±2 standard deviations (95% of data)
- Extreme values (mountain peaks, urban centers) show as outliers

**Interpretation of standardized values:**
- **Z = 0:** Average value for the study area
- **Z = +1:** One standard deviation above mean
- **Z = -1:** One standard deviation below mean
- **Z > +2:** Unusually high (top ~2.5% of values)
- **Z < -2:** Unusually low (bottom ~2.5% of values)

---

### Distribution Analysis with Ridgeline Plots
```r
# =============================================================================
# RIDGELINE PLOT OF STANDARDIZED DISTRIBUTIONS
# =============================================================================

library(imageRy)

cat("📊 Creating ridgeline plot of standardized distributions...\n")

# Downsample for visualization (full resolution too slow)
# Aggregate to 500m (factor of 5)
env_stack_agg <- aggregate(
  env_stack_scaled, 
  fact = 5,              # 100m → 500m
  fun = "mean",          # Average values in 5×5 windows
  na.rm = TRUE
)

cat("  Downsampled from", res(env_stack_scaled)[1], "m to", 
    res(env_stack_agg)[1], "m for visualization\n")

# Create ridgeline plot using imageRy function
ridge_plot <- im.ridgeline(
  env_stack_agg, 
  scale = 3,             # Vertical scale (amount of overlap)
  palette = "viridis"    # Color palette
) +
  labs(
    title = "Distribution of Environmental Variables (Standardized)",
    subtitle = "Z-Score normalized (Mean = 0, SD = 1) - Northern Italy Wolf Habitat",
    x = "Standard Deviations from Mean",
    y = ""
  ) +
  scale_x_continuous(
    breaks = seq(-4, 4, 1),
    limits = c(-3, 3)    # Focus on main distribution
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "right",
    axis.text.y = element_text(size = 11),
    axis.text.x = element_text(size = 10),
    axis.title = element_text(size = 12, face = "bold")
  )

print(ridge_plot)

# Save
ggsave("plots/ridgeline_standardized.png", ridge_plot, 
       width = 12, height = 10, dpi = 300)

cat("✅ Ridgeline plot saved\n\n")
```

### Results
[INSERT: Ridgeline plot showing standardized distributions]

**Distribution characteristics:**

**Normal-like distributions:**
- Elevation, Temperature: Approximately normal (bell-shaped)
- NDVI: Slightly left-skewed (more high values)

**Right-skewed distributions:**
- Population: Extreme right skew (most areas have 0-10 people)
- Road/Railway Density: Right-skewed (sparse in mountains)
- Water Distance: Right-skewed (most areas near water)

**Multi-modal distributions:**
- Slope: Bimodal (flat valleys vs steep mountains)
- Roughness: Multiple peaks (different terrain types)

**Implications for CNN:**
- Skewed variables still standardized (mean = 0, sd = 1)
- CNN learns from relative patterns, not absolute distributions
- Standardization ensures all variables contribute equally

---

## 10. Correlation Analysis and Variable Selection

### Purpose
Identify and remove highly correlated variables to:
1. **Reduce redundancy:** Correlated variables provide duplicate information
2. **Improve model interpretability:** Each variable contributes unique information
3. **Reduce overfitting:** Fewer variables = simpler model = better generalization
4. **Decrease computation:** Fewer channels = faster training

**Correlation threshold:** |r| > 0.7 indicates high correlation

---

### Sample Data for Correlation Analysis
```r
# =============================================================================
# SAMPLE DATA FOR CORRELATION MATRIX
# =============================================================================

cat("📊 Sampling data for correlation analysis...\n")

# Sample 50,000 pixels (sufficient for stable correlation estimates)
n_sample <- 50000

set.seed(123)  # Reproducible sampling

# Generate random sample indices
sample_indices <- sample(
  1:ncell(env_stack_scaled[[1]]), 
  size = min(n_sample, ncell(env_stack_scaled[[1]])), 
  replace = FALSE
)

# Extract values for sampled cells
sample_data <- data.frame(cell = sample_indices)

for (i in 1:nlyr(env_stack_scaled)) {
  var_name <- names(env_stack_scaled)[i]
  sample_data[[var_name]] <- env_stack_scaled[[i]][sample_indices]
}

# Remove cells with any NA values
sample_data <- sample_data %>% 
  select(-cell) %>%
  na.omit()

cat("✅ Sample data prepared:\n")
cat("   Sampled pixels:", n_sample, "\n")
cat("   Valid pixels (no NAs):", nrow(sample_data), "\n")
cat("   Variables:", ncol(sample_data), "\n\n")
```

**Sampling Strategy:**
- **Size:** 50,000 pixels (~0.3% of study area)
- **Method:** Simple random sampling
- **Justification:** Correlation estimates stable with >10,000 samples
- **NA handling:** Complete cases only (remove any row with NA)

---

### Calculate Correlation Matrix
```r
# =============================================================================
# COMPUTE PEARSON CORRELATION MATRIX
# =============================================================================

cat("🔢 Computing correlation matrix...\n")

# Calculate pairwise Pearson correlations
cor_matrix <- cor(sample_data, use = "complete.obs")

cat("✅ Correlation matrix computed\n\n")

# Display correlation matrix (rounded for readability)
cat("Correlation matrix:\n")
print(round(cor_matrix, 2))
cat("\n")
```

**Correlation Matrix Interpretation:**
- **Diagonal = 1.00:** Each variable perfectly correlated with itself
- **r > 0.7:** Strong positive correlation
- **r < -0.7:** Strong negative correlation
- **|r| < 0.3:** Weak correlation (variables mostly independent)

---

### Visualize Correlation Matrix
```r
# =============================================================================
# CORRELATION PLOT (CORRPLOT)
# =============================================================================

library(corrplot)

cat("📊 Creating correlation plot...\n")

# Create high-resolution correlation plot
png("plots/correlation_matrix.png", width = 3000, height = 3000, res = 300)

corrplot(
  cor_matrix, 
  method = "color",           # Color-coded cells
  type = "upper",             # Show upper triangle only
  order = "hclust",           # Hierarchical clustering (groups similar vars)
  tl.col = "black",           # Text label color
  tl.srt = 45,                # Text label rotation (45°)
  tl.cex = 1.2,               # Text label size
  addCoef.col = "black",      # Show correlation coefficients
  number.cex = 0.8,           # Coefficient text size
  col = colorRampPalette(c("#6D9EC1", "white", "#E46726"))(200),  # Blue-White-Orange
  title = "Correlation Matrix - Environmental Variables",
  mar = c(0, 0, 2, 0)         # Plot margins
)

dev.off()

cat("✅ Correlation plot saved: plots/correlation_matrix.png\n\n")
```

### Results
[INSERT: Correlation matrix heatmap]

**Key correlations identified:**

**Strong positive correlations (r > 0.7):**
- [Variable 1] ↔ [Variable 2]: r = [X] (explain why)
- [Variable 3] ↔ [Variable 4]: r = [X] (explain why)

**Strong negative correlations (r < -0.7):**
- [Variable A] ↔ [Variable B]: r = [X] (explain why)

**Weak correlations (|r| < 0.3):**
- Most variable pairs show weak correlation (good!)
- Indicates variables capture different aspects of habitat

**Ecological interpretation:**
- Expected correlations: 
  - Elevation ↔ Temperature (negative, due to lapse rate)
  - Elevation ↔ Population (negative, people live in valleys)
  - Road ↔ Railway Density (positive, both follow transport corridors)

- Unexpected correlations:
  - [If any surprising correlations found, explain]

---

### Identify Highly Correlated Variables
```r
# =============================================================================
# FIND HIGH-CORRELATION PAIRS
# =============================================================================

cat("⚠️  IDENTIFYING HIGHLY CORRELATED VARIABLES (|r| > 0.7)\n")
cat(rep("=", 70), "\n\n", sep = "")

# Find all pairs with |correlation| > 0.7 (excluding diagonal)
high_cor <- which(
  abs(cor_matrix) > 0.7 & cor_matrix != 1, 
  arr.ind = TRUE
)

if (nrow(high_cor) > 0) {
  # Create dataframe of high-correlation pairs
  high_cor_pairs <- data.frame(
    Var1 = rownames(cor_matrix)[high_cor[, 1]],
    Var2 = colnames(cor_matrix)[high_cor[, 2]],
    Correlation = cor_matrix[high_cor]
  ) %>%
    filter(Var1 < Var2) %>%  # Remove duplicates (A-B and B-A)
    arrange(desc(abs(Correlation)))
  
  cat("High-correlation pairs found:\n")
  print(high_cor_pairs)
  
  cat("\n💡 RECOMMENDATION:\n")
  cat("Variables with |r| > 0.7 provide redundant information.\n")
  cat("Consider removing one variable from each pair.\n")
  cat("Choose which to keep based on:\n")
  cat("  1. Ecological relevance for wolves\n")
  cat("  2. Data quality and completeness\n")
  cat("  3. Ease of interpretation\n\n")
  
} else {
  cat("✅ No variable pairs with |r| > 0.7 found\n")
  cat("   All variables are sufficiently independent\n\n")
}
```

**Decision Matrix for Variable Selection:**

If **Elevation** and **Temperature** are highly correlated (r = -0.8):
- **Keep Elevation** because:
  - More direct ecological relevance (wolves select elevation)
  - More stable (climate can vary year-to-year)
  - Better spatial resolution (SRTM vs WorldClim)
- **Remove Temperature**

If **Road Density** and **Population** are highly correlated (r = 0.75):
- **Keep Population** because:
  - More comprehensive human footprint metric
  - Roads can exist in unpopulated areas (mountain passes)
  - Population better predicts wolf avoidance
- **Remove Road Density**

---

### Variable Selection Based on Correlation
```r
# =============================================================================
# SELECT FINAL VARIABLES FOR CNN
# =============================================================================

cat(rep("=", 70), "\n", sep = "")
cat("🎯 VARIABLE SELECTION FOR CNN\n")
cat(rep("=", 70), "\n\n", sep = "")

# Based on correlation analysis, select variables to keep
# Example selection (adjust based on your actual results):

selected_variables <- c(
  "NDVI",               # Vegetation productivity (prey habitat)
  "Roughness",          # Terrain complexity (cover, denning)
  "Water_Distance",     # Essential resource
  "Population",         # Human disturbance (comprehensive metric)
  "Road_Density"        # Linear barriers (if not too correlated with population)
)

cat("Selected variables:\n")
for (v in selected_variables) {
  cat("  ✅", v, "\n")
}
cat("\n")

# Variables removed (if any)
removed_variables <- setdiff(names(env_stack_scaled), selected_variables)

if (length(removed_variables) > 0) {
  cat("Removed variables (redundant or low importance):\n")
  for (v in removed_variables) {
    cat("  ❌", v, "\n")
  }
  cat("\n")
}

cat("Final variable count:", length(selected_variables), "\n")
cat("Reduction:", nlyr(env_stack_scaled) - length(selected_variables), 
    "variables removed\n\n")
```

**Rationale for Final Selection:**

**NDVI (✅ KEEP):**
- Unique information: Vegetation productivity
- Low correlation with other variables
- Direct link to prey availability
- High quality data (MODIS 250m)

**Roughness (✅ KEEP):**
- Unique information: Terrain complexity
- Weakly correlated with elevation (captures different aspect)
- Important for denning and hunting cover
- Not redundant with slope

**Water Distance (✅ KEEP):**
- Unique information: Resource availability
- Independent of other variables
- Essential for wolf physiology
- Clear biological importance

**Population (✅ KEEP):**
- Comprehensive human disturbance metric
- Strongest negative predictor in literature
- Captures multiple disturbance types
- High quality data (GHS-POP 100m)

**Road Density (✅ KEEP):**
- Linear barriers to movement
- Mortality risk (vehicle collisions)
- May be partially redundant with population
- Keep if correlation with population < 0.7

**Elevation (❌ REMOVE - Example):**
- Highly correlated with temperature (r = -0.85)
- Information captured by temperature
- Temperature more directly relevant

**Railway Density (❌ REMOVE - Example):**
- Highly correlated with road density (r = 0.82)
- Much lower density than roads
- Roads capture similar barrier effect

---

### Create Final CNN Input Stack
```r
# =============================================================================
# CREATE FINAL STANDARDIZED STACK FOR CNN
# =============================================================================

cat("📦 Creating final environmental stack for CNN...\n")

# Extract selected variables
env_stack_final <- env_stack_scaled[[selected_variables]]

# Verify selection
cat("\nFinal stack properties:\n")
cat("  Variables:", nlyr(env_stack_final), "\n")
cat("  Names:", paste(names(env_stack_final), collapse = ", "), "\n")
cat("  Resolution:", res(env_stack_final)[1], "m\n")
cat("  CRS:", crs(env_stack_final, describe = TRUE)$name, "\n\n")

# Save final stack
writeRaster(env_stack_final, "env_stack_final_cnn.tif", overwrite = TRUE)

cat("✅ Final CNN stack saved: env_stack_final_cnn.tif\n")
cat("   File size:", round(file.size("env_stack_final_cnn.tif") / 1e6, 1), "MB\n\n")

# Also save variable names for reproducibility
write.csv(
  data.frame(
    index = 1:length(selected_variables),
    variable = selected_variables
  ), 
  "cnn_variables_final.csv", 
  row.names = FALSE
)

cat("✅ Variable list saved: cnn_variables_final.csv\n\n")
```

### Results
[INSERT: Final 5-variable stack visualization panel]

**Final CNN input stack:**
- **Channels:** 5 variables
- **Resolution:** 100m × 100m
- **Study area:** ~[X] km²
- **Total pixels:** ~[X] million per variable
- **File size:** ~[X] MB
- **All variables standardized:** Mean ≈ 0, SD ≈ 1

**Advantages of reduced variable set:**
- ✅ Less redundancy (lower multicollinearity)
- ✅ Faster training (fewer channels)
- ✅ Better interpretability (clear role per variable)
- ✅ Reduced overfitting risk (simpler model)
- ✅ Maintained ecological coverage (all habitat aspects represented)

---

Would you like me to continue with:
11. Wolf Occurrence Data (GBIF Download)
12. Pseudo-Absence Sampling Strategy
13. CNN Patch Extraction
14. Train/Validation/Test Split

Let me know and I'll complete the markdown!
