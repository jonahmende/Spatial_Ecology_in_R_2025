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


