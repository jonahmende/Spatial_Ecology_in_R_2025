## 1. Project Initialization and Dependency Management

The first step in any spatial deep learning workflow is establishing a clean environment. This involves setting the working directory to ensure file paths remain consistent and loading a specific suite of libraries designed to handle the complex transition from **geographic rasters** to **neural network tensors**.

### Core Components:

* **Spatial Engine**: `terra` and `sf` handle the heavy lifting for raster and vector processing.
* **Data Acquisition**: `geodata`, `elevatr`, and `rgbif` allow for programmatic access to administrative, topographic, and biodiversity databases.
* **Deep Learning**: `keras3` provides the interface to TensorFlow, while `abind` is used to restructure spatial patches into 4D tensors.
* **Cartography**: `ggplot2`, `tidyterra`, and `ggspatial` ensure that model outputs are visualized with professional map elements (scale bars, north arrows).

```r
# --- 1. SETUP & ENVIRONMENT ---

# Set the working directory to the project folder
setwd("/Users/jonahmende/Library/Mobile Documents/com~apple~CloudDocs/Unibo/3. semestre/spatial ecology in r")

# Define the required packages for spatial analysis, data acquisition, and Deep Learning
libs <- c(
  "geodata",      # For administrative boundaries and climate data
  "elevatr",      # Access to global elevation datasets
  "terra",        # Primary engine for raster data manipulation
  "sf",           # Handling of vector (shapefile) data
  "rgbif",        # Interface for Global Biodiversity Information Facility
  "caret",        # Machine learning tools for data splitting
  "keras3",       # Interface for TensorFlow/CNN modeling
  "corrplot",     # Visualizing variable correlations
  "dplyr",        # Data manipulation and piping
  "abind",        # Combining multi-dimensional arrays (tensors)
  "ggplot2",      # Core plotting library
  "tidyterra",    # Integration of terra objects with ggplot2
  "ggspatial",    # Adding scale bars and north arrows to maps
  "pROC",         # Evaluating model performance via ROC/AUC
  "imageRy"       # Specialized remote sensing visualization
)

# Load all libraries; character.only ensures the list is read correctly
lapply(libs, require, character.only = TRUE)

# Create a dedicated directory for storing downloaded spatial layers
dir.create("map_data", showWarnings = FALSE)


```

## 2. Defining the Study Area: Administrative Boundaries

Once the environment is ready, we define the geographic scope of our analysis. Using the **Global Administrative Areas (GADM)** database, we isolate the specific regions of Italy that form the core of our study.

Instead of modeling the entire peninsula, this script focuses on the **Northern and Central regions**. These areas are characterized by the rugged Apennine and Alpine terrains—habitats where wolf populations have seen significant recovery. By casting the data into a `MULTIPOLYGON` format via the `sf` package, we ensure the geometry is clean and compatible with subsequent spatial clipping and extraction tasks.

```r
# --- 2. ADMINISTRATIVE BOUNDARIES & STUDY AREA ---

# Download Italy's administrative boundaries (Level 1 = Regions)
italy <- gadm(country = "ITA", level = 1, path = "map_data")

# Filter for specific northern and central regions where wolf presence is significant
regions <- italy[italy$NAME_1 %in% c("Emilia-Romagna", "Toscana", "Lombardia", 
                                     "Veneto", "Piemonte", "Trento", "Umbria", 
                                     "Marche", "Liguria"), ]

# Cast to MULTIPOLYGON for broader compatibility with 'sf' and 'terra' functions
# This step ensures geometry consistency for spatial operations
target_sf <- st_as_sf(regions) %>% st_cast("MULTIPOLYGON")


```

## 3. Topographic Foundation: Elevation Data

The first environmental predictor for our model is a **Digital Elevation Model (DEM)**. Elevation is a critical factor for wolf habitat selection, as it often correlates with prey availability, snow cover, and distance from human disturbance.

We use the `elevatr` package to programmatically retrieve terrain data. A zoom level of **z = 7** is chosen to balance detail with computational efficiency, providing a resolution suitable for national-scale habitat modeling. After retrieval, the raw raster is transformed into a `SpatRaster` object, then meticulously **cropped** and **masked** to match the precise borders of our target Italian regions. This ensures that our model only considers data within the actual study area.

```r
# --- 3. ELEVATION DATA & SPATIAL TEMPLATE ---

# Retrieve Digital Elevation Model (DEM) data via 'elevatr'
# z = 7 provides an appropriate resolution for national-scale modeling (approx. 600m-1km)
elevation_raw <- get_elev_raster(target_sf, z = 7, clip = "bbox")

# Convert to SpatRaster, crop to the study area extent, and mask to the region borders
# Masking ensures values outside the administrative boundaries are set to NA
elevation <- rast(elevation_raw) %>% 
  crop(regions) %>% 
  mask(regions)

# Assign a clean name for the model's feature engineering phase
names(elevation) <- "Elev"

```

## 4. Geographic Fidelity: Pixel Dimension Analysis

When working with geographic coordinate systems (WGS84, EPSG:4326), units are expressed in **decimal degrees** rather than meters. However, because the Earth is an ellipsoid, the physical distance of one degree of longitude shrinks as you move from the equator toward the poles.

Before proceeding with spatial modeling, it is vital to understand the **real-world footprint** of our data. In this step, we calculate the physical dimensions of a single pixel at the center of our study area (latitude ~43°N). This verification confirms that each pixel represents roughly **389m x 532m**. Understanding this "rectangularity" is crucial when the CNN looks at a 32x32 patch, as it tells us the model is analyzing an area of approximately **12.5km x 17km**.

```r
# --- 4. PIXEL DIMENSION ANALYSIS (GEOGRAPHIC VS PROJECTED) ---

# Because geographic coordinates (lat/lon) represent distance differently 
# depending on latitude, we verify the physical size of a pixel in meters.
deg_res <- res(elevation) 
center_lat <- 43.0
center_lon <- 12.0

# Create points to measure East-West and North-South distance of one pixel
# This identifies the "real-world" resolution of our grid
p_start <- vect(matrix(c(center_lon, center_lat), ncol=2), crs="EPSG:4326")
p_east  <- vect(matrix(c(center_lon + deg_res[1], center_lat), ncol=2), crs="EPSG:4326")
p_north <- vect(matrix(c(center_lon, center_lat + deg_res[2]), ncol=2), crs="EPSG:4326")

pixel_width  <- distance(p_start, p_east)
pixel_height <- distance(p_start, p_north)

# Print results to the console for verification
cat("--- Raw Pixel Analysis (WGS84) ---\n")
# Output: --- Raw Pixel Analysis (WGS84) ---
cat("Width: ", round(pixel_width), "m | Height: ", round(pixel_height), "m\n")
# Output: Width: 389 m | Height: 532 m

```


## 5. Spatial Standardization: UTM Projection

For deep learning models like CNNs, the geometry of the input data is as important as the values themselves. In the previous step, we discovered that our raw pixels were rectangular (**389m x 532m**). If fed directly into a CNN, the model would perceive a "stretched" version of the landscape, potentially misinterpreting slopes and spatial patterns.

To correct this, we project the elevation data into **UTM Zone 32N (EPSG:32632)**. This coordinate reference system uses meters instead of degrees, allowing us to enforce a perfectly square pixel resolution of **500m x 500m**. This standardization ensures that a 3x3 convolutional kernel covers the exact same physical area regardless of where it slides across the map of Italy.

```r
# --- 5. STANDARDIZATION & PROJECTION ---

# PROJECTING TO UTM ZONE 32N: This is critical for CNNs. 
# It converts the grid to square pixels (500m x 500m), ensuring spatial consistency.
# This prevents the CNN from perceiving a "stretched" version of the terrain.
elevation <- project(elevation, "EPSG:32632", res = 500)

# Re-mask the projected raster to the projected region borders
# This cleans up the edges after the coordinate transformation
elevation <- mask(elevation, project(regions, "EPSG:32632"))

```


## 6. Multi-Source Environmental Data Acquisition

A robust Habitat Suitability Model requires more than just topography; it needs a multi-dimensional view of the ecosystem. In this stage, we programmatically pull data from several global repositories to build a comprehensive set of predictors.

By using the `elevation` raster as a spatial template (via `ext(elevation)`), we ensure that all incoming layers—ranging from climate to human impact—are immediately cropped to our specific study area in Italy.

### The Predictor Suite:

* **Climate (`Temp`)**: Sourced from WorldClim, representing Annual Mean Temperature.
* **Human Impact (`Footprint`)**: The Global Human Footprint index, measuring the cumulative toll of infrastructure, population density, and industry.
* **Land Cover (`Trees`, `Shrubs`, `Urban`, `Grass`)**: High-resolution layers defining the physical vegetation and built environment, which influence both wolf concealment and prey distribution.

```r
# --- 6. ENVIRONMENTAL VARIABLE ACQUISITION ---

# Use the existing elevation raster to define the spatial extent for all other variables
study_area <- ext(elevation)

# Download and crop bioclimatic and landcover variables
# We store them in a list for organized processing in the next steps
raw_list <- list(
  Temp      = crop(worldclim_country("ITA", var = "bio", path = "map_data")[[1]], study_area),
  Footprint = crop(footprint(year = 2009, path = "map_data"), study_area),
  Trees     = crop(landcover(var = "trees", path = "map_data"), study_area),
  Shrubs    = crop(landcover(var = "shrubs", path = "map_data"), study_area),
  Urban     = crop(landcover(var = "built", path = "map_data"), study_area),
  Grass     = crop(landcover(var = "grassland", path = "map_data"), study_area)
)

```

## 7. Advanced Terrain Modeling: Derivatives & Focal Analysis

To a wolf, the landscape is defined by more than just altitude; it is defined by the **complexity** of the terrain. Steeper slopes offer vantage points and protection, while high ruggedness provides cover and denning sites.

In this section, we derive three standard metrics from our DEM: **Slope**, **Topographic Position Index (TPI)**, and the **Terrain Ruggedness Index (TRI)**. Additionally, we use a custom focal analysis function (a "kernel") to calculate the local standard deviation of elevation—a precise measure of **Roughness**. By sliding a  moving window across the map, the model can identify areas where the terrain changes abruptly, capturing the fine-scale "texture" of the Italian mountains.

```r
# --- 7. TERRAIN DERIVATIVES & CUSTOM KERNEL ---

# Calculate standard terrain metrics using the 'terra' engine
slope     <- terrain(elevation, v = "slope", unit = "degrees")
tpi       <- terrain(elevation, v = "TPI")
tri       <- terrain(elevation, v = "TRI") # Terrain Ruggedness Index

# ImageRy KERNEL FUNCTION: 
# This function applies a moving window (focal analysis) to a raster.
# It summarizes the neighborhood of each pixel using statistical metrics 
# (mean, sd, etc.) to highlight local spatial patterns.

im.kernel <- function(input_image, mw = 3,
                      stat = c("mean", "median", "sd", "var", "cv", "range")) {
  
  if (!inherits(input_image, "SpatRaster")) {
    stop("input_image should be a SpatRaster object.")
  }
  if (terra::nlyr(input_image) != 1) {
    stop("input_image must have a single layer (nlyr == 1).")
  }
  
  if (!inherits(mw, "numeric") || length(mw) != 1 || is.na(mw)) {
    stop("mw must be a single numeric value.")
  }
  mw <- as.integer(mw)
  if (mw < 1 || (mw %% 2) == 0) {
    stop("mw must be a positive odd integer (e.g., 3, 5, 7, ...).")
  }
  
  stat <- match.arg(
    stat,
    choices = c("mean", "median", "sd", "var", "cv", "range"),
    several.ok = TRUE
  )
  
  fun_range <- function(v) {
    if (all(is.na(v))) return(NA_real_)
    rng <- range(v, na.rm = TRUE)
    rng[2] - rng[1]
  }
  
  fun_cv <- function(v) {
    if (all(is.na(v))) return(NA_real_)
    m <- mean(v, na.rm = TRUE)
    if (is.na(m) || m == 0) return(NA_real_)
    stats::sd(v, na.rm = TRUE) / m
  }
  
  out <- list()
  
  if ("mean" %in% stat) {
    out[["mean"]] <- terra::focal(input_image, w = mw, fun = mean, na.rm = TRUE)
  }
  if ("median" %in% stat) {
    out[["median"]] <- terra::focal(input_image, w = mw, fun = stats::median, na.rm = TRUE)
  }
  if ("sd" %in% stat) {
    out[["sd"]] <- terra::focal(input_image, w = mw, fun = stats::sd, na.rm = TRUE)
  }
  if ("var" %in% stat) {
    out[["var"]] <- terra::focal(input_image, w = mw, fun = stats::var, na.rm = TRUE)
  }
  if ("cv" %in% stat) {
    out[["cv"]] <- terra::focal(input_image, w = mw, fun = fun_cv)
  }
  if ("range" %in% stat) {
    out[["range"]] <- terra::focal(input_image, w = mw, fun = fun_range)
  }
  
  res <- terra::rast(out)
  names(res) <- names(out)
  terra::plot(res)
  invisible(res)
}

# Apply the kernel to derive Roughness (Standard Deviation of elevation in a 3x3 window)
# This provides a fine-grained look at terrain variability
roughness <- im.kernel(elevation, mw = 3, stat = "sd")

```


## 8. Multi-Layer Stack Alignment & Harmonization

In the world of spatial modeling, "garbage in, garbage out" often stems from misaligned data. Because our environmental variables originate from different sources (WorldClim, Global Human Footprint, and ESA Land Cover), they arrive with different resolutions, coordinate systems, and extents.

This section is the "harmonization" phase. We use the **500m UTM elevation grid** as the gold standard. Every other layer—from temperature to tree cover—is projected and resampled using **bilinear interpolation** to match this exact grid. Finally, we bundle these layers into a single `SpatRaster` object: the **Environmental Stack**. This stack acts as the multi-channel "image" that our CNN will eventually "see."

```r
# --- 8. MULTI-LAYER STACK ALIGNMENT ---

# Update regional vector to match the projected coordinate system (UTM Zone 32N)
regions_utm <- project(regions, crs(elevation))

# Project and Resample all external layers to match the 500m UTM elevation grid
# 'method = bilinear' is used for continuous variables like temperature and footprint
processed_list <- lapply(raw_list, function(l) {
  l_utm <- project(l, elevation, method = "bilinear")
  l_masked <- mask(l_utm, regions_utm)
  return(l_masked)
})

# Assemble the final environmental stack
# This combines topography, climate, human impact, and land cover into one object
env_stack_og <- c(
  elevation, 
  slope, 
  tpi,
  tri,
  roughness,
  processed_list$Temp, 
  processed_list$Footprint, 
  processed_list$Trees,
  processed_list$Shrubs,
  processed_list$Urban
)

# Rename layers for clear identification during feature importance analysis
names(env_stack_og) <- c("Elev", "Slope", "TPI", "TRI", "Roughness", "Temp", 
                         "HumanFoot", "TreeCover", "ShrubCover", "UrbanPercent")

```


## 9. Spatial Visualization & Quality Assurance

Before committing the data to the deep learning model, it is essential to visualize the entire environmental stack. This step serves as a final quality control check, ensuring that all layers are correctly aligned, masked to the Italian study area, and free of spatial artifacts.

We export this visualization as a high-resolution multi-panel plot. By using the `viridis` color palette, we ensure that the environmental gradients—from the high altitudes of the Alps to the density of the human footprint—are easily distinguishable and perceptually uniform. The final output provides a comprehensive "geographic atlas" of the 10 predictors that will inform our wolf habitat suitability analysis.

```r
# --- 9. VISUALIZATION & EXPORT ---

# 1. Save the Multi-panel plot to a high-resolution PNG
# The large dimensions (2000x2000) allow for detailed inspection of each layer
png("Environmental_Stack_Multipanel.png", width = 2000, height = 2000, res = 180)

# We leave 'main' empty so 'terra' uses the individual layer names for each frame
# 'nc = 3' organizes the 10 layers into a 3-column grid
plot(env_stack_og, 
     nc = 3, 
     col = viridis::viridis(100), 
     mar = c(3, 3, 4, 4)) # Adjusted margins for the 'Super Title'

# Add one single "Super Title" at the top of the entire PNG file
mtext("Environmental Covariates Stack (500m UTM)", 
      side = 3, line = -1.5, outer = TRUE, cex = 1.5, font = 2)

dev.off()

# 2. Re-plot in the R console to verify the data within the active session
plot(env_stack_og, nc = 3, col = viridis::viridis(100))

```











