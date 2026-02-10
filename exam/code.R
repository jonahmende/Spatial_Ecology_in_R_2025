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


# --- 2. ADMINISTRATIVE BOUNDARIES & STUDY AREA ---

# Download Italy's administrative boundaries (Level 1 = Regions)
italy <- gadm(country = "ITA", level = 1, path = "map_data")

# Filter for specific northern and central regions where wolf presence is significant
regions <- italy[italy$NAME_1 %in% c("Emilia-Romagna", "Toscana", "Lombardia", 
                                     "Veneto", "Piemonte", "Trento", "Umbria", 
                                     "Marche", "Liguria"), ]

# Cast to MULTIPOLYGON for broader compatibility with 'sf' and 'terra' functions
target_sf <- st_as_sf(regions) %>% st_cast("MULTIPOLYGON")

# --- 3. ELEVATION DATA & SPATIAL TEMPLATE ---

# Retrieve Digital Elevation Model (DEM) data via 'elevatr'
# z = 7 provides an appropriate resolution for national-scale modeling
elevation_raw <- get_elev_raster(target_sf, z = 7, clip = "bbox")

# Convert to SpatRaster, crop to the study area extent, and mask to the region borders
elevation <- rast(elevation_raw) %>% 
  crop(regions) %>% 
  mask(regions)
names(elevation) <- "Elev"

# --- 4. PIXEL DIMENSION ANALYSIS (GEOGRAPHIC VS PROJECTED) ---

# Because geographic coordinates (lat/lon) represent distance differently 
# depending on latitude, we verify the physical size of a pixel in meters.
deg_res <- res(elevation) 
center_lat <- 43.0
center_lon <- 12.0

# Create points to measure East-West and North-South distance of one pixel
p_start <- vect(matrix(c(center_lon, center_lat), ncol=2), crs="EPSG:4326")
p_east  <- vect(matrix(c(center_lon + deg_res[1], center_lat), ncol=2), crs="EPSG:4326")
p_north <- vect(matrix(c(center_lon, center_lat + deg_res[2]), ncol=2), crs="EPSG:4326")

pixel_width  <- distance(p_start, p_east)
pixel_height <- distance(p_start, p_north)

cat("--- Raw Pixel Analysis (WGS84) ---\n")
cat("Width: ", round(pixel_width), "m | Height: ", round(pixel_height), "m\n")

# --- 5. STANDARDIZATION & PROJECTION ---

# PROJECTING TO UTM ZONE 32N: This is critical for CNNs. 
# It converts the grid to square pixels (500m x 500m), ensuring spatial consistency.
elevation <- project(elevation, "EPSG:32632", res = 500)
elevation <- mask(elevation, project(regions, "EPSG:32632"))

# --- 6. ENVIRONMENTAL VARIABLE ACQUISITION ---

study_area <- ext(elevation)

# Download and crop bioclimatic and landcover variables
raw_list <- list(
  Temp      = crop(worldclim_country("ITA", var = "bio", path = "map_data")[[1]], study_area),
  Footprint = crop(footprint(year = 2009, path = "map_data"), study_area),
  Trees     = crop(landcover(var = "trees", path = "map_data"), study_area),
  Shrubs    = crop(landcover(var = "shrubs", path = "map_data"), study_area),
  Urban     = crop(landcover(var = "built", path = "map_data"), study_area),
  Grass     = crop(landcover(var = "grassland", path = "map_data"), study_area)
)

# --- 7. TERRAIN DERIVATIVES & CUSTOM KERNEL ---

# Calculate standard terrain metrics
slope     <- terrain(elevation, v = "slope", unit = "degrees")
tpi       <- terrain(elevation, v = "TPI")
tri       <- terrain(elevation, v = "TRI") # Terrain Ruggedness Index

# ImageRy KERNEL FUNCTION: Processes rasters using a moving window (focal analysis)
# This is used here to calculate local variability (standard deviation) of elevation.
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
  
  # Combine outputs into a multi-layer SpatRaster
  res <- terra::rast(out)
  
  # Name output layers nicely
  names(res) <- names(out)
  
  # Plot all layers (explicit terra method)
  terra::plot(res)
  
  invisible(res)
}


# Apply kernel to derive Roughness (SD of elevation in a 3x3 window)
roughness <- im.kernel(elevation, mw = 3, stat = "sd")

# --- 8. MULTI-LAYER STACK ALIGNMENT ---

# Update regional vector to match the projected coordinate system
regions_utm <- project(regions, crs(elevation))

# Project and Resample all external layers to match the 500m UTM elevation grid
processed_list <- lapply(raw_list, function(l) {
  l_utm <- project(l, elevation, method = "bilinear")
  l_masked <- mask(l_utm, regions_utm)
  return(l_masked)
})

# Assemble the final environmental stack
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

# Rename layers for clear identification during model input
names(env_stack_og) <- c("Elev", "Slope", "TPI", "TRI", "Roughness", "Temp", 
                         "HumanFoot", "TreeCover", "ShrubCover", "UrbanPercent")

# --- 9. VISUALIZATION & EXPORT ---

# 1. Save the Multi-panel plot
png("Environmental_Stack_Multipanel.png", width = 2000, height = 2000, res = 180)

# We leave 'main' empty so 'terra' uses the individual layer names for each frame
plot(env_stack_og, 
     nc = 3, 
     col = viridis::viridis(100), 
     mar = c(3, 3, 4, 4)) # Increase top margin (3rd value) for the overall title

# Add one single "Super Title" at the top of the entire PNG
mtext("Environmental Covariates Stack (500m UTM)", side = 3, line = -1.5, outer = TRUE, cex = 1.5, font = 2)

dev.off()

# 2. Re-plot in the R console to verify
plot(env_stack_og, nc = 3, col = viridis::viridis(100))


# --- 10. VARIABLE SELECTION & MIN-MAX SCALING ---

# Drop TPI (Topographic Position Index) due to poor resolution at this scale
# We keep only variables with high ecological relevance for wolf presence
env_stack <- subset(env_stack_og, c("Elev", "Slope", "TRI", "Roughness", "Temp", 
                                    "HumanFoot", "TreeCover", "ShrubCover", "UrbanPercent"))

# CNNs perform best when data is within a standard range (0 to 1)
# 1. Calculate the global minimum and maximum for each layer
env_min <- minmax(env_stack)[1,]
env_max <- minmax(env_stack)[2,]

# 2. Apply the Scaling Formula: (Value - Min) / (Max - Min)
env_scaled <- (env_stack - env_min) / (env_max - env_min)

# Check for missing values created during scaling
cat("Number of NA values in scaled stack:", sum(is.na(values(env_scaled))), "\n")

# --- 11. COLLINEARITY ANALYSIS ---

# 1. Randomly sample 10,000 pixels to calculate correlations
# Sampling is faster than using the entire raster for statistical checks
set.seed(123) # Ensures reproducibility
sample_data <- spatSample(env_scaled, size = 10000, method = "random", na.rm = TRUE)

# 2. Generate and Save the Initial Correlation Matrix
cor_matrix <- cor(sample_data)

png("Correlation_Heatmap_Initial.png", width = 1200, height = 1200, res = 150)
corrplot(cor_matrix, 
         method = "color", 
         type = "upper", 
         order = "hclust", 
         addCoef.col = "black", 
         tl.col = "black", 
         tl.srt = 45, 
         diag = FALSE,
         title = "Initial Correlation Matrix",
         mar = c(0,0,1,0))
dev.off()

# 3. Drop Redundant Variables
# To prevent model overfitting and multi-collinearity issues (e.g., Elev vs Temp),
# we retain a core set of distinct environmental predictors.
env_scaled <- subset(env_scaled, c("Elev", "Roughness", "HumanFoot", "TreeCover", "ShrubCover"))

# 4. Verify the Cleaned Stack Correlation
cor_matrix_new <- cor(spatSample(env_scaled, size = 10000, na.rm = TRUE))

png("Correlation_Heatmap_Final.png", width = 1000, height = 1000, res = 150)
corrplot(cor_matrix_new, 
         method = "color", 
         type = "upper", 
         order = "hclust", 
         addCoef.col = "black", 
         tl.col = "black", 
         tl.srt = 45, 
         diag = FALSE,
         title = "Final Optimized Predictors",
         mar = c(0,0,1,0))
dev.off()

# --- 12. ENVIRONMENTAL COVARIATE EXPLORATION ---

# im.ridgeline generates a ggplot object showing data distribution
ridge_plot <- im.ridgeline(env_scaled, scale = 1.5, palette = "viridis") +
  labs(
    title = "Distribution Density of Scaled Environmental Variables",
    subtitle = "Standardized [0-1] range for CNN Input",
    x = "Scaled Value",
    y = "Variable"
  ) +
  theme_minimal()

# Display the plot
print(ridge_plot)

# Save the plot using ggsave (recommended for imageRy/ggplot outputs)
ggsave("Environmental_Ridgelines.png", plot = ridge_plot, width = 8, height = 6, dpi = 300)


# --- 13. SPECIES DATA ACQUISITION (GBIF) ---

# Fetch occurrence data for Canis lupus in specified Italian regions
wolf_obs <- occ_data(scientificName = "Canis lupus", hasCoordinate = TRUE, limit = 5000, 
                     country = "IT", stateProvince = c("Emilia-Romagna", "Toscana", "Lombardia", 
                                                       "Veneto", "Piemonte", "Trento", "Umbria", 
                                                       "Marche", "Liguria"))

# Combine results into a single data frame
pres_pts <- bind_rows(lapply(wolf_obs, function(x) x$data)) 

# Convert to SpatVector and project to match the environmental stack (UTM Zone 32N)
pres_vect <- vect(pres_pts, geom=c("decimalLongitude", "decimalLatitude"), crs="EPSG:4326")
pres_vect <- project(pres_vect, crs(env_scaled))


# --- 14. DATA CLEANING & SPATIAL THINNING ---

# 1. Remove points that fall outside the environmental raster (NA check)
extracted_vals <- terra::extract(env_scaled[[1]], pres_vect)
pres_vect_clean <- pres_vect[which(!is.na(extracted_vals[, 2])), ]


# 2. Spatial Thinning (8km grid)
# This prevents model bias toward over-sampled areas (e.g., near roads/cities)
thinning_grid <- rast(env_scaled)
res(thinning_grid) <- 8000  # 8km resolution
set.seed(123)
pres_final <- spatSample(pres_vect_clean, method = "random", strata = thinning_grid, size = 1)


# --- 15. PSEUDO-ABSENCE GENERATION (DONUT SAMPLING) ---

# 1. Create Buffer Zones (10km exclusion, 30km sampling universe)
# Aggregating prevents the code from processing hundreds of individual circles
pres_buffer_inner <- aggregate(buffer(pres_final, width = 10000)) 
pres_buffer_outer <- aggregate(buffer(pres_final, width = 30000)) 

# 2. Filter "Safe" cells for sampling (avoiding raster edges)
cells_with_data <- as.data.frame(env_scaled[[1]], xy = TRUE, na.rm = TRUE, cells = TRUE)
rc <- rowColFromCell(env_scaled, cells_with_data$cell)
valid_indices <- which(rc[,1] > 16 & rc[,1] < (nrow(env_scaled) - 16) & 
                       rc[,2] > 16 & rc[,2] < (ncol(env_scaled) - 16))
cells_safe_vect <- vect(cells_with_data[valid_indices, ], geom=c("x", "y"), crs=crs(env_scaled))

# 3. Apply Donut Logic: Points must be inside 30km but outside 10km of a presence
inside_outer <- is.related(cells_safe_vect, pres_buffer_outer, "intersects")
inside_inner <- is.related(cells_safe_vect, pres_buffer_inner, "intersects")
abs_coords <- as.data.frame(cells_safe_vect[inside_outer & !inside_inner, ], geom="XY")

# 4. Balanced Sampling: Match number of absences to number of presences
set.seed(123)
abs_final <- vect(abs_coords[sample(nrow(abs_coords), nrow(pres_final)), ], 
                  geom=c("x", "y"), crs = crs(env_scaled))

# --- 16. DATASET STRATIFICATION (TRAIN/VAL/TEST) ---

# Combine and label: Presence = 1, Absence = 0
pres_final$label <- 1
abs_final$label <- 0
all_pts <- rbind(pres_final[, "label"], abs_final[, "label"])

# Function for 70/15/15 Stratified Split
get_stratified_splits <- function(idx_vector) {
  set.seed(42)
  shuffled_idx <- sample(idx_vector)
  n <- length(shuffled_idx)
  train_end <- round(0.70 * n)
  val_end   <- round(0.85 * n)
  return(list(train = shuffled_idx[1:train_end], 
              val = shuffled_idx[(train_end + 1):val_end], 
              test = shuffled_idx[(val_end + 1):n]))
}

# Apply split logic
pres_split <- get_stratified_splits(which(all_pts$label == 1))
abs_split  <- get_stratified_splits(abs_indices <- which(all_pts$label == 0))

all_pts$split <- NA
all_pts$split[c(pres_split$train, abs_split$train)] <- "train"
all_pts$split[c(pres_split$val, abs_split$val)]     <- "val"
all_pts$split[c(pres_split$test, abs_split$test)]   <- "test"


# --- 17. CNN TILE EXTRACTION (4D ARRAY PREP) ---

extract_split_tiles <- function(pts_vector, split_label, stack, patch_size = 32) {
  subset_pts <- pts_vector[pts_vector$split == split_label, ]
  coords <- crds(subset_pts)
  labels <- subset_pts$label
  n_pts <- nrow(subset_pts)
  n_layers <- nlyr(stack)
  
  tiles <- array(0, dim = c(n_pts, patch_size, patch_size, n_layers))
  
  for(i in 1:n_pts) {
    cell <- cellFromXY(stack, coords[i, , drop=FALSE])
    rc   <- rowColFromCell(stack, cell)
    rows <- (rc[1]-15):(rc[1]+16)
    cols <- (rc[2]-15):(rc[2]+16)
    
    try({
      patch_vals <- stack[rows, cols, 1:n_layers]
      tiles[i,,,] <- array(as.matrix(patch_vals), dim = c(patch_size, patch_size, n_layers))
    }, silent = TRUE)
  }
  tiles[is.na(tiles)] <- 0
  return(list(x = tiles, y = as.numeric(labels)))
}

# Create final tensors
train_data <- extract_split_tiles(all_pts, "train", env_scaled)
val_data   <- extract_split_tiles(all_pts, "val", env_scaled)
test_data  <- extract_split_tiles(all_pts, "test", env_scaled)

# --- 18. VISUALIZATION OF SAMPLING STRATEGY ---

wolf_sampling_plot <- ggplot() +
  geom_spatvector(data = regions, fill = "gray98", color = "gray80") +
  geom_spatvector(data = pres_buffer_outer, aes(fill = "Sampling Zone (30km)"), alpha = 0.3, color = NA) +
  geom_spatvector(data = pres_buffer_inner, aes(fill = "Exclusion Zone (10km)"), fill = "white", alpha = 0.5, color = "red", linewidth = 0.2) +
  geom_spatvector(data = abs_final, aes(color = "Pseudo-Absence"), size = 0.8, alpha = 0.7) +
  geom_spatvector(data = pres_final, aes(color = "Presence"), size = 1.2, shape = 17) +
  scale_fill_manual(name = "Buffer Zones", values = c("Sampling Zone (30km)" = "lightblue")) +
  scale_color_manual(name = "Wolf Observations", values = c("Presence" = "red", "Pseudo-Absence" = "darkblue")) +
  annotation_scale(location = "bl", width_hint = 0.4) +
  annotation_north_arrow(location = "bl", which_north = "true", pad_x = unit(0.2, "in"), pad_y = unit(0.4, "in"), style = north_arrow_fancy_orienteering) +
  theme_bw() + 
  labs(title = "Refined Wolf Sampling Strategy", subtitle = "8km spatial thinning & 10-30km donut sampling")

# Save high-res output
ggsave("Wolf_Sampling_Map_Final.png", plot = wolf_sampling_plot, width = 12, height = 10, dpi = 300)


# --- 19. CNN ARCHITECTURE DEFINITION ---

# Input: 32x32 image patches with 5 channels (Elev, Roughness, HumanFoot, Tree, Shrub)
input <- layer_input(shape = c(32, 32, 5))

output <- input %>%
  # 1. Augmentation: Teaching the model that direction doesn't change habitat quality
  layer_random_flip("horizontal_and_vertical") %>%
  layer_random_rotation(factor = 0.1) %>% 
  
  # 2. Convolutional Block 1: Broad feature detection (edges, slopes)
  layer_conv_2d(filters = 8, kernel_size = c(3,3), activation = 'relu', padding = "same") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # 3. Convolutional Block 2: Complex patterns (fragmentation, forest density)
  layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = 'relu', padding = "same") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # 4. Feature Condensing: Global Average Pooling reduces overfitting risk
  layer_global_average_pooling_2d() %>%
  
  # 5. Output Head: Binary classification (Wolf vs. No Wolf)
  layer_dense(units = 32, activation = 'relu', kernel_regularizer = regularizer_l2(0.02)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

model <- keras_model(inputs = input, outputs = output)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.0005),
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

# --- 20. MODEL TRAINING ---

# Class Weights: Penalize missing a wolf more heavily than a false alarm
weights <- list("0" = 1.0, "1" = 2.0)

history <- model %>% fit(
  x_train_rand, y_train_rand,
  epochs = 100,
  batch_size = 32,
  validation_data = list(x_val_rand, y_val_rand),
  class_weight = weights,
  callbacks = list(callback_early_stopping(patience = 25, restore_best_weights = TRUE)),
  verbose = 1
)

# Generate the plot with straight lines (smooth = FALSE)
# This connects each epoch's data point directly to the next
history_plot <- plot(history, smooth = FALSE) + 
  theme_bw() + 
  labs(title = "CNN Training Metrics")

# Display it in your R session
print(history_plot)

# Save it as a high-resolution image
ggsave("CNN_Training_History_Straight.png", 
       plot = history_plot, 
       width = 9, 
       height = 6, 
       dpi = 300)

                             
# --- 21. EVALUATION & ROC CURVE ---

y_pred_test <- as.vector(model %>% predict(x_test_rand))
roc_obj <- roc(as.vector(y_test_rand), y_pred_test)

png("Model_Performance_ROC.png", width = 800, height = 800, res = 120)
plot(roc_obj, col = "#1c4587", lwd = 3, main = paste("ROC Curve (AUC:", round(auc(roc_obj), 3), ")"))
abline(a = 0, b = 1, lty = 2, col = "red")
dev.off()

# Determine the "best" threshold for core habitat definition
best_thresh <- coords(roc_obj, "best", ret = "threshold", transpose = FALSE)

# --- 22. PERMUTATION VARIABLE IMPORTANCE ---

get_importance <- function(model, x_test, y_test, var_names) {
  baseline <- as.numeric(model %>% evaluate(x_test, y_test, verbose = 0))[2]
  importance <- data.frame(Variable = var_names, Accuracy_Loss = 0)
  
  for(i in 1:length(var_names)) {
    x_perm <- x_test
    # Scramble the specific environmental channel across all samples
    flat_size <- dim(x_perm)[1] * 32 * 32
    x_perm[,,,i] <- sample(x_perm[,,,i], size = flat_size, replace = FALSE)
    
    perm_acc <- as.numeric(model %>% evaluate(x_perm, y_test, verbose = 0))[2]
    importance$Accuracy_Loss[i] <- baseline - perm_acc
  }
  return(importance)
}

imp_results <- get_importance(model, x_test_rand, y_test_rand, 
                              c("Elev", "Roughness", "HumanFoot", "TreeCover", "ShrubCover"))

# --- 23. SPATIAL PREDICTION (NATIONAL SCALE) ---

# Aggregate to 2km resolution for national mapping efficiency
env_low_res <- aggregate(env_scaled, fact = 4) 
wolf_map <- rast(env_low_res[[1]])
values(wolf_map) <- NA

# Moving window prediction loop
nr <- nrow(env_low_res); nc <- ncol(env_low_res)
message("Generating National Habitat Map...")

for (r in seq(17, nr - 16, by = 2)) {
  col_seq <- seq(17, nc - 16, by = 2)
  row_batch <- array(0, dim = c(length(col_seq), 32, 32, 5))
  
  for (i in seq_along(col_seq)) {
    c <- col_seq[i]
    patch <- as.matrix(env_low_res[(r-15):(r+16), (c-15):(c+16)])
    patch[is.na(patch)] <- 0
    row_batch[i,,,] <- array(patch, dim = c(32, 32, 5))
  }
  
  preds <- model %>% predict(row_batch, verbose = 0)
  wolf_map[r, col_seq] <- preds
}

# Interpolate missing pixels from the 'by=2' step
wolf_map_final <- focal(wolf_map, w = 3, fun = mean, na.policy = "only", na.rm = TRUE)

# --- 24. FINAL VISUALIZATION ---

final_vis <- ggplot() +
  geom_spatraster(data = wolf_map_final) +
  scale_fill_whitebox_c(palette = "muted", labels = scales::label_percent(), name = "Suitability") +
  geom_spatvector(data = project(regions, crs(wolf_map_final)), fill = NA, color = "black", linewidth = 0.3) +
  theme_minimal() +
  labs(title = "National Wolf Habitat Suitability (CNN)", 
       subtitle = paste("Core Habitat Threshold (Youden's J):", round(best_thresh$threshold, 3)))

ggsave("Final_Wolf_Habitat_Map.png", plot = final_vis, width = 10, height = 12, dpi = 300)
