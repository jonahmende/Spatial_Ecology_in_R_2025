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
