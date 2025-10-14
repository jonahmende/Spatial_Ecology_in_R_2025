# install.packages("sdm")
# install.packages("terra")

library(sdm)
library(terra)

file <- system.file("external/species.shp", package="sdm")
# [1] "/usr/local/lib/R/site-library/sdm/external/species.shp"

rana <- vect(file) # creating a vector of the points
rana

rana$Occurrence

plot(rana)

# how to acces single coordinate data???
