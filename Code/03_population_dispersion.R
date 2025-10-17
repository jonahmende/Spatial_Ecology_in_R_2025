# install.packages("sdm")
# install.packages("terra")

library(terra)
library(sdm)

file <- system.file("external/species.shp", package="sdm")
# [1] "/usr/local/lib/R/site-library/sdm/external/species.shp"

rana <- vect(file) # creating a vector of the points
rana

rana$Occurrence

# how to acces single coordinate data???


plot(rana)

# select all presences and absences
pres <- rana[rana$Occurrence==1]
abse <- rana[rana$Occurrence==0]  # or !=1

# plot the presences and absences with different colors
par(mfrow=c(1,2))
plot(pres, col='blue')
plot(abse, col='orange')
