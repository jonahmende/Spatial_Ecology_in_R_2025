# install.packages("sdm")
# install.packages("terra")
# install.packages("viridis")

library(terra)
library(sdm)
library(viridis)

file <- system.file("external/species.shp", package="sdm")
# [1] "/usr/local/lib/R/site-library/sdm/external/species.shp"

rana <- vect(file) # creating a vector of the points
rana

rana$Occurrence

plot(rana)

# Assuming 'rana' is your SpatVector with points and attributes
# Extract coordinates using the geom() function
coordinates <- geom(rana)

# Convert the coordinates to a data frame
coordinates_df <- as.data.frame(coordinates)

# Extract the 'Occurrence' attribute from the SpatVector
occurrence_df <- as.data.frame(rana$Occurrence)

# Combine the coordinates and the occurrence data into one data frame
final_df <- cbind(coordinates_df, occurrence_df)

# Export the final data frame to a CSV file
write.csv(final_df, "coordinates_with_occurrence.csv", row.names = FALSE)

# View the first few rows of the final table (optional)
head(final_df)

# Add the attribute column (e.g., Occurrence) to the data frame
coordinates_df$Occurrence <- rana$Occurrence

# Export the data frame to a CSV file
write.csv(coordinates_df, "coordinates_with_occurrence.csv", row.names = FALSE)


# select all presences and absences
pres <- rana[rana$Occurrence==1]
abse <- rana[rana$Occurrence==0]  # or !=1

# plot the presences and absences with different colors
par(mfrow=c(1,2))
plot(pres, col='blue')
plot(abse, col='orange')

# Covariates
elev <- system.file("external/elevation.asc", package="sdm")
# [1] "/usr/local/lib/R/site-library/sdm/external/elevation.asc"

elevmap <- rast(elev)

# Exercise: change the colors of the elevation map by the colorRampPalette function
cl <- colorRampPalette(c("green","hotpink","mediumpurple"))(100)
plot(elevmap, col=cl)

# Exercise: plot the presences together with elevation map
points(pres, pch=19)

# Exercise: import temperature and plot presences vs. temperature
temp <- system.file("external/temperature.asc", package="sdm")

tempmap <- rast(temp)
plot(tempmap)
points(pres)

plot(tempmap, col=mako(100))

# Exercise: plot elevation and temperature with presences one beside the other
par(mfrow=c(1,2))
plot(elevmap, col=mako(100))
points(pres)
plot(tempmap, col=mako(100))
points(pres)

# precipitation
prec <- system.file("external/precipitation.asc", package="sdm")

precmap <- rast(prec)
points(pres)

# vegetation
vege <- system.file("external/vegetation.asc", package="sdm")
vegemap <- rast(vege)
plot(vegemap)
points(pres)

# Exercise: plot all the ancillary variable in a multiframe
par(mfrow=c(2,2))
plot(elevmap)
plot(tempmap)
plot(precmap)
plot(vegemap)

cova <- c(elevmap, tempmap, vegemap, precmap)
plot(cova)
plot(cova$temperature)

anci <- c(elevmap, tempmap, precmap, vegemap)
plot(anci)
plot(anci, col=magma(100))
