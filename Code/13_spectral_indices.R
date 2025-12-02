# Spectral Indices from satellite images

library(terra)
library(imageRy)

# Listing files
im.List()

# Importing data
m1992 <- im.import("matogrosso_l5_1992219_lrg.jpg")

# layer 1,2,3 = NIR, red, green
im.plotRGB(m1992, r=1, g=2, b=3)
im.plotRGB(m1992, r=2, g=1, b=3)
im.plotRGB(m1992, r=2, g=3, b=1)

# Exercise: import the newest image
m2006 <- im.import("matogrosso_ast_2006209_lrg.jpg")

im.plotRGB(m2006, r=1, g=2, b=3)

# DVI 1992
# 100 NIR
# 0 red
# dvi = NIR - red = 100

dvi1992 <- m1992[[1]] - m1992[[2]]
dvi2006 <- m2006[[1]] - m2006[[2]]

library(viridis)
par(mfrow=c(1,2))
plot(dvi1992, col=inferno(100))
plot(dvi2006, col=inferno(100))

# NDVI
ndvi1992 <- ( m1992[[1]] - m1992[[2]] ) / ( m1992[[1]] + m1992[[2]] )
ndvi2006 <- ( m2006[[1]] - m2006[[2]] ) / ( m2006[[1]] + m2006[[2]] )

# or
ndvi1992 <- im.ndvi(m1992, nir=1, red=2)
ndvi2006 <- im.ndvi(m2006, nir=1, red=2)


par(mfrow=c(1,2))
plot(ndvi1992, col=inferno(100))
plot(ndvi2006, col=inferno(100))
