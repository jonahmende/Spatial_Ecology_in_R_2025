# This is an example of code for the exam

## How to import external data in R

The packages needed in this script are the following:
```r
library(terra)   # package for mamaging raster and vector data
library(imageRy) # analysing RS data
```


In order to import data in R we should set the working directory:

```r
setwd("/Users/jonahmende/Library/Mobile Documents/com~apple~CloudDocs/Unibo/3. semestre/spatial ecology in r")
```

To check for the folder you can make use of:
```r 
getwd()
```

The import of the data is done by:
```r
group <- rast("DJI_20251128093951_0003_D.JPG")
```
To get info on the image, digit its name by:
```r
group
```

## Visualization of data

In order to plot the image, we will use the RGB scheme:
```r
# layer 1,2,3 = red, green, blue wavelength
im.plotRGB(group, r=1, g=2, b=3)
```

The image is flipped, so we can solve it with the flip() function:
```r
group <- flip(group)
```

The output plot can be exported with the png() function:
```r
png("group_photo.png")
im.plotRGB(group, r=1, g=2, b=3)
dev.off()
```
The output image looks like:

<img width="480" height="480" alt="group_photo" src="https://github.com/user-attachments/assets/29f636ef-ade9-4819-91d4-398c9734d490" />

Let's invert the bands to create a false color image:
```r
png("group_photo_false.png")
im.plotRGB(group, r=2, g=1, b=3)
dev.off()
```

The false color image is something like:

<img width="480" height="480" alt="group_photo_false" src="https://github.com/user-attachments/assets/a299f19e-5d38-42d7-b07a-72ce293cf3cf" />



