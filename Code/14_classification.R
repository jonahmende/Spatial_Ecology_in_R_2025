# This code can be used to classify data

library(terra)
library(imageRy)
library(ggplot2)
library(patchwork)

m1992 <- im.import("matogrosso_l5_1992219_lrg.jpg")
m2006 <- im.import("matogrosso_ast_2006209_lrg.jpg")
plot(m1992)
plot(m2006)

# testing classification - k-means
sun <- im.import("Solar_Orbiter_s_first_views_of_the_Sun_pillars.jpg")
plot(sun)

sunc <- im.classify(sun, num_clusters=3)  # unsupervised classification

# Apply the classification process to the Mato Grosso
m1992c <- im.classify(m1992, num_clusters=2)
# class 1 = human related areas and water
# class 2 = forest

m2006c <- im.classify(m2006, num_clusters=2)
# class 1 = human related areas and water
# class 2 = forest

# calculating frequencies
f1992 <- freq(m1992c)

# Proportions f/tot
tot1992c <- ncell(m1992c)
prop1992 = f1992$count / tot1992c

# Percentages
perc1992 = prop1992 * 100
# 1992: human = 17%, forest = 83%

perc2006 = freq(m2006c) / ncell(m2006c) * 100
# 2006: human = 56%, forest = 43%

# let's implement a dataframe with three columns
class <- c("forest", "human")
perc1992 <- c(83, 17)
perc2006 <- c(43, 56)
tabout <- data.frame(class, perc1992, perc2006)

# using ggplot2 package for the final graph
p1 <- ggplot(tabout, aes(x=class, y=perc1992, color=class)) + geom_bar(stat="identity", fill="white") + ylim(c(0,100)) # + theme(legend=None)
p2 <- ggplot(tabout, aes(x=class, y=perc2006, color=class)) + geom_bar(stat="identity", fill="white") + ylim(c(0,100))
p1  + p2
p1 / p2


# im.fuzzy function  gives out a map for each cluster of the im.classify function showing propability to belong to a cluster on a continious scale
# e.g. im.fuzzy(m1992, num_clusters=2)
