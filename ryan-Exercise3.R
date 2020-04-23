# Exercises 4

# Libraries
library(tidyverse)
library(mosaic)
library(LICORS)  # for kmeans++
library(foreach)
library(mvtnorm)
library(ggplot2)
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(NbClust)

# Data
greenbuildings = read_csv('greenbuildings.csv')
wine = read_csv('wine.csv')
socialmarketing = read_csv('social_marketing.csv')


# 1) Predictive Model Building: Green Buildings
"
Your goals are:
  
to build the best predictive model possible for price; and
to use this model to quantify the average change in rental income per square foot (whether in absolute or percentage terms) associated with green certification, 
holding other features of the building constant.
You can choose whether to consider LEED and EnergyStar separately or to collapse them into a single green certified category. 
You can use any modeling approaches in your toolkit, and you should also feel free to define new variables based on combinations of existing ones. 
Just make sure to explain what youve done

Write a short report detailing your methods, modeling choice, and conclusions
"

# Does not include: CS_PropertyID, cluster, Rent, LEED, Energystar, cluster_rent
# Baseline Model: Linear Model, KNN Regression, Logistic Regression, Linear Probability Model
ggplot(greenbuildings) + 
  geom_point(aes(x=size, y=Rent, color=cluster))

lm_starter = lm(Rent ~ size + empl_gr + leasing_rate + stories + age + renovated + class_a + class_b + green_rating + net + amenities + cd_total_07 + hd_total07 + Precipitation + Gas_Costs + Electricity_Costs, data=greenbuildings)

# Feature Selection: Forward/Backward Selection, Stepwise Selection, AIC/BIC
# stepwise selection
# note that we start with a reasonable guess
lm_step = step(lm_starter, scope=~(.)^3)
# the scope statement says:
# "consider all pairwise interactions for everything in lm_medium (.),
# along with the other variables explicitly named that weren't in medium"
# what variables are included?
getCall(lm_step)
coef(lm_step)

# Performance
# Compare out of sample performance
rmse = function(y, yhat) {
  sqrt( mean( (y - yhat)^2 ) )
}
n = nrow(SaratogaHouses)
n_train = round(0.8*n)  # round to nearest integer
n_test = n - n_train
rmse_vals = do(100)*{
  
  # re-split into train and test cases with the same sample sizes
  train_cases = sample.int(n, n_train, replace=FALSE)
  test_cases = setdiff(1:n, train_cases)
  saratoga_train = SaratogaHouses[train_cases,]
  saratoga_test = SaratogaHouses[test_cases,]
  
  # Fit to the training data
  # use `update` to refit the same model with a different set of data
  lm1 = update(lm_medium, data=saratoga_train)
  lm2 = update(lm_step, data=saratoga_train)
  
  # Predictions out of sample
  yhat_test1 = predict(lm1, saratoga_test)
  yhat_test2 = predict(lm2, saratoga_test)
  
  c(rmse(saratoga_test$price, yhat_test1),
    rmse(saratoga_test$price, yhat_test2))
}


# 2) What Causes What? Planet Money Podcast
# Randomized trial (treatment and control)
# Correlation vs Causation (correlation = chance/trend. causation = direct influence)

# Starts at 8:45
# a) Why can’t I just get data from a few different cities and run the regression of “Crime” on “Police” to 
# understand how more cops in the streets affect crime? (“Crime” refers to some measure of crime rate and “Police” 
# measures the number of cops in a city.) 
# Cops put out for terrorism in washington DC not for crime, orange alert days street crime goes down (causal)
# Robbers hiding in room because of terror level
# Hypothesis Tourists less likely so fewer victims (used metro to see incomers and that did not change)
# Cities differ (population size)
# Confounding Variables 

# b) How were the researchers from UPenn able to isolate this effect? 
# Briefly describe their approach and discuss their result in the “Table 2” below, from the researchers paper.

# c) Why did they have to control for Metro ridership? What was that trying to capture?
# Because it did not change between the two groups. If there was a difference between these resluts, it would have an impact on the findings.

# d) Below I am showing you "Table 4" from the researchers paper. Just focus on the first column of the table. 
# Can you describe the model being estimated here? What is the conclusion?


# 3) Clustering & PCA
"
The data in wine.csv contains information on 11 chemical properties of 6500 different bottles of vinho verde wine from northern Portugal. 
In addition, two other variables about each wine are recorded: whether the wine is red or white
the quality of the wine, as judged on a 1-10 scale by a panel of certified wine snobs.
Run both PCA and a clustering algorithm of your choice on the 11 chemical properties 
(or suitable transformations thereof) and summarize your results. Which dimensionality reduction technique makes more sense to you for this data? 
Convince yourself (and me) that your chosen method is easily capable of distinguishing the reds from the whites, using only the unsupervised information 
contained in the data on chemical properties. Does this technique also seem capable of sorting the higher from the lower quality wines?
"
# Data
head(wine)
summary(wine)

# White Wine Average Rating
white = subset(wine, color == 'white')
summary(white)
mean(white$quality)
# Red Wine Average Rating
red = subset(wine, color == 'red')
summary(red)
mean(red$quality)

# K-Means Clustering (by rating and color)
# Plotting variables by color to see relationships
ggplot(wine) + geom_point(aes(x=color, y=volatile.acidity, color=color))
# fviz_cluster(k2, data = df)
# Sulphates & Chlorides & Total Sulfur Dioxide & Fixed/Volatile Acidity

# As the initial centroids are defined randomly,
# we define a seed for purposes of reprodutability
set.seed(123)
# Let's remove the column with the mammals' names, so it won't be used in the clustering
input <- wine[c(1:11)]
# Extract the centers and scales from the rescaled data (which are named attributes)
X = scale(input, center=TRUE, scale=TRUE)
# Center and scale the data
mu = attr(X,"scaled:center")
sigma = attr(X,"scaled:scale")
# Run k-means with 2 clusters and 25 starts
clust1 = kmeans(X, centers=2, nstart=25)
#cluster: A vector of integers (from 1:k) indicating the cluster to which each point is allocated.
#centers: A matrix of cluster centers.
#totss: The total sum of squares.
#withinss: Vector of within-cluster sum of squares, one component per cluster.
#tot.withinss: Total within-cluster sum of squares, i.e. sum(withinss).
#betweenss: The between-cluster sum of squares, i.e. $totss-tot.withinss$.
#size: The number of points in each cluster.
fviz_cluster(clust1, data = input)
# What are the clusters?
clust1$center[1,]*sigma + mu
clust1$center[2,]*sigma + mu
# Which wines are in which clusters?
which(clust1$cluster == 1)
which(clust1$cluster == 2)
# A few plots with cluster membership shown
# qplot is in the ggplot2 library
#qplot(Weight, Length, data=input, color=factor(clust1$cluster))
#qplot(Horsepower, CityMPG, data=input, color=factor(clust1$cluster))
# Using kmeans++ initialization
clust2 = kmeanspp(X, k=2, nstart=25)
clust2$center[1,]*sigma + mu
clust2$center[2,]*sigma + mu
# Which cars are in which clusters?
which(clust2$cluster == 1)
which(clust2$cluster == 2)
# Compare versus within-cluster average distances from the first run
clust1$withinss
clust2$withinss
sum(clust1$withinss)
sum(clust2$withinss)
clust1$tot.withinss
clust2$tot.withinss
clust1$betweenss
clust2$betweenss
# Determining Optimal Clusters
# Elbow Method
set.seed(123)
fviz_nbclust(input, kmeans, method = "wss")
# Average Silhouette
fviz_nbclust(input, kmeans, method = "silhouette")
# Gap Statistic Method
# compute gap statistic
set.seed(123)
gap_stat <- clusGap(input, FUN = kmeans, nstart = 25,
                    K.max = 10, B = 50)
# Print the result
print(gap_stat, method = "firstmax")
# Visualize
fviz_gap_stat(gap_stat)

# PCA (NCI60)
wine.pca <- prcomp(wine[c(1:11)], center = TRUE, scale = TRUE) # Excludes Quality and Color
summary(wine.pca)
# Standard deviation: This is simply the eigenvalues in our case since the data has been centered and scaled (standardized)
# Proportion of Variance: This is the amount of variance the component accounts for in the data, ie. PC1 accounts for >44% of total variance in the data alone!
# Cumulative Proportion: This is simply the accumulated amount of explained variance, ie. if we used the first 10 components we would be able to account for >95% of total variance in the data.
# Since an eigenvalues <1 would mean that the component actually explains less than a single explanatory variable we would like to discard those. (ex: keep first 3, EV of 4 is 0.98)

screeplot(wine.pca, type = "l", npcs = 15, main = "Screeplot of the first 10 PCs")
abline(h = 1, col="red", lty=5)
legend("topright", legend=c("Eigenvalue = 1"),
       col=c("red"), lty=5, cex=0.6)

cumpro <- cumsum(wine.pca$sdev^2 / sum(wine.pca$sdev^2))
plot(cumpro[0:15], xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(v = 3, col="blue", lty=5)
abline(h = 0.6436, col="blue", lty=5)
legend("topleft", legend=c("Cut-off @ PC3"),
       col=c("blue"), lty=5, cex=0.6)

# We notice is that the first 3 components has an Eigenvalue >1 and explains almost 65% of variance
# We can effectively reduce dimensionality from 11 to 3 while “losing” about 35% of variance

# Let's plot these 3 components
# 1 & 2
plot(wine.pca$x[,1],wine.pca$x[,2], xlab="PC1 (27.54%)", ylab = "PC2 (22.67%)", main = "PC1 / PC2 - plot")
# 2 & 3
plot(wine.pca$x[,2],wine.pca$x[,3], xlab="PC2 (22.67%)", ylab = "PC3 (14.15%)", main = "PC2 / PC3 - plot")
# 1 & 3
plot(wine.pca$x[,1],wine.pca$x[,3], xlab="PC1 (27.54%)", ylab = "PC3 (14.15%)", main = "PC1 / PC3 - plot")

# We want to explain difference between red and white wine
# Let’s  add the response variable (color) to the plot and see if we can make better sense of it
library(factoextra)
fviz_pca_ind(wine.pca, geom.ind = "point", pointshape = 21, 
             pointsize = 2, 
             fill.ind = wine$color, 
             col.ind = "black", 
             palette = "jco", 
             addEllipses = TRUE,
             label = "var",
             col.var = "black",
             repel = TRUE,
             legend.title = "Color") +
  ggtitle("2D PCA-plot from 11 feature dataset") +
  theme(plot.title = element_text(hjust = 0.5))
# With just the first two components we can clearly see some separation between the white and red wines 


# 4) Market Segmentation: NutrientH20 & Twitter

# They collected every Twitter post ("tweet") by each of their followers over a seven-day period in June 2014. 
# Each tweet was categorized based on its content using a pre-specified scheme of 36 different categories, each representing a broad area of interest (e.g. politics, sports, family, etc.)
# Annotators were allowed to classify a post as belonging to more than one category.
# Each Row is a user
# Two interests of note here are "spam" (i.e. unsolicited advertising) and "adult" (posts that are pornographic or otherwise explicit)
# There's also an "uncategorized" label. (avoided as much as possible, same with chatter) 
# Most bots filtered out but some remain
# there is some inevitable error and noisiness in the annotation process.

# Your task to is analyze this data as you see fit, and to prepare a (short!) report for NutrientH20 that identifies any interesting market segments that appear to stand out in their social-media audience. 
# You have complete freedom in deciding how to pre-process the data and how to define "market segment." (Is it a group of correlated interests? A cluster? A principal component? Etc.)
# Clusters? Principal Components? (NBC)

# K-Means
# As the initial centroids are defined randomly,
# we define a seed for purposes of reprodutability
set.seed(123)
input <- social_marketing[c(2:37)]
# Extract the centers and scales from the rescaled data (which are named attributes)
X = scale(input, center=TRUE, scale=TRUE)
# Center and scale the data
mu = attr(X,"scaled:center")
sigma = attr(X,"scaled:scale")
# Run k-means with 2 clusters and 25 starts
clust1 = kmeans(X, centers=2, nstart=25)
fviz_cluster(clust1, data = input)
# What are the clusters?
clust1$center[1,]*sigma + mu
clust1$center[2,]*sigma + mu
# Which wines are in which clusters?
which(clust1$cluster == 1)
which(clust1$cluster == 2)
# A few plots with cluster membership shown
# qplot is in the ggplot2 library
#qplot(Weight, Length, data=input, color=factor(clust1$cluster))
#qplot(Horsepower, CityMPG, data=input, color=factor(clust1$cluster))
# Using kmeans++ initialization
clust2 = kmeanspp(X, k=2, nstart=25)
clust2$center[1,]*sigma + mu
clust2$center[2,]*sigma + mu
# Which cars are in which clusters?
which(clust2$cluster == 1)
which(clust2$cluster == 2)
# Compare versus within-cluster average distances from the first run
clust1$withinss
clust2$withinss
sum(clust1$withinss)
sum(clust2$withinss)
clust1$tot.withinss
clust2$tot.withinss
clust1$betweenss
clust2$betweenss
# Determining Optimal Clusters
# Elbow Method
set.seed(123)
fviz_nbclust(input, kmeans, k.max=50, method = "wss")
# Average Silhouette
fviz_nbclust(input, kmean, k.max=50, method = "silhouette")
# Gap Statistic Method
# compute gap statistic
set.seed(123)
gap_stat <- clusGap(input, FUN = kmeans, nstart = 25,
                    K.max = 50, B = 50)
# Print the result
print(gap_stat, method = "firstmax")
# Visualize
fviz_gap_stat(gap_stat)

# PCA (Probably Better) Congress109?
countdata = social_marketing[c(2:37)]
# First normalize phrase counts to phrase frequencies.
# (often a sensible first step for count data, before z-scoring)
Z = countdata/rowSums(countdata)
# PCA
pc2 = prcomp(Z, scale=TRUE, rank=2)
loadings = pc2$rotation
scores = pc2$x
# Question 1: where do the observations land in PC space?
# a biplot shows the first two PCs
qplot(scores[,1], scores[,2], color=social_marketing$X, xlab='Component 1', ylab='Component 2')
# Confusingly, the default color mapping has Democrats as red and republicans as blue.  This might be confusing, so let's fix that:
qplot(scores[,1], scores[,2], color=social_marketing$X, xlab='Component 1', ylab='Component 2') + scale_color_manual(values=c("blue", "grey", "red"))
# Interpretation: the first PC axis primarily has Republicans as positive numbers and Democrats as negative numbers
# Question 2: how are the individual PCs loaded on the original variables?
# The top words associated with each component
o1 = order(loadings[,1], decreasing=TRUE)
colnames(Z)[head(o1,25)]
colnames(Z)[tail(o1,25)]

o2 = order(loadings[,2], decreasing=TRUE)
colnames(Z)[head(o2,25)]
colnames(Z)[tail(o2,25)]

# Choosing K
# Elbow
library(foreach)
cars = read.csv('../data/cars.csv')
cars = scale(cars[,10:18]) # cluster on measurables
k_grid = seq(2, 20, by=1)
SSE_grid = foreach(k = k_grid, .combine='c') %do% {
  cluster_k = kmeans(cars, k, nstart=50)
  cluster_k$tot.withinss
}
plot(k_grid, SSE_grid)
# CH INdex
N = nrow(cars)
CH_grid = foreach(k = k_grid, .combine='c') %do% {
  cluster_k = kmeans(cars, k, nstart=50)
  W = cluster_k$tot.withinss
  B = cluster_k$betweenss
  CH = (B/W)*((N-k)/(k-1))
  CH
}
plot(k_grid, CH_grid)
# Gap
library(cluster)
cars_gap = clusGap(cars, FUN = kmeans, nstart = 50, K.max = 10, B = 100)
plot(cars_gap)
cars_gap





