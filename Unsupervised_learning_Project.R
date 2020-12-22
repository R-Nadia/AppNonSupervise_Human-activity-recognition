################################
### IMPORT LIBRARIES & DATA
################################
library(ggplot2)
library(Rtsne)
library(plyr)
library(FactoMineR)
library(factoextra)
library(gridExtra)
library(sparcl)
library(summarytools)
library(kohonen)
###Loading data 
data<-read.csv("http://allousame.free.fr/mlds/tp/data2.txt", header = FALSE, sep = " ")
View(data)
labels <-read.csv("http://allousame.free.fr/mlds/tp/labels.txt", header=FALSE)
id <-read.csv("http://allousame.free.fr/mlds/tp/id.txt", head=FALSE)
features <-read.csv("C:/Users/NADIA/Desktop/M2 MLDS/02-Apprentissage non supervisé/Projet/features.txt", head=FALSE, sep=" ")
# ----------------------------------
################################
### PREPROCESSING DATA
################################
data <- data[,-1]
#### Add column features 
dim(features) # 561 * 2
f = t(features)
dim(f) # 2 * 561
#colnames(data) <- f[2,]
data<-cbind(id,labels,data)
names(data)[1] <-"ID"
names(data)[2] <-"labels"
colnames(data)
View(data)

data$labels<- as.factor(data$labels)
data$labels<-revalue(data$labels, c("1"="WALKING", "2"="WALKING_UPSTAIRS", "3"="WALKING_DOWNSTAIRS", "4"="SITTING",
                                    "5"="STANDING","6"="LAYING"))
#### changing features names
colnames(data) = gsub(",", "", colnames(data))
colnames(data)=gsub("-", "", colnames(data))
colnames(data)=gsub("\\(","", colnames(data))
colnames(data)=gsub("\\)","", colnames(data))

View(data)
# ----------------------------------
################################
### EXPLORATERY DATA ANALYSIS
################################
dim(data)
summary(data)
str(data)
colnames(data)
# ----------------------------------
#Plot 01
#dev.off()
ggplot(data) +
  geom_bar(
    mapping = aes(x = ID, fill = labels),
    position = "dodge"
  )
# ----------------------------------
#Plot 02
ggplot(data) +
  geom_bar(mapping = aes(x = labels, fill = labels)) +
  theme(axis.text.x = element_text(angle = 90), legend.position = "none")+
  labs(x="ActivityLabels")
# ----------------------------------
#Plot 03
ggplot(data, aes(x=V201, color=labels), ) +
  geom_density()+
  theme(plot.title = element_text(hjust = 0.5)
  )+
  labs(title="Acceleration magnitude mean",x="ActivityLabels", y = "tBodyAccMagmean")
# ----------------------------------
#Plot 04
ggplot(data, aes(x = labels, y = V201, fill=labels)) + 
  geom_boxplot(outlier.colour="red", outlier.shape=1,
               outlier.size=1)+
  theme(plot.title = element_text(hjust = 0.5),legend.position = "none",
        axis.text.x = element_text(angle = 90)
  )+
  labs(title="Acceleration magnitude mean",x="ActivityLabels", y = "tBodyAccMagmean")
# ----------------------------------
#Plot 05
ggplot(data, aes(x = labels, y = V559, fill=labels)) + 
  geom_boxplot(outlier.colour="red", outlier.shape=1,
               outlier.size=1)+
  theme(plot.title = element_text(hjust = 0.5),legend.position = "none",
        axis.text.x = element_text(angle = 90)
  )+
  labs(title="Angle between X-axis and Gravity_mean",x="ActivityLabels", y = "AngleXgravityMean")
# ----------------------------------
# TSNE vizualisation
# Executing the algorithm on curated data
tsne <- Rtsne(data[3:563], dims = 2, perplexity=50, verbose=TRUE, max_iter = 1000)
exeTimeTsne<- system.time(Rtsne(data[3:563], dims = 2, perplexity=50, verbose=TRUE, max_iter = 1000))
# Plotting
tsne_plot <- data.frame(x = tsne$Y[,1], y = tsne$Y[,2], col = data$labels)
ggplot(tsne_plot) + geom_point(aes(x=x, y=y, color=col))

# ----------------------------------
################################
### MACHINE LEARNING MODELING 
################################
### MODELING WITH PCA
################################
# The PCA to understand data in reduced dimentions
pca <- prcomp(data[3:563], center=TRUE, scale=TRUE)
# Get the Variance
pca_var <- pca$sdev^2
pca_pvar <- pca_var/sum(pca_var)
# ----------------------------------
# Plot Cummulative of PCA to Check important componant/variables
plot(cumsum(pca_pvar),xlab="PCA"
     , ylab="Cumulative Variance"
     ,type='b'
     ,main="PCA Analysis of Data",col="green")
# We're going to take 25 components with 80% of variance
plot(cumsum(pca_pvar),xlab="PCA"
     , ylab="Cumulative Variance"
     ,type='b'
     ,main="PCA Analysis of Data",col="green") +
  abline(h=0.8)+
  abline(v=25)
# ----------------------------------
#Select the 25 PCA component
pca_data <- pca$x[,1:25]
View(pca_data)
dim(pca_data)
fviz_contrib(pca, choice = "var", axes = 1:2, top = 25)
# ----------------------------------
# plot contibutive/important variables
fviz_pca_var(pca, col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), alpha.var = "contrib"
)
# ----------------------------------
eig.val <- get_eigenvalue(pca)
View(eig.val)
#plot les eigenvalues
fviz_eig(pca, addlabels = TRUE, ylim = c(0, 70))
# ----------------------------------
################################
### MACHINE LEARNING MODELING 
################################
### MODELING WITH KMEANS USING PCA-DATA
################################
km <- kmeans(pca_data, centers=6, nstart=20)
print(km$cluster)
table(km$cluster)
# ----------------------------------
BSS <- km$betweenss
TSS <- km$totss
# the quality of the partition
Q <- BSS/TSS * 100
Q
# ----------------------------------
## WARD METHOD
w <- fviz_nbclust(pca_data, kmeans, method = "wss") +
  geom_vline(xintercept = 6, linetype = 2) + 
  labs(subtitle = "Elbow method")
# ----------------------------------
## SILHOUETTE METHOD
s <- fviz_nbclust(pca_data, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")
grid.arrange(w,s, nrow=1)
# ----------------------------------
# Visualization
fviz_cluster(km, pca_data, ellipse.type = "norm")
# ----------------------------------
# Compare kmeans with 2 ans 6 clusters
k2 <- kmeans(pca_data, centers = 2, nstart = 25)
k6 <- kmeans(pca_data, centers = 6, nstart = 25)
p1 <- fviz_cluster(k2, geom = "point", data = pca_data) + ggtitle("k = 2")
p5 <- fviz_cluster(k6, geom = "point",  data = pca_data) + ggtitle("k = 6")
grid.arrange(p1,p5, nrow = 1)
# ----------------------------------
################################
### MACHINE LEARNING MODELING 
################################
### MODELING WITH CAH USING PCA-DATA
################################
# Distance matrice
set.seed(2000)
data_dist <- dist(pca_data, method ="euclidean")
# ----------------------------------
# CAH with three methods
data_cah_W <- hclust(data_dist, method ="ward.D")
data_cah_C <- hclust(data_dist, method ="complete")
# ----------------------------------
# Dendogram plots
plot(data_cah_C, hang=-1, cex=0.75)
rect.hclust(data_cah_C, k=6)
# ---------------------------------
plot(data_cah_W, hang=-1, cex=0.75, labels=F)
rect.hclust(data_cah_W, k=6)
# ---------------------------------
y = cutree(data_cah_W, 6)
ColorDendrogram(data_cah_W, y = y, labels = names(y), main = "Data", 
                branchlength = 80)
# ---------------------------------
# inertie
inertie <- sort(data_cah_W$height, decreasing = TRUE)
plot(inertie[1:10], type = "s", xlab = "Nombre de classes", ylab = "Inertie")
# color important 
points(c(2, 3, 6), inertie[c(2,3, 6)], col = c("green3", "red3","blue3"), cex = 2, lwd = 3)
## Un découpage en deux classes minimise ce critère. Cependant, si l'on souhaite réaliser une analyse un peu plus fine, un nombre de classes plus élevé serait pertinent.
# ---------------------------------
# different partitions visualization
plot(data_cah_W, labels = FALSE, main = "Partition en 2, 3 ou 6 classes", 
     xlab = "", ylab = "", sub = "", axes = FALSE, hang = -1)
rect.hclust(data_cah_W, 2, border = "green3")
rect.hclust(data_cah_W, 3, border = "red3")
rect.hclust(data_cah_W, 6, border = "blue3")
# ----------------------------------
# Compare CAH with 2, 3 and 6 clusters
# Mesure the quality of each partition
sub_grp_2 <- cutree(data_cah_W, k = 2)
sub_grp_3 <- cutree(data_cah_W, k = 3)
sub_grp_6 <- cutree(data_cah_W, k = 6)
p1 <- fviz_cluster(list(data = pca_data, cluster = sub_grp_2))+ ggtitle("k = 2")
p2 <- fviz_cluster(list(data = pca_data, cluster = sub_grp_3))+ ggtitle("k = 3")
p3 <- fviz_cluster(list(data = pca_data, cluster = sub_grp_6))+ ggtitle("k = 6")
grid.arrange(p1, p2, p3, nrow = 1)
# ---------------------------------
# See the number of observations in each cluster
table(sub_grp_2)
table(sub_grp_3)
table(sub_grp_6)
View(freq(sub_grp_6))
# ---------------------------------
# Check individuals in each cluster
d <- sapply(unique(sub_grp_6),function(g)data$ID[sub_grp_6 == g])
View(d[[1]])
View(d[[2]])
View(d[[3]])
View(d[[4]])
View(d[[5]])
View(d[[6]])
# ----------------------------------
################################
### MACHINE LEARNING MODELING 
################################
### MODELING WITH SOM Algorithm
################################
# define a grid for the SOM and train
#sample.size <- nrow(pca_data)
#grid.size <- ceiling(sample.size ^ (1/2.5))
som.grid <- somgrid(xdim = 15, ydim = 15, topo = 'hexagonal', toroidal = T)
som.model <- som(as.matrix(data[3:563]), grid = som.grid, rlen=500)
# ----------------------------------
print(summary(som.model))
print(som.model$grid)
# ----------------------------------
# Visualisation 
#This plot option shows the progress over time
plot(som.model, type="changes")
# ----------------------------------
# Node count plot
## color range for the cells of the map
degrade.bleu <- function(n){
  return(rgb(0,0.4,1,alpha=seq(0,1,1/n)))
}
## count plot
plot(som.model,type="count",palette.name=degrade.bleu)
# ----------------------------------
# cell membership for each individual 
View(som.model$unit.classif)
# ----------------------------------
# number of instances assigned to each node
nb <- table(som.model$unit.classif)
View(sort(nb, decreasing=F))

#check if there are empty nodes
print(length(nb))
sum(is.na(nb))
# ----------------------------------
# U-matrix visualisation
plot(som.model, type="dist.neighbours", main = "SOM neighbour distances")
# ----------------------------------
#colors function for the charts
coolBlueHotRed <- function(n, alpha = 1) {
  rainbow(n, end=4/6, alpha=alpha)[n:1]
}
#plotting the heatmap for each variable
par(mfrow=c(8,4))
for (j in 1:28){
  plot(som.model, type = "property", property = getCodes(som.model)[,j], main=colnames(getCodes(som.model))[j], palette.name=coolBlueHotRed)
}
par(mfrow=c(1,1))
# ----------------------------------
# Visualising cluster results
## use hierarchical clustering to cluster the codebook vectors
som_cluster <- cutree(hclust(dist(som.model$codes[[1]])), 6)
# plot these results
# Colour palette definition
pretty_palette <- c("#1f77b4", '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2')
plot(som.model, type="mapping", bgcol = pretty_palette[som_cluster], keepMargins = T,
     col = NA,main = "Clusters") 
add.cluster.boundaries(som.model, som_cluster)
