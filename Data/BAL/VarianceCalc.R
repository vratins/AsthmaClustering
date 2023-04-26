library(dplyr)
library(ggplot2)

setwd("~/Desktop")

# load data from CSV file
mydata <- read.csv("normdataBAL0715_Variance.txt", header=TRUE, sep="\t")

summary(mydata)
any(is.na(mydata$ProbeName))
any(is.infinite(mydata$ProbeName))
any(is.na(mydata$Variance))
any(is.infinite(mydata$Variance))

sorted_data <- arrange(mydata, mydata$Variance)

# plot(sorted_data$ProbeName, sorted_data$Variance,
#      xlab = "ProbeName",
#      ylab = "Variance",
#      main = "Variance w/ ProbeName")

sorted_ProbeName <- sorted_data$ProbeName
sorted_Variance <- sorted_data$Variance

scatter = ggplot(data=sorted_data, aes(x=sorted_ProbeName, y=sorted_Variance)) + labs(title = "Variance w/ ProbeName", x = "ProbeName", y = "Vars") + geom_point()+geom_smooth(method = "lm", se=FALSE, color="blue", formula=y~x)
print(scatter)
