dat1=read.csv("LifetimeDataANOVA.csv")
dat1
dat1=dat1[,-1]
dat1
# pairs(dat1)
summary(dat1)

dat1$RemoteWork=factor(dat1$RemoteWork)
summary(dat1)

lm1=lm(JobSatisfaction~.,dat1)
summary(lm1)

library(car)
vif(lm1)  ## multicollinearity

## Assumptions?

plot(lm1)

boxplot(residuals(lm1))

step(lm1)

cor(dat1[, -4])


## ANOVA

Dat2=read.csv(file.choose())
Dat2

DietsAnotherWay=c(rep(1, 20), rep(2, 20), rep(3, 20))
DietsAnotherWay

summary(Dat2)
Dat2$Diets=factor(Dat2$Diets)
summary(Dat2)
boxplot(Dat2$Lifetime~Dat2$Diets, col=2:5)

attach(Dat2)
Lifetime
boxplot(Lifetime~Diets, col=2:5)

output1=aov(Lifetime~Diets)
summary(output1)
plot(output1)

