library(ggplot2)
library(dplyr)
library(rmarkdown)
library(ISLR) #good for examining some numerical and graphical summaries of the Smarket data



#1) Recorded the percentage returns for each of the five previous trading days, Lag1 to Lag5

names(Smarket)

dim(Smarket) #Rll = 1250  9

summary(Smarket) 


#2) the cor() function produces a matrix that contains all of the pairwise correlations among the predictors in a data set
  #cor(Smarket)  #command below gives an error message because the Direction variable is qualitative.
cor(Smarket [,-9])


#3) seeing the correlation btw Year and Volume. By plotting the data we see that Volume is increasing over time.
attach(Smarket)
plot(Volume)

#4) Fitting a logistic regression model in order to predict Direction using Lag1 through Lag5 and Volume.
  #The glm() function fits generalized glm() linear models, a class of models that includes logistic regression
  #In the glm() we pass in linear model the argument family=binomial in order to tell R to run a logistic regression
      #rather than some other type of generalized linear model.
glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume ,
            data=Smarket,family=binomial)
summary(glm.fit) #it gives Deviance Residual (with mean, median, and min and max) and it also gives us
                #the coefficients: which is the intercept, the estimate error, the z value, and the probabily(>|z|)

#5) coef() function in order to access just the coefficients for this fitted model.
coef(glm.fit) #6) We can also use the summary() function to access particular aspects of the fitted model
summary (glm.fit)$coef

#6) The predict() function is used to predict the probability that the market will go up, given values of the predictors.
  #The type="response" option tells R to output probabilities of the form P(Y = 1|X)
glm.probs=predict (glm.fit ,type="response")
glm.probs [1:10]

  #contrasts() function indicates that R has created a dummy variable with a 1 for Up.
contrasts(Direction)

#7) The following two commands create a vector of class predictions based on whether the predicted probability 
  #of a market increase is greater than or less than 0.5.
glm.pred=rep("Down",1250)   #creates a vector of 1,250 Down elements
glm.pred[glm.probs >.5]=" Up"   

#8) table() function table() can be used to produce a confusion matrix in order to determine how many
  #observations were correctly or incorrectly classified.
table(glm.pred,Direction )

(507+145)/1250

mean(glm.pred==Direction) #.116

train=(Year <2005)  #The object train is a Boolean vector, since its elements are TRUE and FALSE. 
Smarket.2005= Smarket [!train,]  #contains only the observations for which train is FALSE before 2005
dim(Smarket.2005)  #252 9
Direction.2005= Direction[!train]


#9) Fitting a logistic regression model using only the subset of the observations that correspond to dates before 2005,
  #using the subset argument.

glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,
            data=Smarket,family=binomial, subset=train)
glm.probs=predict(glm.fit, Smarket.2005, type="response")


glm.pred=rep("Down",252)
glm.pred[glm.probs >.5]=" Up"
table(glm.pred, Direction.2005)

mean(glm.pred==Direction.2005) #rll=.305

mean(glm.pred!=Direction.2005) #rll=.694


#10) we have refit the logistic regression using just Lag1 and Lag2, which seemed to have the highest predictive power
  #in the original logistic regression model.
glm.fit=glm(Direction~Lag1+Lag2, data=Smarket, family=binomial,
            subset=train)
glm.probs=predict(glm.fit,Smarket.2005, type="response")
glm.pred=rep("Down",252)
glm.pred[glm.probs >.5]="Up"
table(glm.pred, Direction.2005)

mean(glm.pred==Direction.2005) #.13

106/(106+76) #.58


#11) We want to predict Direction on a day when Lag1 and Lag2 equal 1.2 and 1.1, respectively, and on a day when
  #they equal 1.5 and -0.8. We do this using the predict() function.
predict(glm.fit, newdata=data.frame(Lag1=c(1.2, 1.5), Lag2=c(1.1,-0.8)), type="response")


#Linear Discriminant Analysis: In R, we fit a LDA model using the lda() function, which is part of the MASS library.
library(MASS)
lda.fit=lda(Direction~Lag1+Lag2 ,data=Smarket,subset=train)
lda.fit

plot(lda.fit)

#12) The first element, class, contains LDA’s predictions about the movement of the market. The second element, posterior,
  #is a matrix whose kth column contains the posterior probability that the corresponding observation belongs to the kth
  #class, computed from
lda.pred=predict (lda.fit, Smarket.2005)
names(lda.pred)  #[1] "class" "posterior " "x"

  #LDA and logistic regression predictions are almost identical.
lda.class=lda.pred$class
table(lda.class,Direction.2005)
mean(lda.class==Direction.2005)  #.5595


#13) Applying a 50 % threshold to the posterior probabilities allows us to recreate the predictions contained in lda.pred$class.
sum(lda.pred$posterior[,1]>=.5) #70
sum(lda.pred$posterior[,1]<.5) #182


#14) Notice that the posterior probability output by the model corresponds to the probability that the market will decrease:
lda.pred$posterior [1:20,1]
lda.class [1:20]

#15) Using the posterior probability threshold other than 50 % in order to make predictions
sum(lda.pred$posterior[,1]>.9) #0


#16) 4.6.4 Quadratic Discriminant Analysis: fitting a QDA model to the Smarket data. QDA is implemented
  #in R using the qda() function, which is also part of the MASS library
qda.fit=qda(Direction~Lag1+Lag2 ,data=Smarket ,subset=train)
qda.fit

  #a) The output contains the group means. But it does not contain the coef-ficients of the linear discriminants, because the 
  #QDA classifier involves a quadratic, rather than a linear, function of the predictors.
qda.class=predict(qda.fit, Smarket.2005)$class
table(qda.class, Direction.2005)

mean(qda.class==Direction.2005) #.599


#17) 4.6.5 K-Nearest Neighbors: perform KNN using the knn() function, which is part of the knn() class library. 
  # Classification: 1. A matrix containing the predictors associated with the training data, labeled train.X below.
    #2. A matrix containing the predictors associated with the data for which we wish to make predictions, labeled test.X below.
    #3. A vector containing the class labels for the training observations, labeled train.Direction below.
    #4. A value for K, the number of nearest neighbors to be used by the classifier.

  #We use the cbind() function, short for column bind, to bind the Lag1 and cbind() Lag2 variables together into two matrices
library(class)
train.X=cbind(Lag1 ,Lag2)[train ,]
test.X=cbind(Lag1 ,Lag2)[!train ,]
train.Direction =Direction [train]

  #We set a random seed before we apply knn() because if several observations are tied as nearest neighbors, then R will randomly
  #break the tie. 
set.seed(1)
knn.pred=knn(train.X,test.X,train.Direction,k=1)
table(knn.pred, Direction.2005)
(83+43) /252 #rll = .5        #252 = numbers of entries in the data frame

#18) The results using K = 1 are not very good, since only 50 % of the observations are correctly predicted. Of course, it may 
  #be that K = 1 results in an overly flexible fit to the data. Below, we repeat the analysis using K = 3.
knn.pred=knn(train.X,test.X,train.Direction,k=3)
table(knn.pred, Direction.2005)

mean(knn.pred==Direction.2005) #.536
(87+48)/252 #rll=.536 =>>>K further turns out to provide no further improvements. It appears that for this data, QDA
                          #provides the best results of the methods that we have examined so far.

#19) 4.6.6 An Application to Caravan Insurance Data: applying the KNN approach to the Caravan data set, which is part of the ISLR library.
dim(Caravan) #[1] 5822 86
attach(Caravan)
summary(Purchase)

#20) A good way to handle this problem is to standardize the data so that all standardize variables are given a mean of zero and a standard 
  #deviation of one. Then all variables will be on a comparable scale. The scale() function does just scale() this. In standardizing the 
  #data, we exclude column 86, because that is the qualitative Purchase variable.
standardized.X= scale(Caravan[,-86])
var(Caravan [ ,1]) #165.04
var(Caravan [ ,2]) #.164
var(standardized.X[,1]) #1
var(standardized.X[,2]) #1  Now every column of standardized.X has a standard deviation of one and a mean of zero.

#21)We now split the observations into a test set, containing the first 1,000 observations, and a training set, containing the remaining 
  #observations. We fit a KNN model on the training data using K = 1, and evaluate its performance on the test data.
test=1:1000
train.X= standardized.X[-test ,]  #yields the submatrix containing the observations whose indices do not range from 1 to 1, 000.
test.X= standardized.X[test ,] #yields the submatrix of the data containing the observations whose indices range from 1 to 1, 000
train.Y=Purchase [-test]
test.Y=Purchase [test]
set.seed(1)
knn.pred=knn(train.X,test.X,train.Y,k=1)
mean(test.Y!=knn.pred) #[1] 0.118
mean(test.Y!="No") #[1] 0.059

#22) It turns out that KNN with K = 1 does far better than random guessing among the customers that are predicted to buy insurance. 
  #Among 77 such customers, 9, or 11.7 %, actually do purchase insurance. This is double the rate that one would obtain from random guessing.
table(knn.pred ,test.Y)
9/(68+9) #.117


#23) Using K = 3, the success rate increases to 19 %, and with K = 5 the rate is 26.7 %. This is over four times the rate that results from
  #random guessing. It appears that KNN is finding some real patterns in a difficult data set!
knn.pred=knn(train.X,test.X,train.Y,k=3)
table(knn.pred,test.Y)

5/26 #[1] 0.192

knn.pred=knn(train.X,test.X,train.Y,k=5)
table(knn.pred,test.Y)

4/15 #[1] 0.267

#24) If we instead predict a purchase any time the predicted probability of purchase exceeds 0.25, we get much better results: we predict 
  #that 33 people will purchase insurance, and we are correct for about 33 % of these people.
glm.fit=glm(Purchase~.,data=Caravan,family=binomial, subset=-test) #Warning message:glm.fit: fitted probabilities numerically 0 or 1 occurred

glm.probs=predict (glm.fit,Caravan [test ,], type="response")
glm.pred=rep("No",1000)
glm.pred[glm.probs >.5]=" Yes"
table(glm.pred,test.Y)
  
glm.pred=rep("No",1000)
glm.pred[glm.probs >.25]=" Yes"
table(glm.pred,test.Y)
11/(22+11) #[1] 0.333





