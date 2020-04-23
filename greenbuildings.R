library(tidyverse)
library(dplyr)
library(mosaic)
library(foreach)
library(doMC)  # for parallel computing
library(gamlr) # lasso regression using the gamlr library
library(broom)

getwd()
setwd('/Users/howardyong/Documents/College/School/Spring2020/Statistical Learning:Inference/exercise3')
greenbuildings = read_csv('greenbuildings.csv')

summary(greenbuildings)
dim(greenbuildings)
head(greenbuildings)

#baseline model (includes all features)
lm_base = lm(Rent ~ (. -LEED -Energystar -CS_PropertyID -cluster)^2, data=greenbuildings)
getCall(lm_base)
coef(lm_base)
length(coef(lm_base))

#stepwise AIC forward/backward selection
lm_step = step(lm_base, 
               scope=~(.),
               direction='backward')
getCall(lm_step)
coef(lm_step)
length(coef(lm_step))


rmse = function(y, yhat) {
  sqrt( mean( (y - yhat)^2 ) )
}
n = nrow(greenbuildings)
n_train = round(0.8*n)  # round to nearest integer
n_test = n - n_train
rmse_vals = do(100)*{
  # re-split into train and test cases with the same sample sizes
  train_cases = sample.int(n, n_train, replace=FALSE)
  test_cases = setdiff(1:n, train_cases)
  green_train = greenbuildings[train_cases,]
  green_test = greenbuildings[test_cases,]
  
  # Fit to the training data
  # use `update` to refit the same model with a different set of data
  lm_base = lm(Rent ~ (. -LEED -Energystar -CS_PropertyID -cluster)^2, data=green_train)
  lm_step = update(lm_step, data=green_train)
  
  # Predictions out of sample
  yhat_test1 = predict(lm_base, green_test, na.action=na.exclude)
  yhat_test2 = predict(lm_step, green_test, na.action=na.exclude)
  
  c(rmse(green_test$Rent, yhat_test1),
    rmse(green_test$Rent, yhat_test2))
}
# noticeable improvement over the starting point!
colMeans(rmse_vals)
green_test = na.omit(green_test)
length(green_test$Rent)
length(yhat_test1)
length(yhat_test2)


#LASSO REGRESSION
n = nrow(greenbuildings)
n_train = round(0.8*n)  # round to nearest integer
n_test = n - n_train
train_cases = sample.int(n, n_train, replace=FALSE)
test_cases = setdiff(1:n, train_cases)
green_train = greenbuildings[train_cases,]
green_test = greenbuildings[test_cases,]

green_train = na.omit(green_train)
green_train_scx = sparse.model.matrix(Rent ~ (. -LEED -Energystar -CS_PropertyID -cluster)^2, data=green_train)[,-1]
green_train_scy = log(green_train$Rent)
dim(green_train_scx)
dim(green_train)
length(green_train_scy)

green_test = na.omit(green_test)
green_test_scx = sparse.model.matrix(Rent ~ (. -LEED -Energystar -CS_PropertyID -cluster)^2, data=green_test)[,-1]
green_test_scy = log(green_test$Rent)
dim(green_test_scx)
dim(green_test)
length(green_test_scy)

colnames(green_train_scx)
green_cvlasso1 = cv.gamlr(green_train_scx, green_train_scy, free=1:20, nfolds=10)
plot(green_cvlasso1)
min(green_cvlasso1$cvm)
lambda_best = green_cvlasso1$lambda.min
beta_hat = coef(green_cvlasso1)
coef_names = rownames(coef(green_cvlasso1))
lasso_model = glmnet(green_train_scx, green_train_scy, alpha=1, lambda=lambda_best, standardize=TRUE)
predictions_train1 = predict(lasso_model, s=lambda_best, newx=green_test_scx)
predictions_train2 = predict(lm_base, green_test, na.action=na.exclude)
predictions_train3 = predict(lm_step, green_test, na.action=na.exclude)

plot(lasso_model, xvar='lambda')
abline(v=log(green_cvlasso1$lambda.min), col='red', lty='dashed')
abline(v=log(green_cvlasso1$lambda.1se), col='red', lty='dashed')
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))

  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
}
eval_results(green_test_scy, predictions_train1, green_test_scx)
eval_results(green_test$Rent, predictions_train2, green_test)
eval_results(green_test$Rent, predictions_train3, green_test)
sum_results = matrix(nrow=3, ncol=2, byrow=TRUE)
sum_results = eval_results(green_test_scy, predictions_train1, green_test_scx)
sum_results = rbind(sum_results, c(eval_results(green_test$Rent, predictions_train2, green_test)))
sum_results = rbind(sum_results, c(eval_results(green_test$Rent, predictions_train3, green_test)))
sum_results


green = subset(greenbuildings, greenbuildings[14] == 1)
not_green = subset(greenbuildings, greenbuildings[14] == 0)

#GREEN
n = nrow(green)
n_train = round(0.8*n)  # round to nearest integer
n_test = n - n_train
train_cases = sample.int(n, n_train, replace=FALSE)
test_cases = setdiff(1:n, train_cases)
green_train = green[train_cases,]
green_test = green[test_cases,]

green_train = na.omit(green_train)
green_train_scx = sparse.model.matrix(Rent ~ (. -LEED -Energystar -CS_PropertyID -cluster)^2, data=green_train)[,-1]
green_train_scy = log(green_train$Rent)

green_test = na.omit(green_test)
green_test_scx = sparse.model.matrix(Rent ~ (. -LEED -Energystar -CS_PropertyID -cluster)^2, data=green_test)[,-1]
green_test_scy = log(green_test$Rent)

green_cvlasso1 = cv.gamlr(green_train_scx, green_train_scy, free=1:20, nfolds=10)
lambda_best = green_cvlasso1$lambda.min
beta_hat = coef(green_cvlasso1)
coef_names = rownames(coef(green_cvlasso1))
lasso_model = glmnet(green_train_scx, green_train_scy, alpha=1, lambda=lambda_best, standardize=TRUE)
green_predictions = predict(lasso_model, s=lambda_best, newx=green_test_scx)
dim(green_predictions)
green_predictions <- apply(green_predictions, 1:2,FUN=function(x) exp(x))

#NOT GREEEN
n = nrow(not_green)
n_train = round(0.8*n)  # round to nearest integer
n_test = n - n_train
train_cases = sample.int(n, n_train, replace=FALSE)
test_cases = setdiff(1:n, train_cases)
green_train = not_green[train_cases,]
green_test = not_green[test_cases,]

green_train = na.omit(green_train)
green_train_scx = sparse.model.matrix(Rent ~ (. -LEED -Energystar -CS_PropertyID -cluster)^2, data=green_train)[,-1]
green_train_scy = log(green_train$Rent)

green_test = na.omit(green_test)
green_test_scx = sparse.model.matrix(Rent ~ (. -LEED -Energystar -CS_PropertyID -cluster)^2, data=green_test)[,-1]
green_test_scy = log(green_test$Rent)

green_cvlasso1 = cv.gamlr(green_train_scx, green_train_scy, free=1:20, nfolds=10)
lambda_best = green_cvlasso1$lambda.min
beta_hat = coef(green_cvlasso1)
coef_names = rownames(coef(green_cvlasso1))
lasso_model = glmnet(green_train_scx, green_train_scy, alpha=1, lambda=lambda_best, standardize=TRUE)
not_green_predictions = predict(lasso_model, s=lambda_best, newx=green_test_scx)
dim(not_green_predictions)
not_green_predictions <- apply(not_green_predictions, 1:2,FUN=function(x) exp(x))

avg_green = sum(green_predictions)/dim(green_predictions)[1]
avg_not_green = sum(not_green_predictions)/dim(not_green_predictions)[1]
percentage_diff = (avg_green/avg_not_green - 1) * 100
percentage_diff

avg_occupied_green = sum(green[3])/dim(green[3])[1] * sum(green[6])/dim(green[6])[1]/100
avg_occupied_notgreen = sum(not_green[3])/dim(not_green[3])[1] * sum(not_green[6])/dim(not_green[6])[1]/100

avg_rev_green = avg_occupied_green * avg_green
avg_rev_notgreen = avg_occupied_notgreen * avg_not_green
perecentage_diff_rev = (avg_rev_green/avg_rev_notgreen - 1) * 100
#======================================================================
green_lasso = glmnet(
    x=green_train_scx,
    y=green_train_scy,
    alpha=1
)
plot(green_lasso, xvar='lambda')

green_cvlasso2 <- cv.glmnet(
    x=green_train_scx,
    y=green_train_scy,
    alpha=1
)
plot(green_cvlasso2)

min(green_cvlasso2$cvm)    # minimum MSE
#[1] 0.06606115
green_cvlasso2$lambda.min  # lambda for this min MSE
#[1] 0.0004434067
green_cvlasso2$cvm[green_cvlasso2$lambda == green_cvlasso2$lambda.1se]
#[1] 0.06979483
green_cvlasso2$lambda.1se  #lambda for this MSE
#[1] 0.0242201

green_lasso_min <- glmnet(
  x = green_train_scx,
  y = green_train_scy,
  alpha=1,
)

plot(green_lasso_min, xvar='lambda')
abline(v=log(green_cvlasso2$lambda.min), col='red', lty='dashed')
abline(v=log(green_cvlasso2$lambda.1se), col='red', lty='dashed')

coef(lasso_model, s = "lambda.1se") %>%
  tidy() %>%
  filter(row != "(Intercept)") %>%
  ggplot(aes(value, reorder(row, value), color = value > 0)) +
  geom_point(show.legend = FALSE) +
  ggtitle("Influential variables") +
  xlab("Coefficient") +
  ylab(NULL)

# minimum Lasso MSE
min(green_cvlasso2$cvm)
## [1] 0.06525995


##Go through each row and determine if a value is zero
row_sub = apply(beta_hat, 1, function(row) all(row !=0 ))
##Subset as usual
beta_hat = beta_hat[row_sub,]
lm_lasso1 = lm(Rent ~ cluster + size + empl_gr + leasing_rate + stories 
               + age + renovated + class_a + class_b + green_rating + net 
               + amenities + cd_total_07 + hd_total07 +total_dd_07 + Precipitation
               + Gas_Costs + Electricity_Costs + cluster_rent + cluster:size
               + size:stories + size:age + size:net + size:cd_total_07
               + size:cluster_rent + empl_gr:stories + empl_gr:class_a + empl_gr:hd_total07
               + leasing_rate:cluster_rent + stories:renovated + stories:class_b
               + stories:cd_total_07 + age:renovated + age:class_a
               + age:class_b + age:cd_total_07 + renovated:class_b
               + renovated:green_rating + renovated:net + renovated:cd_total_07
               + renovated:Gas_Costs
               + class_a:amenities + class_a:cluster_rent
               + class_b:amenities + class_b:Gas_Costs
               + class_b:Electricity_Costs + green_rating:amenities
               + net:Gas_Costs + amenities:Gas_Costs + Gas_Costs:Electricity_Costs
               + Electricity_Costs:cluster_rent, data=greenbuildings)
n = nrow(greenbuildings)
n_train = round(0.8*n)  # round to nearest integer
n_test = n - n_train
rmse_vals = do(100)*{
  # re-split into train and test cases with the same sample sizes
  train_cases = sample.int(n, n_train, replace=FALSE)
  test_cases = setdiff(1:n, train_cases)
  green_train = greenbuildings[train_cases,]
  green_test = greenbuildings[test_cases,]
  
  # Fit to the training data
  # use `update` to refit the same model with a different set of data
  lm_base = lm(Rent ~ (. -LEED -Energystar -CS_PropertyID)^2, data=green_train)
  lm_step = update(lm_step, data=green_train)
  lasso1 = lm(Rent ~ cluster + size + empl_gr + leasing_rate + stories 
                               + age + renovated + class_a + class_b + green_rating + net 
                               + amenities + cd_total_07 + hd_total07 +total_dd_07 + Precipitation
                               + Gas_Costs + Electricity_Costs + cluster_rent + cluster:size
                               + cluster:cluster_rent + size:stories + size:age + size:net + size:cd_total_07
                               + size:cluster_rent + empl_gr:stories + empl_gr:class_a + empl_gr:hd_total07
                               + leasing_rate:cluster_rent + stories:renovated + stories:class_b
                               + stories:cd_total_07 + age:renovated + age:class_a
                               + age:class_b + age:cd_total_07 + renovated:class_b
                               + renovated:green_rating + renovated:net + renovated:cd_total_07
                               + renovated:Gas_Costs
                               + class_a:amenities + class_a:cluster_rent
                               + class_b:amenities + class_b:Gas_Costs
                               + class_b:Electricity_Costs + green_rating:amenities
                               + net:Gas_Costs + amenities:Gas_Costs + Gas_Costs:Electricity_Costs
                               + Electricity_Costs:cluster_rent, data=greenbuildings)
  
  # Predictions out of sample
  yhat_test1 = predict(lasso1, green_test, na.action=na.exclude)
  yhat_test2 = predict(lm_base, green_test, na.action=na.exclude)
  yhat_test3 = predict(lm_step, green_test, na.action=na.exclude)
  
  c(rmse(green_test$Rent, yhat_test1),
    rmse(green_test$Rent, yhat_test2),
    rmse(green_test$Rent, yhat_test3))
}
# noticeable improvement over the starting point!
colMeans(rmse_vals)





