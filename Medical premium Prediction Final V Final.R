# https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction
# Load libraries


library(dplyr)
library(ggplot2)
library(janitor)
library(caret)
library(randomForest)
library(tidyverse)
library(rsample)
library(ggthemes)
library(gbm)
library(rpart)
library(pls)
library(gridExtra)
library(tree)
library(RColorBrewer)
library(e1071)
# Read and clean the dataset
medpremium <- read.csv("Medicalpremium.csv")
medpremium <- clean_names(medpremium)

# Data preprocessing
medpremium <- medpremium %>%
  drop_na() %>%
  mutate(diabetes = as.factor(case_when(diabetes == 0 ~ "No",
                                        diabetes == 1 ~ "Yes"))) %>%
  mutate(blood_pressure_problems = as.factor(case_when(blood_pressure_problems == 0 ~ "No",
                                                       blood_pressure_problems == 1 ~ "Yes"))) %>%
  mutate(any_transplants = as.factor(case_when(any_transplants == 0 ~ "No",
                                               any_transplants == 1 ~ "Yes"))) %>%
  mutate(any_chronic_diseases = as.factor(case_when(any_chronic_diseases == 0 ~ "No",
                                                    any_chronic_diseases == 1 ~ "Yes"))) %>%
  mutate(known_allergies = as.factor(case_when(known_allergies == 0 ~ "No",
                                               known_allergies == 1 ~ "Yes"))) %>%
  mutate(history_of_cancer_in_family = as.factor(case_when(history_of_cancer_in_family == 0 ~ "No",
    
                                                           
                                                           
                                                                                                                 history_of_cancer_in_family == 1 ~ "Yes")))

# Exploratory data analysis - Boxplots
v1 <- ggplot(medpremium) +
  geom_boxplot(aes(y = premium_price, x = diabetes, fill = diabetes), show.legend = FALSE) +
  xlab("Diabetes") +
  ylab("Premium Price")

v2 <- ggplot(medpremium) +
  geom_boxplot(aes(y = premium_price, x = any_transplants, fill = any_transplants), show.legend = FALSE) +
  xlab("Any Transplants") +
  ylab("Premium Price")

v3 <- ggplot(medpremium) +
  geom_boxplot(aes(y = premium_price, x = any_chronic_diseases, fill = any_chronic_diseases), show.legend = FALSE) +
  xlab("Chronic Diseases") +
  ylab("Premium Price")

v4 <- ggplot(medpremium) +
  geom_boxplot(aes(y = premium_price, x = blood_pressure_problems, fill = blood_pressure_problems), show.legend = FALSE) +
  xlab("Blood Pressure Problems") +
  ylab("Premium Price")

v5 <- ggplot(medpremium) +
  geom_boxplot(aes(y = premium_price, x = known_allergies, fill = known_allergies), show.legend = FALSE) +
  xlab("Known Allergies") +
  ylab("Premium Price")

v6 <- ggplot(medpremium) +
  geom_boxplot(aes(y = premium_price, x = history_of_cancer_in_family, fill = history_of_cancer_in_family), show.legend = FALSE) +
  xlab("Cancer in Family") +
  ylab("Premium Price")

grid.arrange(v1, v2, v3, v4, v5, v6, nrow = 2)

v7 <- ggplot(medpremium) +
  geom_point(aes(x = age, y = premium_price)) +
  geom_smooth(aes(x = age, y = premium_price)) +
  xlab("Age (years)") +
  ylab("Premium Price")

v8 <- ggplot(medpremium) +
  geom_point(aes(x = weight, y = premium_price)) +
  geom_smooth(aes(x = weight, y = premium_price), colour = "green") +
  xlab("Weight (kg)") +
  ylab("Premium Price")

v9 <- ggplot(medpremium) +
  geom_point(aes(x = height, y = premium_price)) +
  geom_smooth(aes(x = height, y = premium_price), colour = "red") +
  xlab("Height (cm)") +
  ylab("Premium Price")

v10 <- ggplot(medpremium, mapping = aes(x = premium_price, y = factor(number_of_major_surgeries), fill = factor(number_of_major_surgeries))) +
  geom_violin(color = "red", fill = "orange", alpha = 0.2, show.legend = FALSE) +
  labs(fill = "Number of Major Surgeries") +
  ylab("Number of Major Surgeries") +
  xlab("Premium Price")

grid.arrange(v7, v8, v9, v10, nrow = 2)

summary(medpremium)

# Split the data into training and testing sets
set.seed(1234)
med.split <- initial_split(medpremium, prop = 3 / 4)
med.train <- training(med.split)
med.test <- testing(med.split)

# Define evaluation functions
rsquared <- function(pred, actual) {
  1 - (sum((actual - pred)^2) / sum((actual - mean(actual))^2))
}

MSE <- function(pred, actual) {
  sum((actual - pred)^2) / length(actual)
}

# Perform forward stepwise regression
linear.fwd <- step(lm(premium_price ~ ., data = med.train), direction = "forward")

fwd.pred.train <- predict(linear.fwd, med.train)
mse.fwd.train <- MSE(fwd.pred.train, med.train$premium_price)
r2.fwd.train <- rsquared(fwd.pred.train, med.train$premium_price)

fwd.pred.test <- predict(linear.fwd, med.test)
mse.fwd.test <- MSE(fwd.pred.test, med.test$premium_price)
r2.fwd.test <- rsquared(fwd.pred.test, med.test$premium_price)

# Perform Principal Component Regression (PCR)
pcr.model <- pcr(premium_price ~ ., data = med.train, scale = TRUE, validation = "CV")
validationplot(pcr.model, val.type = "MSEP", main = "Premium Price", ylab = "Mean Squared Error")

pcr.pred5.train <- predict(pcr.model, med.train, ncomp = 5)
mse.pcr5.train <- MSE(pcr.pred5.train, med.train$premium_price)
r2.pcr5.train <- rsquared(pcr.pred5.train, med.train$premium_price)

pcr.pred5.test <- predict(pcr.model, med.test, ncomp = 5)
mse.pcr5.test <- MSE(pcr.pred5.test, med.test$premium_price)
r2.pcr5.test <- rsquared(pcr.pred5.test, med.test$premium_price)

pcr.pred9.train <- predict(pcr.model, med.train, ncomp = 9)
mse.pcr9.train <- MSE(pcr.pred9.train, med.train$premium_price)
r2.pcr9.train <- rsquared(pcr.pred9.train, med.train$premium_price)

pcr.pred9.test <- predict(pcr.model, med.test, ncomp = 9)
mse.pcr9.test <- MSE(pcr.pred9.test, med.test$premium_price)
r2.pcr9.test <- rsquared(pcr.pred9.test, med.test$premium_price)

# Perform Random Forest regression
set.seed(11)
medcost.rf.model <- randomForest(premium_price ~ ., data = med.train, mtry = 3, importance = TRUE)

pred.rf.train <- predict(medcost.rf.model, med.train)
mse.rf.train <- MSE(pred.rf.train, med.train$premium_price)
r2.rf.train <- rsquared(pred.rf.train, med.train$premium_price)

pred.rf.test <- predict(medcost.rf.model, med.test)
mse.rf.test <- MSE(pred.rf.test, med.test$premium_price)
r2.rf.test <- rsquared(pred.rf.test, med.test$premium_price)

imp <- data.frame(importance(medcost.rf.model, type = 1))
imp <- rownames_to_column(imp, var = "variable")
ggplot(imp, aes(x = reorder(variable, X.IncMSE), y = X.IncMSE, color = reorder(variable, X.IncMSE))) +
  geom_point(show.legend = FALSE, size = 3) +
  geom_segment(aes(x = variable, xend = variable, y = 0, yend = X.IncMSE), size = 3, show.legend = FALSE) +
  xlab("") +
  ylab("% Increase in MSE") +
  labs(title = "Variable Importance for Prediction of Premium Price") +
  coord_flip() +
  scale_color_manual(values = colorRampPalette(brewer.pal(1, "Purples"))(10)) +
  theme_classic()

imp %>%
  arrange(desc(X.IncMSE)) %>%
  rename(`% Increase in MSE` = X.IncMSE)

# Perform Gradient Boosting regression
set.seed(1234)
medcost.boost.model <- gbm(premium_price ~ ., data = med.train, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)

pred.boost.train <- predict(medcost.boost.model, med.train, n.trees = 5000)
mse.boost.train <- MSE(pred.boost.train, med.train$premium_price)
r2.boost.train <- rsquared(pred.boost.train, med.train$premium_price)

pred.boost.test <- predict(medcost.boost.model, med.test, n.trees = 5000)
mse.boost.test <- MSE(pred.boost.test, med.test$premium_price)
r2.boost.test <- rsquared(pred.boost.test, med.test$premium_price)

# Perform Support Vector Machines (SVM)
set.seed(1234)
svm.model <- svm(premium_price ~ ., data = med.train, kernel = "radial")
pred.svm.train <- predict(svm.model, med.train)
mse.svm.train <- MSE(pred.svm.train, med.train$premium_price)
r2.svm.train <- rsquared(pred.svm.train, med.train$premium_price)
pred.svm.test <- predict(svm.model, med.test)
mse.svm.test <- MSE(pred.svm.test, med.test$premium_price)
r2.svm.test <- rsquared(pred.svm.test, med.test$premium_price)

# Perform Decision Trees
decisiontree.model <- rpart(premium_price ~ ., data = med.train)
pred.decisiontree.train <- predict(decisiontree.model, med.train)
mse.decisiontree.train <- MSE(pred.decisiontree.train, med.train$premium_price)
r2.decisiontree.train <- rsquared(pred.decisiontree.train, med.train$premium_price)
pred.decisiontree.test <- predict(decisiontree.model, med.test)
mse.decisiontree.test <- MSE(pred.decisiontree.test, med.test$premium_price)
r2.decisiontree.test <- rsquared(pred.decisiontree.test, med.test$premium_price)

# Summary of results
results <- data.frame(
  Model = c("Linear Regression (Forward Stepwise)", "Principal Component Regression (PCR, 5 components)",
            "Principal Component Regression (PCR, 9 components)", "Random Forest Regression",
            "Gradient Boosting Regression", "Support Vector Machines (SVM)", "Decision Trees"),
  Train_MSE = c(mse.fwd.train, mse.pcr5.train, mse.pcr9.train, mse.rf.train,
                mse.boost.train, mse.svm.train, mse.decisiontree.train),
  Test_MSE = c(mse.fwd.test, mse.pcr5.test, mse.pcr9.test, mse.rf.test,
               mse.boost.test, mse.svm.test, mse.decisiontree.test),
  Train_R2 = c(r2.fwd.train, r2.pcr5.train, r2.pcr9.train, r2.rf.train,
               r2.boost.train, r2.svm.train, r2.decisiontree.train),
  Test_R2 = c(r2.fwd.test, r2.pcr5.test, r2.pcr9.test, r2.rf.test,
              r2.boost.test, r2.svm.test, r2.decisiontree.test)
)

# Print the results
print(results)