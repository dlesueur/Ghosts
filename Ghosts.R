library(tidyverse)
library(tidymodels)

test_data <- read_csv("test.csv")
train_data <- read_csv("train.csv")
train_data_missing_values <- read_csv("trainWithMissingValues.csv")

colSums(is.na(train_data_missing_values))

ghostRecipe <- recipe(type ~ ., data = train_data_missing_values) %>%
  step_mutate_at(c('color', 'type'), fn=factor) %>%
  step_impute_knn(hair_length, neighbors=5, impute_with = imp_vars('has_soul', 'color')) %>%
  step_impute_knn(rotting_flesh, neighbors=5, impute_with = imp_vars('has_soul', 'color', 'hair_length')) %>%
  step_impute_knn(bone_length, neighbors=5, impute_with = imp_vars('has_soul', 'color', 'hair_length', 'rotting_flesh'))

# ghostRecipe <- recipe(type ~ ., data = train_data_missing_values) %>%
#   step_impute_median(all_numeric_predictors())

preppedRecipe <- prep(ghostRecipe)

baked_train <- bake(preppedRecipe, new_data = train_data_missing_values)  


rmse_vec(train_data[is.na(train_data_missing_values)],
         baked_train[is.na(train_data_missing_values)])




