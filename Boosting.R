library(tidyverse)
library(tidymodels)
library(bonsai)
library(lightgbm)
library(embed)
test_data <- read_csv("test.csv")
train_data <- read_csv("train.csv")

ghostRecipe <- recipe(type ~ ., data = train_data) %>%
  step_mutate_at(c('color'), fn=factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type))

boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(ghostRecipe) %>%
  add_model(boost_model)

boost_tg <- grid_regular(tree_depth(),
                         trees(),
                         learn_rate(),
                         levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats=1)

cv <- boost_wf %>%
  tune_grid(resamples = folds,
            grid = boost_tg, 
            metrics = metric_set(accuracy))

best_tune <- cv %>%
  select_best(metric = "accuracy")

final_boost_wf <- boost_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train_data)

boost_preds <- final_boost_wf %>%
            predict(new_data = test_data, type="class")

submission <- boost_preds %>%
  bind_cols(., test_data) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(x=submission, file="./Boost.csv", delim=",")
