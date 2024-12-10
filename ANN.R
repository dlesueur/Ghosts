library(tidyverse)
library(tidymodels)

test_data <- read_csv("test.csv")
train_data <- read_csv("train.csv")

nn_recipe <- recipe(formula=type ~ ., data=train_data) %>%
            #update_role(id, new_role="id") %>%
            step_mutate_at('color', fn=factor) %>%
            step_dummy(all_nominal_predictors()) %>%
            step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 100) %>% #or 100 or 2507) %>%
              set_engine("keras") %>% #verbose = 0 prints off less9
                set_mode("classification")

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 50)),
                            levels=5)
folds <- vfold_cv(train_data, v = 5, repeats=1)

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)


tuned_nn <- nn_wf %>%
            tune_grid(resamples=folds,
                      grid=nn_tuneGrid, 
                      metrics = metric_set(accuracy, roc_auc))

tuned_nn %>% collect_metrics() %>%
filter(.metric=="accuracy") %>%
ggplot(aes(x=hidden_units, y=mean)) + geom_line()

ggsave("annGhostPlot.png", plot = last_plot())

## CV tune, finalize and predict here and save results
## This takes a few min (10 on my laptop) so run it on becker if you want