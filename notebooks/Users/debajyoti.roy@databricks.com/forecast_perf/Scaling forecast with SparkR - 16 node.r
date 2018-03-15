# Databricks notebook source
# MAGIC %md
# MAGIC # Scaling forecast with \\(SparkR\\)
# MAGIC 
# MAGIC #### For `1` univariate time-series: 
# MAGIC * generate training-set
# MAGIC * impute using Kalman smoothing
# MAGIC * forecast using Seasonal ARIMA
# MAGIC 
# MAGIC #### For `32` univariate time-series: 
# MAGIC * generate training-set
# MAGIC * impute using Kalman smoothing
# MAGIC * forecast using Seasonal ARIMA
# MAGIC 
# MAGIC #### For `1,000,000` univariate time-series: 
# MAGIC * generate training-set as `Parquet`
# MAGIC * impute using Kalman smoothing
# MAGIC * forecast using Seasonal ARIMA, and save as `Parquet`

# COMMAND ----------

library(SparkR)
library(magrittr)
library(ggplot2)
library(imputeTS)

# COMMAND ----------

# MAGIC %md
# MAGIC # Forecasting 1 uni-variate time-series

# COMMAND ----------

groups <- data.frame(id=1:1) %>%
  createDataFrame 

train_schema <- structType(
  structField("group", "string"), 
  structField("t", "integer"), 
  structField("y", "double"))

train_generator <- function(k, v){
  library(magrittr)
  library(dplyr)
  options(stringsAsFactors = FALSE)
      
  g <- v[1,1]
  
  data.frame(
    group = paste("ts-", g, sep=""),
    t = c(1:24),
    y = c(1.1, 1.1, 2.1, 4.1, NA, NA, NA, NA, 22.1, 24.1, NA, NA, 26.1, 28.1, NA, NA, NA, NA, 47.1, 49.1, 50.1, 50.1, 51.1, 53.1)) %>%
  dplyr::mutate(y = y*g)
}

train <- groups%>%
  gapply(
    groups$id,
    train_generator,
    train_schema)

train %>% 
  collect %>%
  ggplot(aes(x = t, y=y, color=group)) + geom_point() + geom_line()

# COMMAND ----------

impute_schema <- structType(
  structField("group", "string"), 
  structField("t", "integer"), 
  structField("y", "double"), 
  structField("y_imputed", "double"))

imputer <- function(k, v){
  library(imputeTS)
  library(tseries)
  library(magrittr)

  y_ts <- v$y %>% 
    ts %>% 
    na.kalman

  data.frame(v, y_ts)
}

imputed <- train %>%
  gapply( 
    train$group,
    imputer,
    impute_schema)

imputed_local <- imputed %>%
  collect

plotNA.imputations(imputed_local$y, imputed_local$y_imputed)

# COMMAND ----------

prediction_schema <- structType(
  structField("group", "string"), 
  structField("t", "integer"),
  structField("y_hat", "double"))

predictor <- function(k, v){
  library("forecast")
  library("tseries")
  options(stringsAsFactors = FALSE)

  v$y_hat <- v$y_imputed

  fc <- ts(v$y_hat, frequency = 24) %>%
    forecast::Arima(order=c(0, 1, 0), seasonal = list(order = c(0, 1, 0), period = 12)) %>%
    forecast::forecast(12) %>%
    data.frame

  data.frame(group= v[1,c("group")],t=1:36, y_hat=c(v$y_hat, fc$Point.Forecast))
}

prediction <- imputed %>%
  gapply( 
    imputed$group,
    predictor,
    prediction_schema)

prediction %>%
  collect %>%
  ggplot(aes(x = t, y=y_hat, color=group)) + geom_point() + geom_line()

# COMMAND ----------

# MAGIC %md
# MAGIC # Forecasting 32 uni-variate time-series

# COMMAND ----------

training_location <- "/mnt/scaling_ts_r_16/training"
prediction_location <- "/mnt/scaling_ts_r_16/prediction"

# COMMAND ----------

group_count <- 32

# COMMAND ----------

training_set <- data.frame(id=1:group_count) %>%
  createDataFrame %>%
  gapply( "id", train_generator, train_schema) %>%
  write.parquet(training_location,mode="overwrite")

read.parquet(training_location) %>%
  collect %>%
  ggplot(aes(x = t, y=y, color=group)) + geom_point() + geom_line()

# COMMAND ----------

read.parquet(training_location)  %>%
  gapply( "group", imputer, impute_schema) %>%
  gapply( "group", predictor, prediction_schema) %>%
  write.parquet(prediction_location, mode="overwrite")

# COMMAND ----------

read.parquet(prediction_location) %>%
  collect %>%
  ggplot(aes(x = t, y=y_hat, color=group)) + geom_point() + geom_line()

# COMMAND ----------

# MAGIC %md
# MAGIC # Forecasting 1,000,000 uni-variate time-series

# COMMAND ----------

group_count <- 1000000

training_set <- data.frame(id=1:group_count) %>%
  createDataFrame %>%
  gapply( "id", train_generator, train_schema) %>%
  write.parquet(training_location,mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC #### verify total row-count

# COMMAND ----------

read.parquet(training_location) %>% 
  count

# COMMAND ----------

# MAGIC %md
# MAGIC #### verify row-count for a group

# COMMAND ----------

read.parquet(training_location) %>%
  filter("group=='ts-77'") %>%
  count

# COMMAND ----------

read.parquet(training_location)  %>%
  gapply( "group", imputer, impute_schema) %>%
  gapply( "group", predictor, prediction_schema) %>%
  write.parquet(prediction_location, mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC #### verify total row-count

# COMMAND ----------

read.parquet(prediction_location) %>%
  count

# COMMAND ----------

# MAGIC %md
# MAGIC #### verify row-count for a group

# COMMAND ----------

read.parquet(prediction_location) %>%
  filter("group=='ts-77'") %>%
  count

# COMMAND ----------

# MAGIC %md
# MAGIC # Performance of following code (DBR 4.0): 
# MAGIC ```
# MAGIC read.parquet(training_location)  %>%
# MAGIC   gapply( "group", imputer, impute_schema) %>%
# MAGIC   gapply( "group", predictor, prediction_schema) %>%
# MAGIC   write.parquet(prediction_location, mode="overwrite")
# MAGIC   ```
# MAGIC 
# MAGIC |Time in minutes|Worker count|AWS instance type|Count of time-series(each with n=24)|
# MAGIC |---|---|---|---|
# MAGIC |17.78|16|i3.xlarge|1,000,000|

# COMMAND ----------

