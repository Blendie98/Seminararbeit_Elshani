# Thesis: Random Forest Forecasting Model for Austrian Housing Construction
# Author: [Blendi Elshani]
# Description: Yearly Random Forest model with evaluation, visualization, and diagnostics

# ========================
# 1. Libraries and Settings
# ========================

install.packages("readr")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("randomForest")
install.packages("tidyr")
install.packages("scales")
install.packages("magrittr")

library(readr)
library(dplyr)
library(ggplot2)
library(randomForest)
library(tidyr)
library(scales)
library(magrittr)



# For reproducibility
set.seed(42)

# ========================
# 2. Load and Explore Data
# ========================


setwd("C:/Users/blend/Desktop/seminar new dataset")

data <- read.csv("Housing_Construction_RF_Iterative_Imputed.csv")

# View structure
str(data)
summary(data)

# ========================
# 3. Basic Diagnostics
# ========================
# Histogram of Building Completions
ggplot(data, aes(x = `Building.Completions`)) +
  geom_histogram(bins = 20, fill = "skyblue", color = "black") +
  theme_minimal() +
  labs(title = "Distribution of Building Completions", x = "Completions", y = "Frequency")




# Boxplot for detecting outliers
ggplot(data, aes(y = `Building.Completions`)) +
  geom_boxplot(fill = "tomato", alpha = 0.6) +
  theme_minimal() +
  labs(title = "Boxplot of Building Completions")


# Set font globally
par(family = "sans", cex = 0.3)  



# Correlation matrix (heatmap style)
cor_data <- cor(select(data, -Year))
cor_data 

heatmap(cor_data, symm = TRUE, main = "Correlation Matrix of Features")






# ========================
# 4. Define Features and Target
# ========================
target <- "Building.Completions"
features <- setdiff(names(data), c("Year", target))
X <- data[, features]
y <- data[[target]]

# ========================
# 5. Time-Aware Train-Test Split
# ========================
split_idx <- floor(0.8 * nrow(data))
X_train <- X[1:split_idx, ]
y_train <- y[1:split_idx]
X_test <- X[(split_idx + 1):nrow(data), ]
y_test <- y[(split_idx + 1):nrow(data)]
years_test <- data$Year[(split_idx + 1):nrow(data)]



# ========================
# 6. Train Random Forest Model
# ========================
rf_model <- randomForest(x = X_train, y = y_train, ntree = 500)
rf_model



# ========================
# 7. Predictions & Evaluation
# ========================
y_pred <- predict(rf_model, X_test)
y_pred



# Evaluation metrics
evaluate_model <- function(actual, predicted) {
  mae <- mean(abs(actual - predicted))
  rmse <- sqrt(mean((actual - predicted)^2))
  r2 <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  return(list(MAE = mae, RMSE = rmse, R2 = r2))
}

metrics <- evaluate_model(y_test, y_pred)
print(metrics)



# ========================
# 8. Actual vs Predicted Plot
# ========================
pred_df <- data.frame(Year = years_test, Actual = y_test, Predicted = y_pred)
pred_df





ggplot(pred_df, aes(x = Year)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 1.3) +
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 1.3, linetype = "dashed") +
  scale_color_manual(values = c("Actual" = "#1f77b4", "Predicted" = "#ff7f0e")) +
  scale_y_continuous(labels = scales::comma) +
  scale_x_continuous(breaks = pretty(pred_df$Year, n = 10)) +
  labs(
    title = "Actual vs Predicted Building Completions",
    subtitle = "Comparison of true and forecasted completions over time",
    x = "Year",
    y = "Number of Completions",
    color = "Legend"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 13, margin = ggplot2::margin(t = 0, r = 0, b = 10, l = 0)),
    axis.title = element_text(face = "bold"),
    legend.position = "top",
    panel.grid.minor = element_blank()
  )




# ========================
# 9. Feature Importance
# ========================

importance(rf_model)


importance_df <- data.frame(Feature = rownames(importance(rf_model)),
                            Importance = importance(rf_model)[, 1]) %>%
  arrange(desc(Importance))

importance_df


ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Feature Importance from Random Forest", x = "Feature", y = "Importance")



# ========================
# 10. Save Outputs
# ========================
ggsave("actual_vs_predicted.png", width = 8, height = 5)
ggsave("feature_importance.png", width = 8, height = 5)



