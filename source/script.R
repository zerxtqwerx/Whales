# ============================================================================
# ПОЛНЫЙ АНАЛИЗ ДАННЫХ (5000 ОБРАЗЦОВ) - ИСПРАВЛЕННАЯ ВЕРСИЯ
# ============================================================================

library(tidyverse)
library(corrplot)
library(caret)
library(ranger)
library(gridExtra)
library(RColorBrewer)
library(doParallel)

registerDoParallel(cores = detectCores() - 1)

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================

setwd("D:/source/repos/python/whales/Whales/stats/")
cat("Рабочая директория:", getwd(), "\n")

if(file.exists("whale_data_for_r.csv")) {
  df_raw <- read.csv("whale_data_for_r.csv", stringsAsFactors = FALSE)
  cat("Загружен файл: whale_data_for_r.csv\n")
} else {
  stop("Файл с данными не найден!")
}

if(file.exists("whale_labels.csv")) {
  labels <- read.csv("whale_labels.csv")
} else {
  stop("Файл с метками не найден!")
}

df_raw$label <- labels$label
df_raw$label <- as.factor(df_raw$label)
levels(df_raw$label) <- c("Класс_0", "Класс_1")

cat("Размер данных:", nrow(df_raw), "x", ncol(df_raw), "\n")
cat("Распределение классов:\n")
print(table(df_raw$label))

# ============================================================================
# 2. ПОДГОТОВКА ДАННЫХ (РАЗВОРАЧИВАНИЕ JSON)
# ============================================================================

parse_json_array <- function(json_str) {
  if(is.na(json_str) || json_str == "" || !is.character(json_str)) return(NA)
  clean_str <- gsub("\\[|\\]", "", json_str)
  values <- as.numeric(strsplit(clean_str, ",")[[1]])
  return(values)
}

json_cols <- c()
for(col in names(df_raw)) {
  if(col != "label" && col != "filename" && col != "filepath" && col != "sample_type") {
    first_val <- df_raw[[col]][1]
    if(is.character(first_val) && grepl("\\[", first_val)) {
      json_cols <- c(json_cols, col)
    }
  }
}

if(length(json_cols) > 0) {
  df_expanded <- data.frame(row.names = 1:nrow(df_raw))
  
  for(col in json_cols) {
    first_array <- parse_json_array(df_raw[[col]][1])
    
    if(length(first_array) > 1 && !any(is.na(first_array))) {
      max_cols <- min(length(first_array), 20)
      for(i in 1:max_cols) {
        col_name <- paste0(col, "_", i)
        col_values <- sapply(1:nrow(df_raw), function(j) {
          arr <- parse_json_array(df_raw[[col]][j])
          if(length(arr) >= i) arr[i] else NA
        })
        df_expanded[[col_name]] <- col_values
      }
    } else if(length(first_array) == 1) {
      df_expanded[[col]] <- df_raw[[col]]
    }
  }
  
  numeric_cols <- names(df_raw)[sapply(df_raw, is.numeric)]
  for(col in numeric_cols) {
    if(col != "label" && !col %in% names(df_expanded)) {
      df_expanded[[col]] <- df_raw[[col]]
    }
  }
  
  df_expanded$label <- df_raw$label
  df <- df_expanded
} else {
  df <- df_raw
}

# Очистка
numeric_cols <- names(df)[sapply(df, is.numeric)]
numeric_cols <- setdiff(numeric_cols, "label")

constant_cols <- c()
for(col in numeric_cols) {
  if(length(unique(df[[col]])) == 1) {
    constant_cols <- c(constant_cols, col)
  }
}

if(length(constant_cols) > 0) {
  cat("Удаляем константные признаки:", length(constant_cols), "\n")
  numeric_cols <- setdiff(numeric_cols, constant_cols)
}

complete_rows <- complete.cases(df[, numeric_cols])
df_complete <- df[complete_rows, ]
numeric_cols_clean <- numeric_cols

cat("\nПосле очистки:\n")
cat("  Образцов:", nrow(df_complete), "\n")
cat("  Признаков:", length(numeric_cols_clean), "\n")

# ============================================================================
# 3. T-TEST
# ============================================================================

dir.create("results", showWarnings = FALSE)

results_ttest <- data.frame()
n_ttest <- min(200, length(numeric_cols_clean))
ttest_cols <- numeric_cols_clean[1:n_ttest]

cat("\n📊 Расчет T-TEST для", length(ttest_cols), "признаков...\n")

for(col in ttest_cols) {
  class0 <- df_complete[[col]][df_complete$label == "Класс_0"]
  class1 <- df_complete[[col]][df_complete$label == "Класс_1"]
  
  if(length(class0) < 2 || length(class1) < 2) next
  
  test <- tryCatch({
    t.test(class0, class1)
  }, error = function(e) {
    return(list(p.value = NA))
  })
  
  results_ttest <- rbind(results_ttest, data.frame(
    признак = col,
    среднее_0 = round(mean(class0, na.rm = TRUE), 4),
    среднее_1 = round(mean(class1, na.rm = TRUE), 4),
    p_value = round(test$p.value, 6),
    значимость = ifelse(!is.na(test$p.value) && test$p.value < 0.001, "***",
                        ifelse(!is.na(test$p.value) && test$p.value < 0.01, "**",
                               ifelse(!is.na(test$p.value) && test$p.value < 0.05, "*", "н/з")))
  ))
}

if(nrow(results_ttest) > 0) {
  results_ttest <- results_ttest[order(results_ttest$p_value), ]
  write.csv(results_ttest, "results/ttest.csv", row.names = FALSE)
  
  cat("\n📊 ТОП-10 ЗНАЧИМЫХ ПРИЗНАКОВ:\n")
  print(head(results_ttest[, c("признак", "среднее_0", "среднее_1", "p_value")], 10))
}

# ============================================================================
# 4. RANDOM FOREST
# ============================================================================

set.seed(42)

top_features <- if(nrow(results_ttest) > 0) {
  head(results_ttest$признак, min(100, length(numeric_cols_clean)))
} else {
  numeric_cols_clean[1:min(100, length(numeric_cols_clean))]
}

train_index <- createDataPartition(df_complete$label, p = 0.8, list = FALSE)
X_train <- df_complete[train_index, top_features]
X_test <- df_complete[-train_index, top_features]
y_train <- df_complete$label[train_index]
y_test <- df_complete$label[-train_index]

cat("\n📊 Random Forest обучение на", ncol(X_train), "признаках...\n")
rf_model <- ranger(x = X_train, y = y_train, num.trees = 200,
                   importance = "permutation", probability = TRUE, seed = 42)

importance_df <- data.frame(
  признак = names(rf_model$variable.importance),
  важность = rf_model$variable.importance
) %>% arrange(desc(важность))

write.csv(importance_df, "results/rf_importance.csv", row.names = FALSE)

cat("\n📊 ТОП-10 ВАЖНЫХ ПРИЗНАКОВ:\n")
print(head(importance_df, 10))

# ============================================================================
# 5. АНАЛИЗ ОШИБОК (ПЕРЕМЕЩЕНО ВПЕРЕД)
# ============================================================================

cat("\n📊 Анализ ошибок...\n")

# Получаем предсказания классов
y_pred <- predict(rf_model, X_test)$predictions

# Проверяем, что предсказания существуют
if(length(y_pred) > 0 && length(y_test) > 0) {
  
  # Создаем датафрейм с ошибками
  error_df <- data.frame(
    true_class = as.character(y_test),
    pred_class = as.character(y_pred),
    stringsAsFactors = FALSE
  )
  
  # Вычисляем ошибки
  error_df$is_error <- error_df$true_class != error_df$pred_class
  
  # Считаем ошибку
  error_rate <- mean(error_df$is_error, na.rm = TRUE)
  cat("  Ошибка на тесте:", round(error_rate * 100, 2), "%\n")
  cat("  Правильных:", sum(!error_df$is_error, na.rm = TRUE), "\n")
  cat("  Неправильных:", sum(error_df$is_error, na.rm = TRUE), "\n")
  
  # Матрица ошибок
  cat("\n  Матрица ошибок:\n")
  confusion_matrix <- table(error_df$true_class, error_df$pred_class)
  print(confusion_matrix)
  
  cm_df <- as.data.frame.matrix(confusion_matrix)
  write.csv(cm_df, "results/confusion_matrix.csv", row.names = TRUE)
  
} else {
  cat("  Ошибка: не удалось получить предсказания\n")
  error_rate <- NA
}

# ============================================================================
# 6. ROC-AUC И PRECISION-RECALL (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# ============================================================================

cat("\n📊 Расчет ROC-AUC и Precision-Recall...\n")

library(pROC)

rf_probs <- predict(rf_model, X_test, type = "response")$predictions

# Берем вероятности для положительного класса
if(is.matrix(rf_probs)) {
  pos_idx <- which(colnames(rf_probs) == "Класс_1")
  if(length(pos_idx) == 0) pos_idx <- 2
  prob_positive <- rf_probs[, pos_idx]
  cat("  Использую колонку:", colnames(rf_probs)[pos_idx], "\n")
} else {
  prob_positive <- rf_probs
}

y_test_numeric <- ifelse(y_test == "Класс_1", 1, 0)

# Проверка вероятностей
cat("\n  Проверка: для класса 1 средняя вероятность =", 
    round(mean(prob_positive[y_test_numeric == 1]), 4), "\n")
cat("  Проверка: для класса 0 средняя вероятность =", 
    round(mean(prob_positive[y_test_numeric == 0]), 4), "\n")

# ========== РАСЧЕТ AUC ЧЕРЕЗ pROC ==========
roc_obj <- roc(y_test_numeric, prob_positive, quiet = TRUE)
auc_value <- as.numeric(auc(roc_obj))
cat("\n  AUC (pROC) =", round(auc_value, 4), "\n")

# Если AUC < 0.5, инвертируем
if(auc_value < 0.5) {
  cat("  ⚠️ AUC < 0.5, инвертируем вероятности...\n")
  prob_positive <- 1 - prob_positive
  roc_obj <- roc(y_test_numeric, prob_positive, quiet = TRUE)
  auc_value <- as.numeric(auc(roc_obj))
  cat("  Исправленный AUC =", round(auc_value, 4), "\n")
}

# ========== ROC КРИВАЯ ==========
png("results/roc_curve.png", width = 800, height = 600)
plot(roc_obj, col = "blue", lwd = 2,
     main = paste("ROC Curve (AUC =", round(auc_value, 4), ")"))
abline(a = 0, b = 1, lty = 2, col = "gray")
grid()
dev.off()
cat("  ✅ ROC кривая сохранена: results/roc_curve.png\n")

# ========== PR КРИВАЯ ==========
thresholds <- seq(0, 1, length.out = 100)
precision <- numeric(length(thresholds))
recall <- numeric(length(thresholds))

for(i in 1:length(thresholds)) {
  y_pred <- ifelse(prob_positive >= thresholds[i], 1, 0)
  tp <- sum(y_pred == 1 & y_test_numeric == 1)
  fp <- sum(y_pred == 1 & y_test_numeric == 0)
  fn <- sum(y_pred == 0 & y_test_numeric == 1)
  precision[i] <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  recall[i] <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
}

valid <- is.finite(precision) & is.finite(recall)
precision <- precision[valid]
recall <- recall[valid]

order_idx <- order(recall)
recall <- recall[order_idx]
precision <- precision[order_idx]

ap <- 0
for(i in 2:length(recall)) {
  ap <- ap + (recall[i] - recall[i-1]) * precision[i]
}
ap <- ifelse(is.na(ap), 0, ap)

png("results/pr_curve.png", width = 800, height = 600)
plot(recall, precision, type = "l", col = "darkgreen", lwd = 2,
     xlab = "Recall", ylab = "Precision",
     main = paste("Precision-Recall Curve (AP =", round(ap, 4), ")"))
grid()
dev.off()
cat("  ✅ PR кривая сохранена: results/pr_curve.png\n")

# Сохраняем метрики
metrics_df <- data.frame(
  metric = c("AUC", "Average Precision", "Test Error"),
  value = c(auc_value, ap, ifelse(exists("error_rate"), error_rate, NA))
)
write.csv(metrics_df, "results/metrics.csv", row.names = FALSE)

cat("\n🎯 ИТОГОВЫЕ МЕТРИКИ:\n")
cat("  ROC-AUC =", round(auc_value, 4), "\n")
cat("  Average Precision =", round(ap, 4), "\n")
# ============================================================================
# 7. ИТОГОВЫЙ ОТЧЕТ
# ============================================================================

sink("results/summary.txt")

cat("=", rep("=", 58), "\n", sep="")
cat("АНАЛИЗ ДАННЫХ ДЛЯ ДЕТЕКЦИИ КИТОВ\n")
cat("=", rep("=", 58), "\n\n", sep="")

cat("1. РАЗМЕР ДАННЫХ:\n")
cat("   Образцов:", nrow(df_complete), "\n")
cat("   Признаков:", length(numeric_cols_clean), "\n")
cat("   Класс 0 (не кит):", sum(df_complete$label == "Класс_0"), 
    "(", round(sum(df_complete$label == "Класс_0")/nrow(df_complete)*100, 1), "%)\n")
cat("   Класс 1 (кит):", sum(df_complete$label == "Класс_1"), 
    "(", round(sum(df_complete$label == "Класс_1")/nrow(df_complete)*100, 1), "%)\n\n")

cat("2. КАЧЕСТВО МОДЕЛИ (Random Forest):\n")
cat("   ROC-AUC:", round(auc_value, 4), "\n")
cat("   Average Precision:", round(ap, 4), "\n")
if(exists("error_rate") && !is.na(error_rate)) {
  cat("   Test Error:", round(error_rate * 100, 2), "%\n")
}
cat("\n")

cat("3. ТОП-10 ЗНАЧИМЫХ ПРИЗНАКОВ (T-TEST):\n")
if(nrow(results_ttest) > 0) {
  print(head(results_ttest[, c("признак", "p_value")], 10))
}
cat("\n")

cat("4. ТОП-10 ВАЖНЫХ ПРИЗНАКОВ (RANDOM FOREST):\n")
print(head(importance_df, 10))
cat("\n")

cat("5. МАТРИЦА ОШИБОК:\n")
if(exists("confusion_matrix")) {
  print(confusion_matrix)
}

sink()

cat("\n✅ ВСЕ АНАЛИЗЫ ЗАВЕРШЕНЫ!\n")
cat("📁 Результаты сохранены в папке 'results/':\n")
cat("   - ttest.csv\n")
cat("   - rf_importance.csv\n")
cat("   - roc_curve.png\n")
cat("   - pr_curve.png\n")
cat("   - confusion_matrix.csv\n")
cat("   - metrics.csv\n")
cat("   - summary.txt\n")