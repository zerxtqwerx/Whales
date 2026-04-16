setwd("D:/source/repos/python/whales/Whales/source/")
setwd("")
cat("Рабочая директория:", getwd(), "")


df_full <- read.csv("whale_data_for_r.csv")


df <- df_full[, !names(df_full) %in% c("filename")]

numeric_cols <- names(df)[sapply(df, is.numeric)]
df_numeric <- df[, numeric_cols]

print(head(df_numeric, 3))

mean_vals <- round(apply(df_numeric, 2, mean, na.rm = TRUE), 3)
cat("Средние значения:")
print(mean_vals)

median_vals <- sapply(df_numeric, median, na.rm = TRUE)
cat("Медианы:")
print(median_vals)

get_mode <- function(x) {
  x <- x[!is.na(x)]
  uniq_x <- unique(x)
  uniq_x[which.max(tabulate(match(x, uniq_x)))]
}
mode_vals <- sapply(df_numeric, get_mode)
cat("Моды:")
print(mode_vals)

variance_vals <- round(apply(df_numeric, 2, var, na.rm = TRUE), 3)
cat("Дисперсии:")
print(variance_vals)

sd_vals <- round(apply(df_numeric, 2, sd, na.rm = TRUE), 3)
cat("Стандартные отклонения:")
print(sd_vals)

range_vals <- apply(df_numeric, 2, function(x) diff(range(x, na.rm = TRUE)))
cat("Размах:")
print(range_vals)

cat("Квантили:")
print(apply(df_numeric, 2, quantile, na.rm = TRUE))

par(mfrow = c(2, 2))
boxplot(df_numeric[, 1:min(9, ncol(df_numeric))], 
        main = "Сравнение признаков", 
        col = rainbow(9), las = 2, cex.axis = 0.7)

hist(mean_vals, main = "Распределение средних", 
     xlab = "Средние значения", col = rainbow(9))

plot(density(scale(df_numeric[,1])), main = "Плотность (1-й признак)", 
     col = "blue", lwd = 2)
if("label" %in% names(df)) {
  boxplot(df_numeric[,1] ~ df$label, 
          main = "Первый признак по классам",
          col = c("skyblue", "salmon"))
}

par(mfrow = c(1, 1))


cat("Размерность данных:", dim(df_numeric), "")
cat("Названия колонок:")
print(names(df_numeric))
cat("Типы данных:")
print(sapply(df_numeric, class))

if("label" %in% names(df)) {
  cat("Распределение классов:")
  print(table(df$label))
}


if("label" %in% names(df)) {
  
  df_class0 <- df[df$label == 0, numeric_cols]
  df_class1 <- df[df$label == 1, numeric_cols]
  
  cat("КЛАСС 0:")
  cat("Средние:")
  print(round(apply(df_class0, 2, mean, na.rm = TRUE), 3))
  cat("Медианы:")
  print(apply(df_class0, 2, median, na.rm = TRUE))
  
  cat("КЛАСС 1:")
  cat("Средние:")
  print(round(apply(df_class1, 2, mean, na.rm = TRUE), 3))
  cat("Медианы:")
  print(apply(df_class1, 2, median, na.rm = TRUE))
  
  par(mfrow = c(1, 2))
  
  top_features <- names(sort(apply(df_numeric, 2, var), decreasing = TRUE))[1:3]
  
  for(feature in top_features) {
    boxplot(df[[feature]] ~ df$label,
            main = feature,
            col = c("skyblue", "salmon"),
            xlab = "Класс", ylab = feature)
  }
  
  par(mfrow = c(1, 1))
}

first_feature <- names(df_numeric)[1]
df_sorted <- df_numeric[order(df_numeric[[first_feature]], decreasing = TRUE), ]
cat("Отсортировано по", first_feature, "(первые 5 строк):")
print(head(df_sorted[, 1:min(3, ncol(df_sorted))], 5))


high_values <- subset(df_numeric, df_numeric[[first_feature]] >= 
                        quantile(df_numeric[[first_feature]], 0.75, na.rm = TRUE))
cat("Подмножество с высокими значениями", first_feature, ":")
print(dim(high_values))

low_values <- subset(df_numeric, df_numeric[[first_feature]] <= 
                       quantile(df_numeric[[first_feature]], 0.25, na.rm = TRUE))
cat("Подмножество с низкими значениями", first_feature, ":")
print(dim(low_values))


cat("Объекты в рабочем пространстве:")
print(ls())
