# ============================================
# АНАЛИЗ ПРИЗНАКОВ КИТОВ - РАБОЧАЯ ВЕРСИЯ
# ============================================

# 1. УСТАНАВЛИВАЕМ ПРАВИЛЬНУЮ ПАПКУ
setwd("D:/source/repos/python/whales/Whales/source/")
cat("Рабочая директория:", getwd(), "\n")

# 2. ПРОВЕРЯЕМ ФАЙЛЫ
csv_files <- list.files(pattern = "\\.csv$")
cat("Найденные CSV файлы:\n")
print(csv_files)

if(length(csv_files) == 0) {
  stop("❌ Нет CSV файлов! Проверьте путь.")
}

# 3. ЗАГРУЖАЕМ ДАННЫЕ
# Пробуем загрузить whale_data_for_r.csv
if("whale_data_for_r.csv" %in% csv_files) {
  df <- read.csv("whale_data_for_r.csv")
  cat("\n✅ Загружен: whale_data_for_r.csv\n")
} else {
  # Если нет, берем первый попавшийся
  df <- read.csv(csv_files[1])
  cat("\n✅ Загружен:", csv_files[1], "\n")
}

# 4. ПРОВЕРЯЕМ ДАННЫЕ
cat("\n📊 РАЗМЕР ДАННЫХ:", dim(df), "\n")
cat("📊 КОЛОНКИ:", names(df), "\n")

# 5. ИЩЕМ МЕТКИ
if("label" %in% names(df)) {
  cat("\n✅ Найдена колонка 'label'\n")
  print(table(df$label))
} else {
  cat("\n❌ Нет колонки 'label'\n")
  cat("Доступные колонки:", names(df), "\n")
}

# 6. УСТАНАВЛИВАЕМ ПАКЕТЫ (если нужно)
packages <- c("ggplot2", "dplyr", "corrplot")
for(pkg in packages) {
  if(!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# 7. ПРОСТАЯ ВИЗУАЛИЗАЦИЯ
if("label" %in% names(df)) {
  # Распределение классов
  barplot(table(df$label), 
          main = "Распределение классов",
          col = c("skyblue", "salmon"))
  
  # Если есть числовые признаки
  numeric_cols <- names(df)[sapply(df, is.numeric)]
  if(length(numeric_cols) > 0) {
    # Берем первые 4 числовых признака
    plot_cols <- numeric_cols[1:min(4, length(numeric_cols))]
    
    # Парные графики
    pairs(df[, plot_cols], 
          col = as.numeric(df$label) + 1,
          main = "Парные графики признаков")
  }
}

cat("\n✅ ГОТОВО!\n")

install.packages("GGally")
# Для наглядности создадим более понятную визуализацию
library(ggplot2)
library(GGally)

# Если у вас есть данные с метками
if("label" %in% names(df)) {
  # Преобразуем label в фактор для цветов
  df$label <- as.factor(df$label)
  
  # Выбираем несколько признаков для анализа
  features_to_plot <- c("num_beads", "duration", "tempo", "bead_regularity")
  available_features <- features_to_plot[features_to_plot %in% names(df)]
  
  if(length(available_features) >= 2) {
    # Более понятный парный график с ggpairs
    ggpairs(df, columns = available_features, 
            aes(color = label, alpha = 0.5),
            title = "Парные графики признаков") +
      theme_minimal()
  }
}



# 1. Общая статистика по всем признакам
cat("📊 ОБЩАЯ СТАТИСТИКА ПО ВСЕМ ПРИЗНАКАМ\n")
cat("======================================\n")

# Группировка по классам
for(class in unique(df$label)) {
  cat("\nКЛАСС", class, ":\n")
  cat("--------------------\n")
  # Берем только числовые колонки
  numeric_cols <- names(df)[sapply(df, is.numeric)]
  for(col in numeric_cols) {
    vals <- df[[col]][df$label == class]
    cat(sprintf("%-25s: mean=%.3f, sd=%.3f, min=%.3f, max=%.3f\n", 
                col, mean(vals, na.rm=TRUE), sd(vals, na.rm=TRUE), 
                min(vals, na.rm=TRUE), max(vals, na.rm=TRUE)))
  }
}

# 2. t-тесты для каждого признака (сравнение классов)
cat("\n\n📊 T-ТЕСТЫ (сравнение классов 0 и 1)\n")
cat("======================================\n")
cat("p-value < 0.05 означает, что признак значимо различается между классами\n\n")

results <- data.frame()
for(col in numeric_cols) {
  test <- t.test(df[[col]] ~ df$label)
  sig <- ifelse(test$p.value < 0.05, "✅ ЗНАЧИМО", "❌ НЕ значимо")
  cat(sprintf("%-25s: p-value = %.6f - %s\n", col, test$p.value, sig))
  
  results <- rbind(results, data.frame(
    признак = col,
    p_value = test$p.value,
    значимый = test$p.value < 0.05,
    среднее_класс0 = test$estimate[1],
    среднее_класс1 = test$estimate[2],
    разница = abs(test$estimate[1] - test$estimate[2])
  ))
}

# 3. ТОП самых важных признаков (по разнице средних)
cat("\n\n🏆 ТОП-10 ПРИЗНАКОВ (по разнице средних)\n")
cat("========================================\n")
results <- results[order(-results$разница), ]
print(head(results, 10))

# 4. Корреляция между признаками
cat("\n\n📈 КОРРЕЛЯЦИЯ МЕЖДУ ПРИЗНАКАМИ\n")
cat("================================\n")
cor_matrix <- cor(df[, numeric_cols], use = "complete.obs")

# Находим сильные корреляции (>0.7)
high_cor <- which(abs(cor_matrix) > 0.7 & upper.tri(cor_matrix), arr.ind = TRUE)
if(nrow(high_cor) > 0) {
  cat("Сильные корреляции (>0.7):\n")
  for(i in 1:nrow(high_cor)) {
    cat(sprintf("  %s — %s : r = %.3f\n", 
                rownames(cor_matrix)[high_cor[i,1]],
                colnames(cor_matrix)[high_cor[i,2]],
                cor_matrix[high_cor[i,1], high_cor[i,2]]))
  }
} else {
  cat("Сильных корреляций нет\n")
}

# 5. Простая модель для проверки
cat("\n\n🤖 ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ\n")
cat("==========================\n")
model <- glm(label ~ ., data = df[, c(numeric_cols, "label")], family = binomial())
summary(model)

# 6. ИТОГОВАЯ ОЦЕНКА
cat("\n\n📊 ИТОГОВАЯ ОЦЕНКА КАЧЕСТВА ПРИЗНАКОВ\n")
cat("======================================\n")

significant <- sum(results$значимый)
total <- nrow(results)
cat(sprintf("Значимых признаков: %d из %d (%.1f%%)\n", significant, total, significant/total*100))

if(significant/total > 0.3) {
  cat("✅ Хороший набор данных - много значимых признаков\n")
} else if(significant/total > 0.1) {
  cat("⚠️ Среднее качество - есть значимые признаки\n")
} else {
  cat("❌ Плохое качество - почти нет значимых признаков\n")
}

# Лучшие признаки
best_features <- head(results[results$значимый, "признак"], 5)
cat("\nЛучшие признаки для классификации:\n")
for(f in best_features) {
  cat(sprintf("  - %s\n", f))
}















# ИСПРАВЛЕННЫЙ АНАЛИЗ (пропускает константные признаки)
# =====================================================

cat("📊 АНАЛИЗ ПРИЗНАКОВ\n")
cat("==================\n\n")

# Получаем числовые колонки
numeric_cols <- names(df)[sapply(df, is.numeric)]
cat("Всего числовых признаков:", length(numeric_cols), "\n\n")

# 1. ПРОВЕРКА КАЖДОГО ПРИЗНАКА
results <- data.frame()
constant_features <- c()

for(col in numeric_cols) {
  # Проверяем не константный ли признак
  values_class0 <- df[[col]][df$label == 0]
  values_class1 <- df[[col]][df$label == 1]
  
  # Если в одном из классов все значения одинаковые
  if(length(unique(values_class0)) == 1 || length(unique(values_class1)) == 1) {
    constant_features <- c(constant_features, col)
    next
  }
  
  # Основные статистики
  mean0 <- mean(values_class0, na.rm = TRUE)
  mean1 <- mean(values_class1, na.rm = TRUE)
  sd0 <- sd(values_class0, na.rm = TRUE)
  sd1 <- sd(values_class1, na.rm = TRUE)
  
  # t-тест
  test <- tryCatch({
    t.test(values_class0, values_class1)
  }, error = function(e) {
    return(list(p.value = NA, estimate = c(mean0, mean1)))
  })
  
  p_val <- ifelse(is.na(test$p.value), 1, test$p.value)
  
  results <- rbind(results, data.frame(
    признак = col,
    класс0_среднее = round(mean0, 4),
    класс0_стд = round(sd0, 4),
    класс1_среднее = round(mean1, 4),
    класс1_стд = round(sd1, 4),
    разница_средних = round(abs(mean0 - mean1), 4),
    p_value = round(p_val, 6),
    значимый = p_val < 0.05
  ))
}

# 2. ВЫВОД РЕЗУЛЬТАТОВ
cat("📌 ПРОПУЩЕННЫЕ ПРИЗНАКИ (константные):\n")
if(length(constant_features) > 0) {
  cat(paste("  -", constant_features, collapse = "\n"), "\n\n")
} else {
  cat("  Нет константных признаков\n\n")
}

cat("📊 ТОП-10 ПРИЗНАКОВ (по разнице средних):\n")
cat("==========================================\n")
results <- results[order(-results$разница_средних), ]
print(head(results, 10))

cat("\n\n📊 ТОП-10 ПРИЗНАКОВ (по значимости):\n")
cat("======================================\n")
results_sig <- results[order(results$p_value), ]
print(head(results_sig, 10))

# 3. СТАТИСТИКА ПО ЗНАЧИМОСТИ
cat("\n\n📊 ОБЩАЯ СТАТИСТИКА:\n")
cat("===================\n")
significant <- sum(results$значимый, na.rm = TRUE)
total <- nrow(results)
cat(sprintf("Всего проанализировано признаков: %d\n", total))
cat(sprintf("Из них значимых (p < 0.05): %d (%.1f%%)\n", 
            significant, significant/total*100))
cat(sprintf("Константных признаков: %d\n", length(constant_features)))

# 4. ЛУЧШИЕ ПРИЗНАКИ
cat("\n\n🏆 ЛУЧШИЕ ПРИЗНАКИ ДЛЯ КЛАССИФИКАЦИИ:\n")
cat("=====================================\n")
best <- head(results[results$значимый, ], 5)
if(nrow(best) > 0) {
  for(i in 1:nrow(best)) {
    cat(sprintf("\n%d. %s\n", i, best$признак[i]))
    cat(sprintf("   Класс 0: %.4f ± %.4f\n", best$класс0_среднее[i], best$класс0_стд[i]))
    cat(sprintf("   Класс 1: %.4f ± %.4f\n", best$класс1_среднее[i], best$класс1_стд[i]))
    cat(sprintf("   Разница: %.4f (p = %.6f) %s\n", 
                best$разница_средних[i], best$p_value[i],
                ifelse(best$значимый[i], "✅", "❌")))
  }
} else {
  cat("Нет значимых признаков!\n")
}

# 5. ВИЗУАЛИЗАЦИЯ ЛУЧШИХ ПРИЗНАКОВ
if(nrow(best) > 0) {
  par(mfrow = c(2, 3))
  for(i in 1:min(6, nrow(best))) {
    feature <- best$признак[i]
    boxplot(df[[feature]] ~ df$label,
            main = feature,
            xlab = "Класс", ylab = feature,
            col = c("lightblue", "lightcoral"),
            notch = TRUE)
    grid()
  }
}

# 6. КОРРЕЛЯЦИЯ МЕЖДУ ЛУЧШИМИ ПРИЗНАКАМИ
if(nrow(best) >= 2) {
  cat("\n\n📈 КОРРЕЛЯЦИЯ МЕЖДУ ЛУЧШИМИ ПРИЗНАКАМИ:\n")
  cat("=======================================\n")
  
  best_features <- best$признак[1:min(5, nrow(best))]
  cor_data <- df[, best_features]
  cor_matrix <- cor(cor_data, use = "complete.obs")
  
  for(i in 1:(ncol(cor_matrix)-1)) {
    for(j in (i+1):ncol(cor_matrix)) {
      if(abs(cor_matrix[i,j]) > 0.7) {
        cat(sprintf("⚠️  Сильная корреляция: %s — %s = %.3f\n", 
                    rownames(cor_matrix)[i], colnames(cor_matrix)[j], cor_matrix[i,j]))
      }
    }
  }
}

# 7. ПРОСТАЯ ОЦЕНКА
cat("\n\n📊 ИТОГОВАЯ ОЦЕНКА:\n")
cat("=================\n")

if(significant > total * 0.3) {
  cat("✅ ОТЛИЧНО! Много значимых признаков\n")
} else if(significant > total * 0.1) {
  cat("⚠️ СРЕДНЕ. Есть несколько хороших признаков\n")
} else {
  cat("❌ ПЛОХО. Почти нет значимых признаков\n")
}

if(length(constant_features) > total * 0.2) {
  cat("⚠️ Много константных признаков - они бесполезны\n")
}





























# ============================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ КЛАССИФИКАЦИИ КИТОВ
# ============================================

# Установка пакетов (запустите один раз)
# install.packages(c("ggplot2", "dplyr", "tidyr", "corrplot", "RColorBrewer"))

# Загружаем библиотеки
library(ggplot2)
library(dplyr)
library(tidyr)
library(corrplot)
library(RColorBrewer)

# Закрываем все графические устройства перед началом
graphics.off()
while (length(dev.list()) > 0) {
  dev.off()
}

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================

cat("\n📂 ЗАГРУЗКА ДАННЫХ\n")
cat("===================\n")

# Загружаем признаки из Python
if(file.exists("whale_optimized_features.csv")) {
  df <- read.csv("whale_optimized_features.csv", stringsAsFactors = FALSE)
  cat("✅ Загружен: whale_optimized_features.csv\n")
} else if(file.exists("whale_features.csv")) {
  df <- read.csv("whale_features.csv", stringsAsFactors = FALSE)
  cat("✅ Загружен: whale_features.csv\n")
} else {
  stop("❌ Файл с признаками не найден!")
}

# Преобразуем label в фактор
df$label <- as.factor(df$label)
levels(df$label) <- c("Класс 0", "Класс 1")

cat("\n📊 ИНФОРМАЦИЯ О ДАННЫХ:\n")
cat("  Размер:", dim(df), "\n")
cat("  Классы:\n")
print(table(df$label))
print(prop.table(table(df$label)))

# Выбираем числовые признаки
numeric_cols <- names(df)[sapply(df, is.numeric)]
if("filename" %in% names(df)) numeric_cols <- setdiff(numeric_cols, "filename")
cat("\n  Числовых признаков:", length(numeric_cols), "\n")

# ============================================
# 2. РАСПРЕДЕЛЕНИЕ КЛАССОВ
# ============================================

cat("\n📊 ВИЗУАЛИЗАЦИЯ 1: Распределение классов\n")

# Создаем и сразу сохраняем
png("01_class_distribution.png", width = 800, height = 600)
p1 <- ggplot(df, aes(x = label, fill = label)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..), vjust=-0.5, size=5) +
  scale_fill_manual(values = c("skyblue", "salmon")) +
  labs(title = "Распределение классов",
       subtitle = paste("Всего образцов:", nrow(df)),
       x = "Класс", y = "Количество") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(size=16, face="bold"))
print(p1)
dev.off()
cat("  ✅ Сохранен: 01_class_distribution.png\n")

# ============================================
# 3. BOXPLOT ДЛЯ КАЖДОГО ПРИЗНАКА (ПО ОТДЕЛЬНОСТИ)
# ============================================

cat("\n📊 ВИЗУАЛИЗАЦИЯ 2: Boxplot признаков\n")

# Ограничим количество для скорости
n_plot <- min(20, length(numeric_cols))

for(i in 1:n_plot) {
  feature <- numeric_cols[i]
  
  # Создаем файл
  png(paste0("02_boxplot_", i, "_", gsub("[^A-Za-z0-9]", "", feature), ".png"), 
      width = 800, height = 600)
  
  p <- ggplot(df, aes(x = label, y = .data[[feature]], fill = label)) +
    geom_boxplot(alpha = 0.7, outlier.size = 1.5) +
    geom_jitter(width = 0.2, alpha = 0.3, size = 0.8) +
    scale_fill_manual(values = c("skyblue", "salmon")) +
    labs(title = paste("Распределение", feature),
         x = "Класс", y = feature) +
    theme_minimal() +
    theme(legend.position = "none")
  
  print(p)
  dev.off()
  
  if(i %% 5 == 0) cat("  Обработано", i, "признаков\n")
}
cat("  ✅ Сохранено", n_plot, "boxplot'ов\n")

# ============================================
# 4. ГИСТОГРАММЫ ДЛЯ КАЖДОГО ПРИЗНАКА
# ============================================

cat("\n📊 ВИЗУАЛИЗАЦИЯ 3: Гистограммы признаков\n")

for(i in 1:n_plot) {
  feature <- numeric_cols[i]
  
  png(paste0("03_hist_", i, "_", gsub("[^A-Za-z0-9]", "", feature), ".png"), 
      width = 800, height = 600)
  
  p <- ggplot(df, aes(x = .data[[feature]], fill = label)) +
    geom_histogram(position = "identity", alpha = 0.6, bins = 30) +
    scale_fill_manual(values = c("skyblue", "salmon")) +
    labs(title = paste("Гистограмма", feature),
         x = feature, y = "Частота") +
    theme_minimal()
  
  print(p)
  dev.off()
  
  if(i %% 5 == 0) cat("  Обработано", i, "признаков\n")
}
cat("  ✅ Сохранено", n_plot, "гистограмм\n")

# ============================================
# 5. КОРРЕЛЯЦИОННАЯ МАТРИЦА
# ============================================

cat("\n📊 ВИЗУАЛИЗАЦИЯ 4: Корреляционная матрица\n")

# Вычисляем корреляционную матрицу
cor_matrix <- cor(df[, numeric_cols], use = "complete.obs")

# Сохраняем CSV
write.csv(cor_matrix, "correlation_matrix.csv")
cat("  ✅ Сохранена: correlation_matrix.csv\n")

# Тепловая карта
png("04_correlation_heatmap.png", width = 1200, height = 1000, res = 150)
corrplot(cor_matrix, method = "color", type = "upper",
         tl.cex = 0.6, tl.col = "black",
         col = colorRampPalette(c("navy", "white", "darkred"))(200),
         title = "Корреляционная матрица признаков",
         mar = c(0, 0, 2, 0))
dev.off()
cat("  ✅ Сохранена: 04_correlation_heatmap.png\n")

# Сильные корреляции
high_cor <- which(abs(cor_matrix) > 0.8 & upper.tri(cor_matrix), arr.ind = TRUE)
if(nrow(high_cor) > 0) {
  cat("\n⚠️ Сильные корреляции (>0.8):\n")
  for(i in 1:min(10, nrow(high_cor))) {
    cat(sprintf("  %s — %s: %.3f\n", 
                rownames(cor_matrix)[high_cor[i,1]],
                colnames(cor_matrix)[high_cor[i,2]],
                cor_matrix[high_cor[i,1], high_cor[i,2]]))
  }
}

# ============================================
# 6. PCA АНАЛИЗ
# ============================================

cat("\n📊 ВИЗУАЛИЗАЦИЯ 5: PCA анализ\n")

# Стандартизация
pca_data <- scale(df[, numeric_cols])
pca <- prcomp(pca_data, center = FALSE, scale. = FALSE)

# Объясненная дисперсия
var_explained <- pca$sdev^2 / sum(pca$sdev^2)
cum_var <- cumsum(var_explained)

# График объясненной дисперсии
png("05_pca_variance.png", width = 1000, height = 600)
par(mar = c(5, 5, 4, 5))
plot(var_explained[1:min(20, length(var_explained))], 
     type = "b", pch = 19, col = "blue",
     xlab = "Главная компонента", 
     ylab = "Доля дисперсии",
     main = "PCA - Объясненная дисперсия")
lines(cum_var[1:min(20, length(cum_var))], type = "b", pch = 19, col = "red")
legend("topright", legend = c("Индивидуальная", "Накопленная"),
       col = c("blue", "red"), lty = 1, pch = 19)
dev.off()
cat("  ✅ Сохранен: 05_pca_variance.png\n")

# PCA проекция
png("06_pca_plot.png", width = 1000, height = 800)
par(mar = c(5, 5, 4, 2))
plot(pca$x[,1], pca$x[,2], 
     col = c("skyblue", "salmon")[as.numeric(df$label)],
     pch = 19, cex = 1.2,
     xlab = paste0("PC1 (", round(var_explained[1]*100, 1), "%)"),
     ylab = paste0("PC2 (", round(var_explained[2]*100, 1), "%)"),
     main = "PCA проекция")
legend("topright", legend = levels(df$label), 
       col = c("skyblue", "salmon"), pch = 19)
dev.off()
cat("  ✅ Сохранен: 06_pca_plot.png\n")

# ============================================
# 7. T-SNE (ЕСЛИ УСТАНОВЛЕН)
# ============================================

cat("\n📊 ВИЗУАЛИЗАЦИЯ 6: t-SNE анализ\n")

if(require(Rtsne, quietly = TRUE)) {
  set.seed(42)
  tsne <- Rtsne(pca_data, dims = 2, perplexity = min(30, floor(nrow(df)/5)),
                check_duplicates = FALSE, verbose = FALSE)
  
  png("07_tsne_plot.png", width = 1000, height = 800)
  plot(tsne$Y[,1], tsne$Y[,2], 
       col = c("skyblue", "salmon")[as.numeric(df$label)],
       pch = 19, cex = 1.2,
       xlab = "t-SNE 1", ylab = "t-SNE 2",
       main = "t-SNE визуализация")
  legend("topright", legend = levels(df$label), 
         col = c("skyblue", "salmon"), pch = 19)
  dev.off()
  cat("  ✅ Сохранен: 07_tsne_plot.png\n")
} else {
  cat("  ⚠️ Пакет Rtsne не установлен. Пропускаем t-SNE.\n")
  cat("  Установите: install.packages('Rtsne')\n")
}

# ============================================
# 8. СТАТИСТИЧЕСКИЕ ТЕСТЫ
# ============================================

cat("\n📊 СТАТИСТИЧЕСКИЙ АНАЛИЗ\n")
cat("==========================\n")

results <- data.frame()

for(col in numeric_cols) {
  test <- t.test(df[[col]] ~ df$label)
  results <- rbind(results, data.frame(
    Признак = col,
    Среднее_класс0 = round(test$estimate[1], 4),
    Среднее_класс1 = round(test$estimate[2], 4),
    Разница = round(abs(test$estimate[1] - test$estimate[2]), 4),
    p_value = round(test$p.value, 6),
    Значимость = ifelse(test$p.value < 0.001, "***",
                        ifelse(test$p.value < 0.01, "**",
                               ifelse(test$p.value < 0.05, "*", "ns")))
  ))
}

results <- results[order(results$p_value), ]

cat("\n📋 ТОП-10 ЗНАЧИМЫХ ПРИЗНАКОВ:\n")
print(head(results, 10))

write.csv(results, "feature_statistics.csv", row.names = FALSE)
cat("  ✅ Сохранена: feature_statistics.csv\n")

# График p-values для топ-20
p_val_df <- head(results, 20)
p_val_df$Признак <- factor(p_val_df$Признак, levels = rev(p_val_df$Признак))

png("08_pvalues_plot.png", width = 1000, height = 800)
p <- ggplot(p_val_df, aes(x = Признак, y = -log10(p_value), fill = Значимость)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Статистическая значимость признаков (топ-20)",
       x = "Признак", y = "-log10(p-value)") +
  scale_fill_manual(values = c("***" = "darkred", "**" = "red", 
                               "*" = "orange", "ns" = "gray")) +
  theme_minimal() +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "blue")
print(p)
dev.off()
cat("  ✅ Сохранен: 08_pvalues_plot.png\n")

# ============================================
# 9. ИТОГОВЫЙ ОТЧЕТ
# ============================================

cat("\n📊 СОЗДАНИЕ ИТОГОВОГО ОТЧЕТА\n")
cat("==============================\n")

sink("visualization_report.txt")
cat("============================================\n")
cat("ОТЧЕТ ПО ВИЗУАЛИЗАЦИИ ДАННЫХ КИТОВ\n")
cat("============================================\n\n")
cat("Дата анализа:", date(), "\n")
cat("Количество образцов:", nrow(df), "\n")
cat("Количество признаков:", length(numeric_cols), "\n\n")

cat("РАСПРЕДЕЛЕНИЕ КЛАССОВ:\n")
print(table(df$label))
cat("\n")

cat("ТОП-10 ЗНАЧИМЫХ ПРИЗНАКОВ:\n")
print(head(results, 10))
cat("\n")

cat("СОЗДАННЫЕ ФАЙЛЫ:\n")
cat("1. 01_class_distribution.png\n")
cat("2. 02_boxplot_*_*.png (", n_plot, "файлов)\n")
cat("3. 03_hist_*_*.png (", n_plot, "файлов)\n")
cat("4. 04_correlation_heatmap.png\n")
cat("5. 05_pca_variance.png\n")
cat("6. 06_pca_plot.png\n")
if(require(Rtsne, quietly = TRUE)) cat("7. 07_tsne_plot.png\n")
cat("8. 08_pvalues_plot.png\n")
cat("9. correlation_matrix.csv\n")
cat("10. feature_statistics.csv\n")

sink()

# Закрываем все устройства
graphics.off()

cat("\n✅ ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!\n")
cat("📁 Все графики сохранены в текущей папке\n")
cat("📊 Отчет: visualization_report.txt\n")