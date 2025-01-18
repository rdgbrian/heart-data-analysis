# Load necessary libraries
library(ggplot2)
library(dplyr)

# 1. Identify column types (continuous, categorical)
identify_column_types <- function(df) {
  column_types <- sapply(df, function(col) {
    if (is.numeric(col) && length(unique(col)) > 10) {
      "continuous"
    } else if (is.numeric(col)) {
      "integer"
    } else {
      "categorical"
    }
  })
  return(column_types)
}

# 2. Summary statistics for numeric columns
numeric_summary_stats <- function(df, col_name) {
  col <- df[[col_name]]
  stats <- list(
    max = max(col, na.rm = TRUE),
    min = min(col, na.rm = TRUE),
    mean = mean(col, na.rm = TRUE),
    median = median(col, na.rm = TRUE),
    sd = sd(col, na.rm = TRUE),
    range = range(col, na.rm = TRUE)
  )
  return(stats)
}

# 3. Proportion calculation for categorical columns
categorical_proportions <- function(df, col_name) {
  col <- df[[col_name]]
  proportions <- prop.table(table(col))
  return(proportions)
}

# 4. Plot functions
plot_numeric <- function(df, col_name) {
  col <- df[[col_name]]
  
  # Histogram
  ggplot(df, aes_string(x = col_name)) +
    geom_histogram(bins = 30, fill = "skyblue", color = "black") +
    ggtitle(paste("Histogram of", col_name)) +
    theme_minimal()
  
  # Boxplot
  ggplot(df, aes_string(y = col_name)) +
    geom_boxplot(fill = "lightgreen", color = "black") +
    ggtitle(paste("Boxplot of", col_name)) +
    theme_minimal()
}

plot_categorical <- function(df, col_name) {
  ggplot(df, aes_string(x = col_name)) +
    geom_bar(fill = "lightcoral") +
    ggtitle(paste("Bar Chart of", col_name)) +
    theme_minimal()
}

# 5. Main function to process data
analyze_data <- function(df) {
  column_types <- identify_column_types(df)
  
  for (col_name in names(column_types)) {
    cat("\nColumn:", col_name, "\n")
    if (column_types[col_name] == "continuous" || column_types[col_name] == "integer") {
      cat("Type:", column_types[col_name], "\n")
      stats <- numeric_summary_stats(df, col_name)
      print(stats)
      print(plot_numeric(df, col_name))
    } else if (column_types[col_name] == "categorical") {
      cat("Type: Categorical\n")
      proportions <- categorical_proportions(df, col_name)
      print(proportions)
      print(plot_categorical(df, col_name))
    }
  }
}

# Run the main function with your DataFrame, e.g., analyze_data(your_dataframe)



df = read.csv("heart.csv")
results <- analyze_data(df)

# Example usage:
# analyze_dataframe(your_dataframe)
