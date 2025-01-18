library(dplyr)
library(ggplot2)

# Function to determine the type of each column
detect_column_type <- function(df) {
  column_types <- sapply(df, function(column) {
    
    if (is.numeric(column) & length(unique(column)) > 3) {
      if (all(column == round(column))) {
        "integer"
      } else {
        "continuous"
      }
    } else {
      "categorical"
    }
  })
  return(column_types)
}

# Function to calculate summary statistics for continuous and integer columns
summarize_numeric <- function(column) {
  summary_stats <- list(
    max = max(column, na.rm = TRUE),
    min = min(column, na.rm = TRUE),
    mean = mean(column, na.rm = TRUE),
    median = median(column, na.rm = TRUE),
    Q1 = quantile(column,0.25, na.rm = TRUE),
    Q3 = quantile(column,0.75,na.rm = TRUE),
    IQR = IQR(column,na.rm=TRUE),
    sd = sd(column, na.rm = TRUE),
    range = diff(range(column, na.rm = TRUE))
  )
  return(summary_stats)
}

# Function to calculate proportions for categorical columns
summarize_categorical <- function(column) {
  proportions <- prop.table(table(column))
  return(proportions)
}

# Function to visualize numeric data (histogram and bar chart)
plot_numeric <- function(column, column_name) {
  # Histogram
  hist_plot <- ggplot(data = data.frame(column), aes(x = column)) +
    geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
    labs(title = paste("Histogram of", column_name), x = column_name, y = "Frequency")
  
  # Box chart
  box_plot <- ggplot(data = data.frame(column), aes(x = column)) +
    geom_boxplot() +
    labs(title = paste("Box plot of", column_name), x = column_name)
  
  return(list(hist_plot = hist_plot, box_plot = box_plot))
  #return(list(hist_plot = hist_plot))
}

# Function to visualize categorical data (bar chart)
plot_categorical <- function(column, column_name) {
  cat_plot <- ggplot(data = data.frame(column), aes(x = factor(column))) +
    geom_bar(fill = "orange", color = "black") +
    labs(title = paste("Bar Chart of", column_name), x = column_name, y = "Count")
  
  return(cat_plot)
}

# Main function to analyze and visualize the entire data frame
analyze_dataframe <- function(df) {
  column_types <- detect_column_type(df)
  
  results <- list()
  
  for (col_name in names(column_types)) {
    col_type <- column_types[col_name]
    column_data <- df[[col_name]]
    print(col_name)
    
    
    if (col_type %in% c("integer", "continuous")) {
      # Summary statistics
      results[[col_name]] <- list(
        type = col_type,
        summary = summarize_numeric(column_data)
      )
      # Plots
      plots <- plot_numeric(column_data, col_name)
      print(plots$hist_plot)
      print(plots$box_plot)
      
      # Save plots
      ggsave(paste0(col_name, "_histogram.png"), plot = plots$hist_plot, width = 8, height = 6)
      ggsave(paste0(col_name, "_boxplot.png"), plot = plots$box_plot, width = 8, height = 6)
      
    } else if (col_type == "categorical") {
      # Proportions
      results[[col_name]] <- list(
        type = col_type,
        proportions = summarize_categorical(column_data)
      )
      # Plot
      cat_plot <- plot_categorical(column_data, col_name)
      print(cat_plot)
      
      ggsave(paste0(col_name, "_categorical_plot.png"), plot = cat_plot, width = 8, height = 6)
    }
  }
  
  return(results)
}

df = read.csv("heart.csv")
results <- analyze_dataframe(df)
View(results)


