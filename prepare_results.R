rm(list = ls())

setwd("~/doctorado/experiments/decentralized_learning")

library(tidyverse)
library(ggbasic)
library(SKM)

source("~/Desktop/notes/utils.R")

attack_levels <- c(
  "No attack", "Random", "Sign flipping", "Label flipping",
  "Solitary targeted", "Solitary untargeted"
)
Metrics <- merge_results("results", "/metrics.csv") %>%
  mutate(
    Loss = log10(loss),
    Attack = factor(Attack, levels = attack_levels)
  )
write_csv(Metrics, "results/all_metrics.csv")

plot_results <- function(data, group) {
  data <- data %>%
    mutate(Value = ifelse(Metric == "loss", Value / max(Value), Value))

  plot <- line_plot(
    data,
    x = "round",
    y = "Value",
    facet_col = "Metric",
    fill_by = "Protocol",
    line_width = 0.5,
    with_points = FALSE,
    font_size = 8,
    theme = "paper",
    y_breaks_num = 5,
    x_breaks_num = 5
  )

  plot_dir <- file.path("plots", group$Dataset)
  mkdir(plot_dir)

  ggsave(
    file.path(plot_dir, sprintf("%s.png", group$Attack)),
    plot,
    width = 2400,
    height = 1200,
    units = "px"
  )

  return(1)
}

Metrics %>%
  select(-IdenticalAttack) %>%
  pivot_longer(
    cols = c(
      "accuracy", "f1_score", "attack_success_rate", "label_recall", "loss"
    ),
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  na.omit() %>%
  droplevels() %>%
  group_by(Dataset, Attack) %>%
  group_map(plot_results, .keep = TRUE)

Final <- Metrics %>%
  group_by(Dataset, Protocol, Attack, WeightingMode) %>%
  filter(round == 99)

bar_plot(
  Final,
  x = "Attack",
  # y = "attack_success_rate",
  # y = "f1_score",
  y = "accuracy",

  fill_by = "Protocol",
  facet_col = "Dataset",

  font_size = 8,
  theme = "paper",
  x_angle = 45
)
