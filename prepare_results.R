rm(list = ls())

setwd("~/doctorado/experiments/decentralized_learning_run")

library(tidyverse)
library(ggbasic)
library(SKM)

source("~/Desktop/notes/utils.R")
source("plots_utils.R")

Metrics <- merge_results("results", "/metrics.csv") %>%
  mutate(IdenticalAttack = factor(
    ifelse(IdenticalAttack, "Uniform", "Diverse")
  ))

write_csv(Metrics, "results/all_metrics.csv")

Metrics %>%
  pivot_longer(
    cols = c("accuracy", "f1_score", "attack_success_rate", "label_recall"),
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  na.omit() %>%
  droplevels() %>%
  group_by(Dataset, Attack) %>%
  group_map(plot_by_round, .keep = TRUE)

# For paper --------------------------------------------------------------------
new_attacks <- c(
  "No attack", "Label flipping", "Random", "Sign flipping",
  "Solitary targeted", "Solitary random"
)
names(new_attacks) <- c(
  "No attack", "Label flipping", "Random", "Sign flipping",
  "Solitary targeted", "Solitary untargeted"
)

Final <- Metrics %>%
  group_by(Dataset, Protocol, Attack, IdenticalAttack) %>%
  filter(round == 99) %>%
  filter(Attack != "Solitary targeted") %>%
  ungroup() %>%
  mutate(
    Attack = factor(
      new_attacks[as.character(Attack)],
      new_attacks
    )
  ) %>%
  select(
    Dataset, Protocol, Attack, IdenticalAttack,
    accuracy, f1_score, attack_success_rate, label_recall
  )

Baseline <- Final %>%
  filter(Protocol == "Baseline" & Attack == "No attack") %>%
  group_by(Dataset) %>%
  slice(1) %>%
  select(-IdenticalAttack) %>%
  droplevels()

Final <- Final %>%
  filter(Protocol != "Baseline") %>%
  droplevels()

Final %>%
  group_by(Dataset) %>%
  group_map(plot_results, .keep = TRUE)


Temp <- Final %>%
  filter(Dataset == "SMS Spam") %>%
  droplevels()

source("~/doctorado/experiments/decentralized_learning/plots_utils.R")

temp_plot <- plot_xy(
  Temp,
  x = "Attack",
  y = "accuracy",
  dataset = "SMS Spam"
)

save_plot(temp_plot, sprintf("plots/temp.pdf"))
