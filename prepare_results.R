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
  group_by(Dataset, Attack) %>%
  group_map(plot_by_round, .keep = TRUE)

# For paper --------------------------------------------------------------------
source("~/doctorado/experiments/decentralized_learning/plots_utils.R")

Final <- Metrics %>%
  group_by(Dataset, Protocol, Attack, IdenticalAttack) %>%
  filter(round == 99) %>%
  filter(Attack != "Solitary targeted") %>%
  ungroup() %>%
  mutate(
    Attack = factor(
      replace(
        as.character(Attack),
        Attack == "Solitary untargeted",
        "Solitary random"
      ),
      levels = c(
        "No attack", "Label flipping", "Sign flipping",
        "Random", "Solitary random"
      )
    ),
    Protocol = fct_recode(Protocol, `No defense` = "DL")
  ) %>%
  select(
    Dataset, Protocol, Attack, IdenticalAttack,
    accuracy, f1_score, attack_success_rate, label_recall
  )

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
