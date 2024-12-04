to_name <- function(text) {
  return(to_title_case(gsub("_", " ", text)))
}

plot_by_round <- function(data, group) {
  plot <- line_plot(
    data,
    x = "round",
    y = "Value",
    facet_col = "Metric",
    facet_row = "IdenticalAttack",
    fill_by = "Protocol",
    line_width = 0.5,
    with_points = FALSE,
    font_size = 10,
    theme = "paper",
    y_breaks_num = 5,
    x_breaks_num = 5
  )

  plot_dir <- file.path("plots", "by_round", group$Dataset)
  mkdir(plot_dir)

  ggsave(
    file.path(plot_dir, sprintf("%s.pdf", group$Attack)),
    plot,
    width = 2200,
    height = 1000,
    units = "px"
  )

  return(1)
}

save_plot <- function(plot, file_path) {
  dir <- dirname(file_path)
  mkdir(dir)

  ggsave(
    file_path,
    plot,
    width = 900,
    height = 350,
    units = "px",
    dpi = 300
  )
}

plot_xy <- function(Data, x, y, dataset) {
  bar_plot(
    Data,
    x = x,
    y = y,
    y_label = to_name(y),

    fill_by = "Protocol",
    facet_col = "IdenticalAttack",

    y_breaks_num = 5,
    font_size = 4,
    theme = "paper",
    x_angle = 25
  ) +
  theme(
    strip.text.x = element_text(size = 3),
    legend.title = element_blank(),
    legend.key.size = unit(4, "points"),
    legend.margin = margin(t = 2, unit = "points"),
    plot.margin = margin(1, 1, 1, 1)
  )
}

plot_att_succ_rate <- function(Data, dataset) {
  Data %>%
    filter(!is.na(attack_success_rate)) %>%
    plot_xy(x = "Protocol", y = "attack_success_rate", dataset = dataset)
}

mnist_plots <- function(Data, group) {
  accuracy_plot <- plot_xy(
    Data,
    x = "Attack",
    y = "accuracy",
    dataset = "MNIST"
  )
  save_plot(accuracy_plot, "plots/final/MNIST/accuracy.pdf")

  att_sr_plot <- plot_att_succ_rate(Data, dataset = "MNIST")
  save_plot(att_sr_plot, "plots/final/MNIST/attack_success_rate.pdf")
}

spam_plots <- function(Data, group) {
  accuracy_plot <- plot_xy(
    Data,
    x = "Attack",
    y = "accuracy",
    dataset = "SMS Spam"
  )
  save_plot(accuracy_plot, "plots/final/SMS Spam/accuracy.pdf")

  f1_score_plot <- plot_xy(
    Data,
    x = "Attack",
    y = "f1_score",
    dataset = "SMS Spam"
  )
  save_plot(f1_score_plot, "plots/final/SMS Spam/f1_score.pdf")
}

plot_results <- function(Data, group) {
  dataset <- as.character(group$Dataset)

  if (dataset == "MNIST") mnist_plots(Data, group) else spam_plots(Data, group)
}
