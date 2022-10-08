library("tidyverse")
library("carData")

data("Prestige")
data <- Prestige

plot_loess <- function(
    loess_model,
    data,
    x_col,
    y_col,
    filename
) {
  print(
    data %>%
      ggplot(aes_string(x=x_col, y=y_col)) +
      geom_point() +
      geom_line(
        aes(
          x=sort(data[, x_col]),  # sort the x values
          # Argsort the x values and return the fitted loess values.
          y=loess_model$fitted[order(data[, x_col])]
        ),
        col="darkorange", lwd=1.5
      ) +
      theme_bw()
  )

  ggsave(paste0("./images/", filename, ".png"), width=5, height=5, dpi=100)
}


# Fit Loess model.
loess_m1 <- loess(prestige ~ income, data=Prestige)
plot_loess(loess_m1, data, "income", "prestige", "r_1")

loess_m2 <- loess(prestige ~ income, data=Prestige, span=.3, degree=1)
plot_loess(loess_m2, data, "income", "prestige", "r_2")

loess_m3 <- loess(prestige ~ education, data=Prestige)
plot_loess(loess_m3, data, "education", "prestige", "r_3")

loess_m4 <- loess(prestige ~ education, data=Prestige, span=.1)
plot_loess(loess_m4, data, "education", "prestige", "r_4")


fmri_data <- read.csv("../../data/fmri_data.csv")

loess_m5 <- loess(pulse ~ index, data=fmri_data, span=.1)
plot_loess(loess_m5, fmri_data, "index", "pulse", "r_5")

loess_m6 <- loess(pulse ~ index, data=fmri_data, span=.02)
plot_loess(loess_m6, fmri_data, "index", "pulse", "r_6")

loess_m7 <- loess(radius_5 ~ index, data=fmri_data, span=.1)
plot_loess(loess_m7, fmri_data, "index", "radius_5", "r_7")

loess_m8 <- loess(radius_5 ~ index, data=fmri_data, span=.4)
plot_loess(loess_m8, fmri_data, "index", "radius_5", "r_8")
