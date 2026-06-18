# Helper: minimal valid data for dd_choice tests
.dd_test_data <- function(n = 10, delay_max = 365) {
  set.seed(1)
  data.frame(
    subj     = rep(1L, n),
    amt_LL   = sample(c(12, 15, 20, 25), n, replace = TRUE),
    delay_LL = sample(seq(7, delay_max, length.out = 6), n, replace = TRUE),
    amt_SS   = 10,
    delay_SS = 0,
    choice   = rbinom(n, 1, 0.5)
  )
}

.dd_opts <- list(
  LL = c(amt = "amt_LL", delay = "delay_LL"),
  SS = c(amt = "amt_SS", delay = "delay_SS")
)

.dd_form <- bmf(logk ~ 1, logphi ~ 1)

# ==============================================================================
# Constructor
# ==============================================================================

test_that("dd_choice creates a bmmodel object", {
  expect_silent(
    dd_choice("choice", .dd_opts)
  )
  model <- dd_choice("choice", .dd_opts)
  expect_s3_class(model, "bmmodel")
  expect_s3_class(model, "attr_choice")
  expect_s3_class(model, "dd_choice")
})

test_that("dd_choice default arguments are hyperbolic + softmax", {
  model <- dd_choice("choice", .dd_opts)
  expect_equal(model$other_vars$discount_fn, "hyperbolic")
  expect_equal(model$other_vars$choice_rule, "softmax")
})

test_that("dd_choice stores choice_col and options correctly", {
  model <- dd_choice("choice", .dd_opts)
  expect_equal(model$resp_vars$choice_col, "choice")
  expect_equal(model$other_vars$options, .dd_opts)
})

test_that("dd_choice hyperbolic+softmax has correct parameters", {
  model <- dd_choice("choice", .dd_opts,
                     discount_fn = "hyperbolic", choice_rule = "softmax")
  expect_setequal(names(model$parameters), c("logk", "logphi"))
})

test_that("dd_choice hyperbolic+luce omits logphi", {
  model <- dd_choice("choice", .dd_opts,
                     discount_fn = "hyperbolic", choice_rule = "luce")
  expect_setequal(names(model$parameters), "logk")
})

test_that("dd_choice hyperboloid adds logs parameter", {
  model <- dd_choice("choice", .dd_opts, discount_fn = "hyperboloid")
  expect_true("logs" %in% names(model$parameters))
})

test_that("dd_choice qh adds logit_beta parameter", {
  model <- dd_choice("choice", .dd_opts, discount_fn = "qh")
  expect_true("logit_beta" %in% names(model$parameters))
})

test_that("dd_choice all discount_fn values accepted", {
  for (fn in c("hyperbolic", "exponential", "hyperboloid", "qh")) {
    expect_silent(dd_choice("choice", .dd_opts, discount_fn = fn))
  }
})

test_that("dd_choice rejects missing required args", {
  expect_error(dd_choice(), "missing")
  expect_error(dd_choice("choice"), "missing")
})

test_that("dd_choice rejects options without names", {
  expect_error(
    dd_choice("choice", list(c(amt = "a", delay = "d"), c(amt = "b", delay = "e"))),
    "named"
  )
})

test_that("dd_choice rejects options missing amt or delay", {
  bad_opts <- list(LL = c(amount = "amt_LL", delay = "delay_LL"),
                   SS = c(amt = "amt_SS", delay = "delay_SS"))
  expect_error(dd_choice("choice", bad_opts), "amt")
})

test_that("dd_choice rejects unknown discount_fn", {
  expect_error(dd_choice("choice", .dd_opts, discount_fn = "power"), "arg")
})

test_that("dd_choice rejects unknown choice_rule", {
  expect_error(dd_choice("choice", .dd_opts, choice_rule = "bayes"), "arg")
})

# ==============================================================================
# check_data
# ==============================================================================

test_that("check_data.dd_choice passes on valid data", {
  model <- dd_choice("choice", .dd_opts)
  dat   <- .dd_test_data()
  expect_silent(check_data(model, dat, .dd_form))
})

test_that("check_data.dd_choice errors when choice_col missing", {
  model <- dd_choice("choice", .dd_opts)
  dat   <- .dd_test_data()
  dat$choice <- NULL
  expect_error(check_data(model, dat, .dd_form), "choice")
})

test_that("check_data.dd_choice errors when choice values not 0/1", {
  model <- dd_choice("choice", .dd_opts)
  dat   <- .dd_test_data()
  dat$choice <- dat$choice + 2L
  expect_error(check_data(model, dat, .dd_form), "0 and 1")
})

test_that("check_data.dd_choice errors when option column missing", {
  model <- dd_choice("choice", .dd_opts)
  dat   <- .dd_test_data()
  dat$amt_LL <- NULL
  expect_error(check_data(model, dat, .dd_form), "amt_LL")
})

test_that("check_data.dd_choice errors when amounts are not positive", {
  model <- dd_choice("choice", .dd_opts)
  dat   <- .dd_test_data()
  dat$amt_LL[1] <- -5
  expect_error(check_data(model, dat, .dd_form), "positive")
})

test_that("check_data.dd_choice errors when delays are negative", {
  model <- dd_choice("choice", .dd_opts)
  dat   <- .dd_test_data()
  dat$delay_LL[1] <- -1
  expect_error(check_data(model, dat, .dd_form), "non-negative")
})

test_that("check_data.dd_choice warns on narrow delay range (G1)", {
  model <- dd_choice("choice", .dd_opts)
  dat   <- .dd_test_data(delay_max = 3)   # max k_ref*delay = 0.02 * 3 = 0.06 < 0.5
  expect_warning(check_data(model, dat, .dd_form), "G1")
})

test_that("check_data.dd_choice no G1 warning with wide delays", {
  model <- dd_choice("choice", .dd_opts)
  dat   <- .dd_test_data(delay_max = 365)
  expect_no_warning(check_data(model, dat, .dd_form))
})

test_that("check_data.attr_choice passes negative amounts (sign-agnostic base)", {
  # The base class must not reject signed outcomes — that's dd_choice's job.
  # Create a minimal attr_choice object (not dd_choice) to test the base directly.
  base_model <- structure(
    list(
      resp_vars  = list(choice_col = "choice"),
      other_vars = list(
        options = list(
          LL = c(amt = "amt_LL", delay = "delay_LL"),
          SS = c(amt = "amt_SS", delay = "delay_SS")
        )
      )
    ),
    class = c("bmmodel", "attr_choice")
  )
  dat <- .dd_test_data()
  dat$amt_LL[1] <- -5   # loss trial — valid for CPT, must not error at base
  expect_no_error(check_data(base_model, dat, .dd_form))
})

test_that("check_data.dd_choice rejects negative amounts that pass attr_choice", {
  model <- dd_choice("choice", .dd_opts)
  dat   <- .dd_test_data()
  dat$amt_LL[1] <- -5
  expect_error(check_data(model, dat, .dd_form), "positive")
})

test_that("check_data.dd_choice qh adds is_delayed indicator columns", {
  model <- dd_choice("choice", .dd_opts, discount_fn = "qh")
  form  <- bmf(logk ~ 1, logit_beta ~ 1, logphi ~ 1)
  dat   <- .dd_test_data()
  out   <- check_data(model, dat, form)
  expect_true("is_delayed_delay_LL" %in% colnames(out))
  expect_true("is_delayed_delay_SS" %in% colnames(out))
  expect_true(all(out$is_delayed_delay_SS == 0L))
})

# ==============================================================================
# bmf2bf
# ==============================================================================

test_that("bmf2bf.dd_choice produces a brmsformula", {
  model   <- dd_choice("choice", .dd_opts)
  formula <- .dd_form
  bf      <- bmf2bf(model, formula)
  expect_s3_class(bf, "brmsformula")
})

test_that("bmf2bf.dd_choice hyperbolic formula contains log1p", {
  model   <- dd_choice("choice", .dd_opts, discount_fn = "hyperbolic")
  bf      <- bmf2bf(model, .dd_form)
  bf_str  <- deparse(bf)
  expect_true(any(grepl("log1p", bf_str)))
})

test_that("bmf2bf.dd_choice exponential formula contains exp(logk) * delay", {
  model  <- dd_choice("choice", .dd_opts, discount_fn = "exponential")
  bf     <- bmf2bf(model, bmf(logk ~ 1, logphi ~ 1))
  bf_str <- deparse(bf)
  expect_true(any(grepl("exp\\(logk\\)", bf_str)))
})

test_that("bmf2bf.dd_choice softmax mu uses logphi", {
  model  <- dd_choice("choice", .dd_opts, choice_rule = "softmax")
  bf     <- bmf2bf(model, .dd_form)
  bf_str <- deparse(bf)
  expect_true(any(grepl("logphi", bf_str)))
})

test_that("bmf2bf.dd_choice luce mu has no logphi", {
  model  <- dd_choice("choice", .dd_opts, choice_rule = "luce")
  bf     <- bmf2bf(model, bmf(logk ~ 1))
  bf_str <- deparse(bf)
  expect_false(any(grepl("logphi", bf_str)))
})

# ==============================================================================
# configure_model
# ==============================================================================

test_that("configure_model.dd_choice returns named list with formula and data", {
  model  <- dd_choice("choice", .dd_opts)
  dat    <- .dd_test_data()
  dat    <- check_data(model, dat, .dd_form)
  result <- configure_model(model, dat, .dd_form)
  expect_true(is.list(result))
  expect_true("formula" %in% names(result))
  expect_true("data" %in% names(result))
})

test_that("configure_model.dd_choice uses bernoulli logit family", {
  model  <- dd_choice("choice", .dd_opts)
  dat    <- .dd_test_data()
  dat    <- check_data(model, dat, .dd_form)
  result <- configure_model(model, dat, .dd_form)
  expect_equal(result$formula$family$family, "bernoulli")
  expect_equal(result$formula$family$link, "logit")
})
