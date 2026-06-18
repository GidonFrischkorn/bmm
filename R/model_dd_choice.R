############################################################################# !
# MODELS                                                                  ####
############################################################################# !

.DD_DISCOUNT_FNS <- c("hyperbolic", "exponential", "hyperboloid", "qh")
.DD_CHOICE_RULES <- c("softmax", "luce")

.dd_choice_params <- function(discount_fn, choice_rule) {
  base <- list(
    logk = "Log-discount rate. Higher values = steeper discounting (more impatient)."
  )
  extra <- switch(
    discount_fn,
    hyperboloid = list(
      logs = "Log-scaling exponent s. V = A / (1 + k*D)^s; s=1 reduces to hyperbolic."
    ),
    qh = list(
      logit_beta = "Logit-present-bias. beta = inv_logit(logit_beta) scales immediate-future gap."
    ),
    list()
  )
  phi <- if (choice_rule == "softmax") {
    list(logphi = "Log-sensitivity phi. Higher phi = sharper choice (less noise).")
  } else {
    list()
  }
  c(base, extra, phi)
}

.dd_choice_links <- function(discount_fn, choice_rule) {
  base <- list(logk = "identity")
  extra <- switch(
    discount_fn,
    hyperboloid = list(logs = "identity"),
    qh          = list(logit_beta = "identity"),
    list()
  )
  phi <- if (choice_rule == "softmax") list(logphi = "identity") else list()
  c(base, extra, phi)
}

.dd_choice_priors <- function(discount_fn, choice_rule) {
  base <- list(
    logk = list(main = "normal(-4, 1.5)", effects = "normal(0, 0.5)")
  )
  extra <- switch(
    discount_fn,
    hyperboloid = list(
      logs = list(main = "normal(0, 0.5)", effects = "normal(0, 0.3)")
    ),
    qh = list(
      logit_beta = list(main = "normal(-0.7, 1)", effects = "normal(0, 0.5)")
    ),
    list()
  )
  phi <- if (choice_rule == "softmax") {
    list(logphi = list(main = "normal(0.7, 1)", effects = "normal(0, 0.5)"))
  } else {
    list()
  }
  c(base, extra, phi)
}

# Returns a character string for the brms nlf() expression that computes the
# log-discounted value for one option under a given discount function.
# lv_name: internal variable name (e.g. "lVLL"); amt/delay: column names.
.dd_lv_expr <- function(discount_fn, lv_name, amt_col, delay_col) {
  switch(
    discount_fn,
    hyperbolic  = glue::glue(
      "{lv_name} ~ log({amt_col}) - log1p(exp(logk) * {delay_col})"
    ),
    exponential = glue::glue(
      "{lv_name} ~ log({amt_col}) - exp(logk) * {delay_col}"
    ),
    hyperboloid = glue::glue(
      "{lv_name} ~ log({amt_col}) - exp(logs) * log1p(exp(logk) * {delay_col})"
    ),
    qh = {
      is_del_col <- paste0("is_delayed_", delay_col)
      glue::glue(
        "{lv_name} ~ log({amt_col}) + {is_del_col} * log_inv_logit(logit_beta) - exp(logk) * {delay_col}"
      )
    },
    stop2("Unknown discount_fn: {discount_fn}")
  )
}

# Returns the NLF expression string for `mu` (linear predictor for choice).
.dd_mu_expr <- function(choice_rule, lv_names) {
  lv1 <- lv_names[1]; lv2 <- lv_names[2]
  if (choice_rule == "softmax") {
    glue::glue("mu ~ exp(logphi) * (exp({lv1}) - exp({lv2}))")
  } else {
    glue::glue("mu ~ {lv1} - {lv2}")
  }
}


.model_dd_choice <- function(
  choice_col  = NULL,
  options     = NULL,
  discount_fn = "hyperbolic",
  choice_rule = "softmax",
  call        = NULL,
  ...
) {
  structure(
    list(
      resp_vars = list(choice_col = choice_col),
      other_vars = list(
        options     = options,
        discount_fn = discount_fn,
        choice_rule = choice_rule
      ),
      domain   = "Decision Making",
      task     = "Binary delay-discounting choice",
      name     = "Delay Discounting Choice Model",
      citation = glue::glue(
        "Mazur, J. E. (1987). An adjusting procedure for studying delayed reinforcement. \\
        In M. L. Commons, J. E. Mazur, J. A. Nevin, & H. Rachlin (Eds.), \\
        Quantitative analyses of behavior: Vol. 5. The effect of delay and of \\
        intervening events on reinforcement value (pp. 55-73). Lawrence Erlbaum."
      ),
      version = discount_fn,
      requirements = glue::glue(
        "- Provide a binary choice variable (0/1) in `choice_col`.\n",
        "- Provide attribute columns for each option via `options = list(...)`. \n",
        "  Each option must have an `amt` (amount > 0) and a `delay` (>= 0) column.\n",
        "- For `discount_fn = 'qh'`, include both delayed and immediate trials."
      ),
      parameters     = .dd_choice_params(discount_fn, choice_rule),
      links          = .dd_choice_links(discount_fn, choice_rule),
      fixed_parameters = list(),
      default_priors = .dd_choice_priors(discount_fn, choice_rule),
      void_mu        = FALSE
    ),
    class = c("bmmodel", "attr_choice", "dd_choice"),
    call  = call
  )
}


############################################################################# !
# USER-FACING CONSTRUCTOR                                                  ####
############################################################################# !

#' @title Delay Discounting Choice Model
#' @name dd_choice
#'
#' @description
#' A hierarchical Bayesian model for binary delay-discounting choice tasks. On
#' each trial a participant chooses between two options that differ in reward
#' amount and delay. The model estimates a per-participant discount rate
#' \eqn{k} (and optionally a sensitivity parameter \eqn{\phi}) from the choice
#' sequence.
#'
#' Supported discount functions:
#' \describe{
#'   \item{`"hyperbolic"`}{V = A / (1 + k * D) — Mazur (1987)}
#'   \item{`"exponential"`}{V = A * exp(-k * D)}
#'   \item{`"hyperboloid"`}{V = A / (1 + k * D)^s — Green & Myerson (2004)}
#'   \item{`"qh"`}{V = beta * A * exp(-k * D) for D > 0; V = A for D = 0
#'     (quasi-hyperbolic, Laibson 1997)}
#' }
#'
#' @param choice_col Character. Name of the column containing binary choices
#'   (1 = chose first option, 0 = chose second option).
#' @param options A named list specifying the column names for each option's
#'   attributes. Each element must be a named character vector with entries
#'   `amt` (amount > 0) and `delay` (delay >= 0). Option labels are
#'   user-chosen (e.g. `"LL"`, `"SS"`). Example:
#'   ```r
#'   list(LL = c(amt = "amt_LL", delay = "delay_LL"),
#'        SS = c(amt = "amt_SS", delay = "delay_SS"))
#'   ```
#' @param discount_fn Character. The discount function to use. One of
#'   `"hyperbolic"` (default), `"exponential"`, `"hyperboloid"`, or `"qh"`.
#' @param choice_rule Character. The choice rule. One of `"softmax"` (default,
#'   estimates sensitivity \eqn{\phi}) or `"luce"` (log-ratio rule, no
#'   sensitivity parameter).
#' @param ... Used internally for testing; ignore.
#'
#' @return An object of class `bmmodel`.
#'
#' @details
#' **Model parameters**
#'
#' All discount functions include `logk` (log-discount-rate). Additional
#' parameters depend on the `discount_fn`:
#' - `"hyperboloid"`: also estimates `logs` (log-scaling exponent).
#' - `"qh"`: also estimates `logit_beta` (logit-present-bias).
#' - `choice_rule = "softmax"`: also estimates `logphi` (log-sensitivity).
#'
#' **Priors**
#'
#' Default priors are weakly informative and centred near typical human
#' discounting rates:
#' - `logk ~ Normal(-4, 1.5)` — centred at k ≈ 0.018 per day
#' - `logphi ~ Normal(0.7, 1)` — centred at phi ≈ 2
#' - `logs ~ Normal(0, 0.5)` — centred at s = 1 (hyperbolic limit)
#' - `logit_beta ~ Normal(-0.7, 1)` — centred at beta ≈ 0.33
#'
#' **Formula**
#'
#' Users specify random effects and covariates for each parameter via `bmf()`:
#' ```r
#' bmf(
#'   logk   ~ 1 + (1 | subj),
#'   logphi ~ 1
#' )
#' ```
#' Parameters without an explicit formula receive a fixed-intercept formula
#' (`~ 1`) automatically.
#'
#' **Identifiability**
#'
#' - Delay range should be wide (G1 guard): at least one delay 10× the
#'   smallest delay, or `max(k_ref × delay) ≥ 0.5`.
#' - With `choice_rule = "softmax"`, k and phi are confounded when all
#'   delays are short and the discount function is near-linear. Use wide
#'   delays or fix phi via `bmf(logphi = log(1))`.
#' - For `"qh"`, include both immediate (D = 0) and delayed trials.
#'
#' @references
#' Mazur, J. E. (1987). An adjusting procedure for studying delayed
#' reinforcement. In M. L. Commons et al. (Eds.), *Quantitative analyses of
#' behavior: Vol. 5* (pp. 55-73). Lawrence Erlbaum.
#'
#' Green, L., & Myerson, J. (2004). A discounting framework for choice with
#' delayed and probabilistic rewards. *Psychological Bulletin, 130*(5), 769-792.
#'
#' Laibson, D. (1997). Golden eggs and hyperbolic discounting. *Quarterly
#' Journal of Economics, 112*(2), 443-478.
#'
#' @keywords bmmodel
#'
#' @examplesIf isTRUE(Sys.getenv("BMM_EXAMPLES"))
#' # Simulate a small dataset
#' set.seed(42)
#' n_subj <- 5; n_trials <- 40
#' dat <- do.call(rbind, lapply(seq_len(n_subj), function(s) {
#'   k_s <- exp(rnorm(1, -4, 0.6))
#'   phi_s <- exp(0.7)
#'   grid <- expand.grid(
#'     amt_LL  = c(15, 20, 25),
#'     delay_LL = c(14, 30, 90, 180, 365)
#'   )
#'   des <- grid[sample(nrow(grid), n_trials, replace = TRUE), ]
#'   des$amt_SS   <- 10
#'   des$delay_SS <- 0
#'   des$subj     <- s
#'   V_LL <- des$amt_LL / (1 + k_s * des$delay_LL)
#'   V_SS <- des$amt_SS / (1 + k_s * des$delay_SS)
#'   des$choice <- rbinom(n_trials, 1, plogis(phi_s * (V_LL - V_SS)))
#'   des
#' }))
#' dat$subj <- as.factor(dat$subj)
#'
#' model <- dd_choice(
#'   choice_col  = "choice",
#'   options     = list(LL = c(amt = "amt_LL", delay = "delay_LL"),
#'                      SS = c(amt = "amt_SS", delay = "delay_SS")),
#'   discount_fn = "hyperbolic",
#'   choice_rule = "softmax"
#' )
#'
#' formula <- bmf(
#'   logk   ~ 1 + (1 | subj),
#'   logphi ~ 1
#' )
#'
#' fit <- bmm(formula = formula, data = dat, model = model, chains = 2, iter = 1000)
#' summary(fit)
#'
#' @export
dd_choice <- function(choice_col, options,
                      discount_fn = "hyperbolic",
                      choice_rule = "softmax",
                      ...) {
  call <- match.call()
  stop_missing_args()

  stopif(
    !is.list(options) || length(options) < 2,
    "The `options` argument must be a named list with at least two options."
  )
  stopif(
    is.null(names(options)) || any(!nzchar(names(options))),
    "All elements of `options` must be named (e.g. list(LL = ..., SS = ...))."
  )
  for (opt in names(options)) {
    stopif(
      !all(c("amt", "delay") %in% names(options[[opt]])),
      "Option '{opt}' in `options` must have both 'amt' and 'delay' entries."
    )
  }

  discount_fn <- match.arg(discount_fn, .DD_DISCOUNT_FNS)
  choice_rule <- match.arg(choice_rule, .DD_CHOICE_RULES)

  .model_dd_choice(
    choice_col  = choice_col,
    options     = options,
    discount_fn = discount_fn,
    choice_rule = choice_rule,
    call        = call,
    ...
  )
}


############################################################################# !
# CHECK_DATA S3 methods                                                   ####
############################################################################# !

#' @export
check_data.dd_choice <- function(model, data, formula) {
  options     <- model$other_vars$options
  discount_fn <- model$other_vars$discount_fn
  if (is.null(options)) return(NextMethod("check_data"))
  opt_names   <- names(options)

  # ---- amounts must be positive --------------------------------------------
  for (opt in opt_names) {
    amt_col <- options[[opt]]["amt"]
    stopif(
      any(data[[amt_col]] <= 0, na.rm = TRUE),
      "Amount column '{amt_col}' must contain only positive values."
    )
  }

  # ---- delays must be non-negative -----------------------------------------
  for (opt in opt_names) {
    delay_col <- options[[opt]]["delay"]
    stopif(
      any(data[[delay_col]] < 0, na.rm = TRUE),
      "Delay column '{delay_col}' must contain only non-negative values."
    )
  }

  # ---- G1: delay-range guard -----------------------------------------------
  # Warn if the discount function is near-linear (delays too short for k to
  # be identified). Heuristic: max(k_ref * delay) >= 0.5 where k_ref = 0.02.
  k_ref <- 0.02
  max_kd <- max(vapply(opt_names, function(opt) {
    delay_vals <- data[[options[[opt]]["delay"]]]
    max(k_ref * delay_vals, na.rm = TRUE)
  }, numeric(1)))
  warnif(
    max_kd < 0.5,
    "Delay-range guard (G1): max(k_ref * delay) = {round(max_kd, 3)} < 0.5.
    The discount function may be near-linear across the observed delay range,
    making it difficult to identify k. Include longer delays (e.g. >= 25 days
    for a typical human discount rate of k ~ 0.02)."
  )

  # ---- QH: create is_delayed indicator columns -----------------------------
  if (discount_fn == "qh") {
    for (opt in opt_names) {
      delay_col    <- options[[opt]]["delay"]
      is_del_col   <- paste0("is_delayed_", delay_col)
      data[[is_del_col]] <- as.integer(data[[delay_col]] > 0)
    }
  }

  NextMethod("check_data")
}


############################################################################# !
# CHECK_FORMULA S3 methods                                                ####
############################################################################# !

#' @export
check_formula.dd_choice <- function(model, data, formula) {
  NextMethod("check_formula")
}


############################################################################# !
# BMF2BF S3 methods                                                       ####
############################################################################# !

#' @export
bmf2bf.dd_choice <- function(model, formula) {
  choice_col  <- model$resp_vars$choice_col
  options     <- model$other_vars$options
  discount_fn <- model$other_vars$discount_fn
  choice_rule <- model$other_vars$choice_rule
  opt_names   <- names(options)

  # Internal log-value variable names (e.g. "lVLL", "lVSS")
  lv_names <- paste0("lV", opt_names)

  # NLF expressions for log-discounted values per option
  lv_nlfs <- mapply(
    function(lv_nm, opt) {
      amt_col   <- options[[opt]]["amt"]
      delay_col <- options[[opt]]["delay"]
      glue_nlf(.dd_lv_expr(discount_fn, lv_nm, amt_col, delay_col))
    },
    lv_names, opt_names,
    SIMPLIFY = FALSE
  )

  # NLF expression for mu (linear predictor before logistic link)
  mu_nlf <- glue_nlf(.dd_mu_expr(choice_rule, lv_names))

  # Base brms formula: response ~ mu (nl = TRUE handled by adding nlf terms)
  brms_formula <- brms::bf(
    stats::as.formula(glue::glue("{choice_col} ~ mu")),
    nl = TRUE
  )

  # Add internal NLF terms
  brms_formula <- brms_formula + mu_nlf
  brms_formula <- Reduce(`+`, lv_nlfs, init = brms_formula)

  brms_formula
}


############################################################################# !
# CONFIGURE_MODEL S3 METHODS                                              ####
############################################################################# !

#' @export
configure_model.dd_choice <- function(model, data, formula) {
  formula <- bmf2bf(model, formula)
  formula$family <- brms::bernoulli(link = "logit")
  nlist(formula, data)
}
