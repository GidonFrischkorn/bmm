############################################################################# !
# ATTR_CHOICE — shared S3 base for attribute-based choice models          ####
############################################################################# !
# All user-facing choice constructors (pt_choice, dd_choice, …) inherit from
# attr_choice so that common validation logic lives in one place.

############################################################################# !
# CHECK_DATA — shared attribute checks                                    ####
############################################################################# !

#' @export
check_data.attr_choice <- function(model, data, formula) {
  choice_col <- model$resp_vars$choice_col
  options    <- model$other_vars$options

  if (is.null(choice_col) || is.null(options)) {
    return(NextMethod("check_data"))
  }

  # ---- choice column -------------------------------------------------------
  stopif(
    !choice_col %in% colnames(data),
    "Choice column '{choice_col}' is missing from the data."
  )
  choice_vals <- data[[choice_col]]
  stopif(
    !all(choice_vals %in% c(0L, 1L, 0, 1, NA_real_), na.rm = FALSE),
    "Choice column '{choice_col}' must contain only 0 and 1."
  )

  # ---- option attribute columns --------------------------------------------
  opt_names <- names(options)
  all_attr_cols <- unname(unlist(options))
  missing_cols <- setdiff(all_attr_cols, colnames(data))
  stopif(
    length(missing_cols) > 0,
    "The following option attribute columns are missing from the data: {collapse_comma(missing_cols)}"
  )

  # ---- amounts must be positive --------------------------------------------
  for (opt in opt_names) {
    amt_col <- options[[opt]]["amt"]
    stopif(
      any(data[[amt_col]] <= 0, na.rm = TRUE),
      "Amount column '{amt_col}' must contain only positive values."
    )
  }

  NextMethod("check_data")
}
