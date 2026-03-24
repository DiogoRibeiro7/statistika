! ==========================================================================
! Statistical distribution CDF/PDF functions.
!
! High-performance Fortran implementations of distribution functions
! built on top of the special_functions module.
! ==========================================================================

module distributions_mod
  use iso_c_binding
  use special_functions
  implicit none

  integer, parameter :: ddp = c_double

contains

  ! =========================================================================
  ! Chi-squared CDF: P(X <= x) for X ~ chi^2(df)
  ! = regularized_gamma_p(df/2, x/2)
  ! =========================================================================
  pure function chi2_cdf(x, df) result(res)
    real(ddp), intent(in) :: x, df
    real(ddp) :: res

    if (x <= 0.0_ddp) then
      res = 0.0_ddp
    else
      res = f_regularized_gamma_p(df / 2.0_ddp, x / 2.0_ddp)
    end if
  end function chi2_cdf

  ! =========================================================================
  ! Chi-squared PDF: f(x) = x^(k/2-1) * exp(-x/2) / (2^(k/2) * Gamma(k/2))
  ! =========================================================================
  pure function chi2_pdf(x, df) result(res)
    real(ddp), intent(in) :: x, df
    real(ddp) :: res, half_df

    if (x <= 0.0_ddp) then
      res = 0.0_ddp
      return
    end if

    half_df = df / 2.0_ddp
    res = exp((half_df - 1.0_ddp) * log(x) - x / 2.0_ddp - &
              half_df * log(2.0_ddp) - f_gamma_ln(half_df))
  end function chi2_pdf

  ! =========================================================================
  ! Student's t CDF: P(X <= x) for X ~ t(df)
  ! Uses the regularized incomplete beta function.
  ! =========================================================================
  pure function t_cdf(x, df) result(res)
    real(ddp), intent(in) :: x, df
    real(ddp) :: res, t2, u

    t2 = x * x
    u = df / (df + t2)

    if (x >= 0.0_ddp) then
      res = 1.0_ddp - 0.5_ddp * f_regularized_beta(u, df / 2.0_ddp, 0.5_ddp)
    else
      res = 0.5_ddp * f_regularized_beta(u, df / 2.0_ddp, 0.5_ddp)
    end if
  end function t_cdf

  ! =========================================================================
  ! Student's t PDF
  ! f(x) = Gamma((df+1)/2) / (sqrt(df*pi) * Gamma(df/2)) * (1 + x^2/df)^(-(df+1)/2)
  ! =========================================================================
  pure function t_pdf(x, df) result(res)
    real(ddp), intent(in) :: x, df
    real(ddp) :: res, ln_coeff

    ln_coeff = f_gamma_ln((df + 1.0_ddp) / 2.0_ddp) - &
               f_gamma_ln(df / 2.0_ddp) - &
               0.5_ddp * log(df * PI)
    res = exp(ln_coeff - (df + 1.0_ddp) / 2.0_ddp * log(1.0_ddp + x * x / df))
  end function t_pdf

  ! =========================================================================
  ! F-distribution CDF: P(X <= x) for X ~ F(d1, d2)
  ! Uses the regularized incomplete beta function.
  ! =========================================================================
  pure function f_cdf(x, d1, d2) result(res)
    real(ddp), intent(in) :: x, d1, d2
    real(ddp) :: res

    if (x <= 0.0_ddp) then
      res = 0.0_ddp
    else
      res = f_regularized_beta(d1 * x / (d1 * x + d2), d1 / 2.0_ddp, d2 / 2.0_ddp)
    end if
  end function f_cdf

  ! =========================================================================
  ! F-distribution PDF
  ! =========================================================================
  pure function f_pdf(x, d1, d2) result(res)
    real(ddp), intent(in) :: x, d1, d2
    real(ddp) :: res, ln_coeff

    if (x <= 0.0_ddp) then
      res = 0.0_ddp
      return
    end if

    ln_coeff = (d1 / 2.0_ddp) * log(d1 / d2) + &
               (d1 / 2.0_ddp - 1.0_ddp) * log(x) - &
               ((d1 + d2) / 2.0_ddp) * log(1.0_ddp + d1 * x / d2) - &
               f_gamma_ln(d1 / 2.0_ddp) - f_gamma_ln(d2 / 2.0_ddp) + &
               f_gamma_ln((d1 + d2) / 2.0_ddp)
    res = exp(ln_coeff)
  end function f_pdf

  ! =========================================================================
  ! Standard normal CDF: Phi(x) = 0.5 * erfc(-x / sqrt(2))
  ! =========================================================================
  pure function normal_cdf(x) result(res)
    real(ddp), intent(in) :: x
    real(ddp) :: res
    res = 0.5_ddp * f_erfc(-x / sqrt(2.0_ddp))
  end function normal_cdf

  ! =========================================================================
  ! Standard normal PDF
  ! =========================================================================
  pure function normal_pdf(x) result(res)
    real(ddp), intent(in) :: x
    real(ddp) :: res
    res = exp(-0.5_ddp * x * x) / SQRT_2PI
  end function normal_pdf

  ! =========================================================================
  ! Gamma distribution CDF: P(X <= x) for X ~ Gamma(shape, scale)
  ! = regularized_gamma_p(shape, x/scale)
  ! =========================================================================
  pure function gamma_cdf(x, shape, scale) result(res)
    real(ddp), intent(in) :: x, shape, scale
    real(ddp) :: res

    if (x <= 0.0_ddp) then
      res = 0.0_ddp
    else
      res = f_regularized_gamma_p(shape, x / scale)
    end if
  end function gamma_cdf

  ! =========================================================================
  ! Beta distribution CDF: P(X <= x) for X ~ Beta(a, b)
  ! = regularized_beta(x, a, b)
  ! =========================================================================
  pure function beta_cdf(x, a, b) result(res)
    real(ddp), intent(in) :: x, a, b
    real(ddp) :: res

    if (x <= 0.0_ddp) then
      res = 0.0_ddp
    else if (x >= 1.0_ddp) then
      res = 1.0_ddp
    else
      res = f_regularized_beta(x, a, b)
    end if
  end function beta_cdf

  ! =========================================================================
  ! Poisson CDF: P(X <= k) for X ~ Poisson(lambda)
  ! = 1 - regularized_gamma_p(k+1, lambda)
  ! =========================================================================
  pure function poisson_cdf(k, lambda) result(res)
    integer, intent(in) :: k
    real(ddp), intent(in) :: lambda
    real(ddp) :: res

    if (k < 0) then
      res = 0.0_ddp
    else
      res = 1.0_ddp - f_regularized_gamma_p(real(k + 1, ddp), lambda)
    end if
  end function poisson_cdf

end module distributions_mod

! ===========================================================================
! C-callable wrappers
! ===========================================================================

subroutine c_chi2_cdf(x, df, result) bind(C, name="fortran_chi2_cdf")
  use iso_c_binding
  use distributions_mod
  real(c_double), intent(in), value :: x, df
  real(c_double), intent(out) :: result
  result = chi2_cdf(x, df)
end subroutine

subroutine c_chi2_pdf(x, df, result) bind(C, name="fortran_chi2_pdf")
  use iso_c_binding
  use distributions_mod
  real(c_double), intent(in), value :: x, df
  real(c_double), intent(out) :: result
  result = chi2_pdf(x, df)
end subroutine

subroutine c_t_cdf(x, df, result) bind(C, name="fortran_t_cdf")
  use iso_c_binding
  use distributions_mod
  real(c_double), intent(in), value :: x, df
  real(c_double), intent(out) :: result
  result = t_cdf(x, df)
end subroutine

subroutine c_t_pdf(x, df, result) bind(C, name="fortran_t_pdf")
  use iso_c_binding
  use distributions_mod
  real(c_double), intent(in), value :: x, df
  real(c_double), intent(out) :: result
  result = t_pdf(x, df)
end subroutine

subroutine c_f_cdf(x, d1, d2, result) bind(C, name="fortran_f_cdf")
  use iso_c_binding
  use distributions_mod
  real(c_double), intent(in), value :: x, d1, d2
  real(c_double), intent(out) :: result
  result = f_cdf(x, d1, d2)
end subroutine

subroutine c_f_pdf(x, d1, d2, result) bind(C, name="fortran_f_pdf")
  use iso_c_binding
  use distributions_mod
  real(c_double), intent(in), value :: x, d1, d2
  real(c_double), intent(out) :: result
  result = f_pdf(x, d1, d2)
end subroutine

subroutine c_normal_cdf_dist(x, result) bind(C, name="fortran_normal_cdf_dist")
  use iso_c_binding
  use distributions_mod
  real(c_double), intent(in), value :: x
  real(c_double), intent(out) :: result
  result = normal_cdf(x)
end subroutine

subroutine c_normal_pdf(x, result) bind(C, name="fortran_normal_pdf")
  use iso_c_binding
  use distributions_mod
  real(c_double), intent(in), value :: x
  real(c_double), intent(out) :: result
  result = normal_pdf(x)
end subroutine

subroutine c_gamma_cdf(x, shape, scale, result) bind(C, name="fortran_gamma_cdf")
  use iso_c_binding
  use distributions_mod
  real(c_double), intent(in), value :: x, shape, scale
  real(c_double), intent(out) :: result
  result = gamma_cdf(x, shape, scale)
end subroutine

subroutine c_beta_cdf(x, a, b, result) bind(C, name="fortran_beta_cdf")
  use iso_c_binding
  use distributions_mod
  real(c_double), intent(in), value :: x, a, b
  real(c_double), intent(out) :: result
  result = beta_cdf(x, a, b)
end subroutine

! --------------------------------------------------------------------------
! Batch CDF evaluations for vectorized performance
! --------------------------------------------------------------------------

subroutine c_chi2_cdf_batch(x, pn, df, result) bind(C, name="fortran_chi2_cdf_batch")
  use iso_c_binding
  use distributions_mod
  real(c_double), intent(in) :: x(*)
  integer(c_int), intent(in) :: pn
  real(c_double), intent(in), value :: df
  real(c_double), intent(out) :: result(*)
  integer :: n, i
  n = pn
  do i = 1, n
    result(i) = chi2_cdf(x(i), df)
  end do
end subroutine

subroutine c_t_cdf_batch(x, pn, df, result) bind(C, name="fortran_t_cdf_batch")
  use iso_c_binding
  use distributions_mod
  real(c_double), intent(in) :: x(*)
  integer(c_int), intent(in) :: pn
  real(c_double), intent(in), value :: df
  real(c_double), intent(out) :: result(*)
  integer :: n, i
  n = pn
  do i = 1, n
    result(i) = t_cdf(x(i), df)
  end do
end subroutine

subroutine c_normal_cdf_batch(x, pn, result) bind(C, name="fortran_normal_cdf_batch")
  use iso_c_binding
  use distributions_mod
  real(c_double), intent(in) :: x(*)
  integer(c_int), intent(in) :: pn
  real(c_double), intent(out) :: result(*)
  integer :: n, i
  n = pn
  do i = 1, n
    result(i) = normal_cdf(x(i))
  end do
end subroutine
