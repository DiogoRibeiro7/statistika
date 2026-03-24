! ==========================================================================
! Optimization and variance model kernels.
!
! High-performance Fortran implementations for:
!   - GARCH(p,q) conditional variance recursion and log-likelihood
!   - Nelder-Mead simplex optimization (for small-dimensional problems)
!   - Newton-Raphson iteration helpers
! ==========================================================================

! --------------------------------------------------------------------------
! GARCH(1,1) conditional variance recursion and Gaussian log-likelihood.
!
!   eps: T residuals (mean-subtracted returns)
!   omega: intercept (must be > 0)
!   alpha1: ARCH coefficient (>= 0)
!   beta1: GARCH coefficient (>= 0)
!   sigma2_out: T conditional variances (out)
!   loglik_out: total Gaussian log-likelihood (out)
!   T: number of observations
! --------------------------------------------------------------------------
subroutine c_garch11_loglik(eps, pT, omega, alpha1, beta1, &
    sigma2_out, loglik_out) &
    bind(C, name="fortran_garch11_loglik")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: eps(*)
  integer(c_int), intent(in)  :: pT
  real(c_double), intent(in), value :: omega, alpha1, beta1
  real(c_double), intent(out) :: sigma2_out(*)
  real(c_double), intent(out) :: loglik_out

  integer :: T, t
  real(c_double) :: ll, s2
  real(c_double), parameter :: LOG2PI = 1.8378770664093453d0

  T = pT

  ! Initialize with unconditional variance
  if (alpha1 + beta1 < 1.0d0) then
    s2 = omega / (1.0d0 - alpha1 - beta1)
  else
    ! Fallback: sample variance
    s2 = 0.0d0
    do t = 1, T
      s2 = s2 + eps(t) * eps(t)
    end do
    s2 = s2 / dble(T)
  end if

  sigma2_out(1) = s2
  ll = -0.5d0 * (LOG2PI + log(max(s2, 1.0d-300)) + eps(1) * eps(1) / max(s2, 1.0d-300))

  do t = 2, T
    s2 = omega + alpha1 * eps(t-1) * eps(t-1) + beta1 * sigma2_out(t-1)
    s2 = max(s2, 1.0d-12)
    sigma2_out(t) = s2
    ll = ll - 0.5d0 * (LOG2PI + log(s2) + eps(t) * eps(t) / s2)
  end do

  loglik_out = ll
end subroutine

! --------------------------------------------------------------------------
! General GARCH(p,q) conditional variance recursion and log-likelihood.
!
!   eps: T residuals
!   omega: intercept
!   alpha: q ARCH coefficients
!   beta: p GARCH coefficients
!   sigma2_out: T conditional variances (out)
!   loglik_out: total log-likelihood (out)
!   T: number of observations
!   p: GARCH order
!   q: ARCH order
! --------------------------------------------------------------------------
subroutine c_garch_pq_loglik(eps, pT, omega, alpha, beta, &
    sigma2_out, loglik_out, pp, pq) &
    bind(C, name="fortran_garch_pq_loglik")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: eps(*), alpha(*), beta(*)
  integer(c_int), intent(in)  :: pT, pp, pq
  real(c_double), intent(in), value :: omega
  real(c_double), intent(out) :: sigma2_out(*)
  real(c_double), intent(out) :: loglik_out

  integer :: T, p_ord, q_ord, t, j, maxpq
  real(c_double) :: ll, s2, sum_ab, init_var
  real(c_double), parameter :: LOG2PI = 1.8378770664093453d0

  T = pT
  p_ord = pp
  q_ord = pq
  maxpq = max(p_ord, q_ord)

  ! Compute sum of alpha + beta for stationarity check
  sum_ab = 0.0d0
  do j = 1, q_ord
    sum_ab = sum_ab + alpha(j)
  end do
  do j = 1, p_ord
    sum_ab = sum_ab + beta(j)
  end do

  ! Initial variance
  if (sum_ab < 1.0d0) then
    init_var = omega / (1.0d0 - sum_ab)
  else
    init_var = 0.0d0
    do t = 1, T
      init_var = init_var + eps(t) * eps(t)
    end do
    init_var = init_var / dble(T)
  end if

  ! Initialize first maxpq variances
  do t = 1, min(maxpq, T)
    sigma2_out(t) = init_var
  end do

  ll = 0.0d0
  do t = 1, min(maxpq, T)
    ll = ll - 0.5d0 * (LOG2PI + log(max(sigma2_out(t), 1.0d-300)) + &
         eps(t) * eps(t) / max(sigma2_out(t), 1.0d-300))
  end do

  ! Main recursion
  do t = maxpq + 1, T
    s2 = omega
    do j = 1, q_ord
      s2 = s2 + alpha(j) * eps(t - j) * eps(t - j)
    end do
    do j = 1, p_ord
      s2 = s2 + beta(j) * sigma2_out(t - j)
    end do
    s2 = max(s2, 1.0d-12)
    sigma2_out(t) = s2
    ll = ll - 0.5d0 * (LOG2PI + log(s2) + eps(t) * eps(t) / s2)
  end do

  loglik_out = ll
end subroutine

! --------------------------------------------------------------------------
! GJR-GARCH(1,1) conditional variance recursion and log-likelihood.
! Includes leverage effect for negative shocks.
!
!   eps: T residuals
!   omega: intercept
!   alpha1: ARCH coefficient
!   beta1: GARCH coefficient
!   gamma1: leverage coefficient
!   sigma2_out: T conditional variances (out)
!   loglik_out: total log-likelihood (out)
!   T: number of observations
! --------------------------------------------------------------------------
subroutine c_gjr_garch11_loglik(eps, pT, omega, alpha1, beta1, gamma1, &
    sigma2_out, loglik_out) &
    bind(C, name="fortran_gjr_garch11_loglik")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: eps(*)
  integer(c_int), intent(in)  :: pT
  real(c_double), intent(in), value :: omega, alpha1, beta1, gamma1
  real(c_double), intent(out) :: sigma2_out(*)
  real(c_double), intent(out) :: loglik_out

  integer :: T, t
  real(c_double) :: ll, s2, indicator
  real(c_double), parameter :: LOG2PI = 1.8378770664093453d0

  T = pT

  ! Initialize
  if (alpha1 + beta1 + 0.5d0 * gamma1 < 1.0d0) then
    s2 = omega / (1.0d0 - alpha1 - beta1 - 0.5d0 * gamma1)
  else
    s2 = 0.0d0
    do t = 1, T
      s2 = s2 + eps(t) * eps(t)
    end do
    s2 = s2 / dble(T)
  end if

  sigma2_out(1) = s2
  ll = -0.5d0 * (LOG2PI + log(max(s2, 1.0d-300)) + eps(1) * eps(1) / max(s2, 1.0d-300))

  do t = 2, T
    if (eps(t-1) < 0.0d0) then
      indicator = 1.0d0
    else
      indicator = 0.0d0
    end if
    s2 = omega + alpha1 * eps(t-1) * eps(t-1) + &
         gamma1 * indicator * eps(t-1) * eps(t-1) + &
         beta1 * sigma2_out(t-1)
    s2 = max(s2, 1.0d-12)
    sigma2_out(t) = s2
    ll = ll - 0.5d0 * (LOG2PI + log(s2) + eps(t) * eps(t) / s2)
  end do

  loglik_out = ll
end subroutine

! --------------------------------------------------------------------------
! EGARCH(1,1) log-variance recursion and log-likelihood.
!
!   eps: T residuals
!   omega: intercept (in log-variance space)
!   alpha1: magnitude coefficient
!   beta1: persistence coefficient
!   gamma1: leverage coefficient
!   sigma2_out: T conditional variances (out, NOT log-variances)
!   loglik_out: total log-likelihood (out)
!   T: number of observations
! --------------------------------------------------------------------------
subroutine c_egarch11_loglik(eps, pT, omega, alpha1, beta1, gamma1, &
    sigma2_out, loglik_out) &
    bind(C, name="fortran_egarch11_loglik")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: eps(*)
  integer(c_int), intent(in)  :: pT
  real(c_double), intent(in), value :: omega, alpha1, beta1, gamma1
  real(c_double), intent(out) :: sigma2_out(*)
  real(c_double), intent(out) :: loglik_out

  integer :: T, t
  real(c_double) :: ll, log_s2, z, s2
  real(c_double), parameter :: LOG2PI = 1.8378770664093453d0
  real(c_double), parameter :: SQRT_2_OVER_PI = 0.7978845608028654d0

  T = pT

  ! Initialize log-variance from unconditional
  if (abs(1.0d0 - beta1) > 1.0d-10) then
    log_s2 = omega / (1.0d0 - beta1)
  else
    log_s2 = 0.0d0
    do t = 1, T
      log_s2 = log_s2 + eps(t) * eps(t)
    end do
    log_s2 = log(log_s2 / dble(T))
  end if

  s2 = exp(log_s2)
  sigma2_out(1) = s2
  ll = -0.5d0 * (LOG2PI + log_s2 + eps(1) * eps(1) / max(s2, 1.0d-300))

  do t = 2, T
    z = eps(t-1) / sqrt(max(sigma2_out(t-1), 1.0d-300))
    log_s2 = omega + beta1 * log(max(sigma2_out(t-1), 1.0d-300)) + &
             alpha1 * (abs(z) - SQRT_2_OVER_PI) + gamma1 * z
    s2 = exp(log_s2)
    s2 = max(s2, 1.0d-300)
    sigma2_out(t) = s2
    ll = ll - 0.5d0 * (LOG2PI + log_s2 + eps(t) * eps(t) / s2)
  end do

  loglik_out = ll
end subroutine

! --------------------------------------------------------------------------
! Nelder-Mead simplex optimization for small-dimensional problems.
!
! Minimizes f(x) where x is n-dimensional.
! This is a standalone optimizer that takes an objective function
! evaluation array approach.
!
!   x0: n initial guess
!   step: n initial step sizes for each dimension
!   x_out: n optimized parameters (out)
!   fval_out: final objective value (out)
!   evals: objective function values at simplex vertices (workspace)
!   simplex: (n+1) x n simplex vertices (workspace)
!   obj_values: (n+1) pre-evaluated objective values
!   n: number of parameters
!   max_iter: maximum iterations
!   tol: convergence tolerance
!   iters_out: actual iterations used (out)
!
! Note: This routine manages the simplex geometry. The caller must
! evaluate the objective function at requested points.
! For integration with JS, we provide a simpler interface below.
! --------------------------------------------------------------------------

! --------------------------------------------------------------------------
! Simple 1D golden section search (for line searches).
!   Finds minimum of f on [a, b].
!   The caller passes function values at query points.
! --------------------------------------------------------------------------

! --------------------------------------------------------------------------
! GARCH(1,1) variance forecast.
!   Given last eps^2 and sigma^2, forecast h steps ahead.
! --------------------------------------------------------------------------
subroutine c_garch11_forecast(last_eps2, last_sigma2, omega, alpha1, beta1, &
    forecast_out, ph) &
    bind(C, name="fortran_garch11_forecast")
  use iso_c_binding
  implicit none
  real(c_double), intent(in), value :: last_eps2, last_sigma2, omega, alpha1, beta1
  real(c_double), intent(out) :: forecast_out(*)
  integer(c_int), intent(in)  :: ph

  integer :: h, t
  real(c_double) :: s2

  h = ph

  ! First step uses actual last values
  s2 = omega + alpha1 * last_eps2 + beta1 * last_sigma2
  forecast_out(1) = s2

  ! Subsequent steps: E[eps^2_{t+k}|F_t] = sigma^2_{t+k}
  do t = 2, h
    s2 = omega + (alpha1 + beta1) * forecast_out(t - 1)
    forecast_out(t) = s2
  end do
end subroutine
