! ==========================================================================
! Time series analysis kernels.
!
! High-performance Fortran implementations for autocorrelation (ACF),
! partial autocorrelation (PACF via Durbin-Levinson), differencing,
! and exponential smoothing.
! ==========================================================================

! --------------------------------------------------------------------------
! Autocovariance function: gamma(k) for k = 0..maxLag
!   series: n observations
!   mu: precomputed mean
!   gamma_out: (maxLag+1) output autocovariances
! --------------------------------------------------------------------------
subroutine c_autocovariance(series, pn, mu, gamma_out, pmaxlag) &
    bind(C, name="fortran_autocovariance")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: series(*)
  integer(c_int), intent(in)  :: pn, pmaxlag
  real(c_double), intent(in), value :: mu
  real(c_double), intent(out) :: gamma_out(*)

  integer :: n, maxlag, k, t
  real(c_double) :: s

  n = pn
  maxlag = pmaxlag

  do k = 0, maxlag
    s = 0.0d0
    do t = 1, n - k
      s = s + (series(t) - mu) * (series(t + k) - mu)
    end do
    gamma_out(k + 1) = s / dble(n)
  end do
end subroutine

! --------------------------------------------------------------------------
! ACF: r(k) = gamma(k) / gamma(0) for k = 0..maxLag
! --------------------------------------------------------------------------
subroutine c_acf(series, pn, acf_out, pmaxlag) &
    bind(C, name="fortran_acf")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: series(*)
  integer(c_int), intent(in)  :: pn, pmaxlag
  real(c_double), intent(out) :: acf_out(*)

  integer :: n, maxlag, k, t
  real(c_double) :: mu, s, gamma0

  n = pn
  maxlag = pmaxlag

  ! Compute mean
  mu = 0.0d0
  do t = 1, n
    mu = mu + series(t)
  end do
  mu = mu / dble(n)

  ! Compute gamma(0)
  gamma0 = 0.0d0
  do t = 1, n
    gamma0 = gamma0 + (series(t) - mu) ** 2
  end do
  gamma0 = gamma0 / dble(n)

  if (gamma0 == 0.0d0) then
    do k = 0, maxlag
      acf_out(k + 1) = 0.0d0
    end do
    return
  end if

  ! Compute ACF for each lag
  do k = 0, maxlag
    s = 0.0d0
    do t = 1, n - k
      s = s + (series(t) - mu) * (series(t + k) - mu)
    end do
    acf_out(k + 1) = (s / dble(n)) / gamma0
  end do
end subroutine

! --------------------------------------------------------------------------
! PACF via Durbin-Levinson recursion.
!   acf_in: ACF values for lags 0..maxLag (length maxLag+1)
!   pacf_out: PACF values for lags 0..maxLag (length maxLag+1)
! --------------------------------------------------------------------------
subroutine c_pacf_durbin_levinson(acf_in, pacf_out, pmaxlag) &
    bind(C, name="fortran_pacf_durbin_levinson")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: acf_in(*)
  real(c_double), intent(out) :: pacf_out(*)
  integer(c_int), intent(in)  :: pmaxlag

  integer :: maxlag, k, j
  real(c_double) :: num, den
  real(c_double), allocatable :: phi_prev(:), phi_curr(:)

  maxlag = pmaxlag

  pacf_out(1) = 1.0d0  ! lag 0

  if (maxlag < 1) return

  allocate(phi_prev(maxlag))
  allocate(phi_curr(maxlag))

  ! lag 1
  phi_prev(1) = acf_in(2)  ! acf[1]
  pacf_out(2) = acf_in(2)

  do k = 2, maxlag
    num = acf_in(k + 1)  ! acf[k]
    den = 1.0d0
    do j = 1, k - 1
      num = num - phi_prev(j) * acf_in(k - j + 1)
      den = den - phi_prev(j) * acf_in(j + 1)
    end do

    if (den == 0.0d0) then
      phi_curr(k) = 0.0d0
    else
      phi_curr(k) = num / den
    end if

    ! Update coefficients
    do j = 1, k - 1
      phi_curr(j) = phi_prev(j) - phi_curr(k) * phi_prev(k - j)
    end do

    pacf_out(k + 1) = phi_curr(k)

    ! Swap for next iteration
    do j = 1, k
      phi_prev(j) = phi_curr(j)
    end do
  end do

  deallocate(phi_prev)
  deallocate(phi_curr)
end subroutine

! --------------------------------------------------------------------------
! Exponential smoothing (Simple Exponential Smoothing).
!   series: n observations
!   alpha: smoothing parameter (0 < alpha <= 1)
!   smoothed: n output smoothed values
! --------------------------------------------------------------------------
subroutine c_exponential_smoothing(series, pn, alpha, smoothed) &
    bind(C, name="fortran_exponential_smoothing")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: series(*)
  integer(c_int), intent(in)  :: pn
  real(c_double), intent(in), value :: alpha
  real(c_double), intent(out) :: smoothed(*)

  integer :: n, t

  n = pn
  smoothed(1) = series(1)

  do t = 2, n
    smoothed(t) = alpha * series(t) + (1.0d0 - alpha) * smoothed(t - 1)
  end do
end subroutine

! --------------------------------------------------------------------------
! Holt-Winters double exponential smoothing.
!   series: n observations
!   alpha: level smoothing (0 < alpha <= 1)
!   beta: trend smoothing (0 < beta <= 1)
!   level_out: n output level values
!   trend_out: n output trend values
!   fitted_out: n output fitted values
! --------------------------------------------------------------------------
subroutine c_holt_winters(series, pn, alpha, beta, level_out, trend_out, fitted_out) &
    bind(C, name="fortran_holt_winters")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: series(*)
  integer(c_int), intent(in)  :: pn
  real(c_double), intent(in), value :: alpha, beta
  real(c_double), intent(out) :: level_out(*), trend_out(*), fitted_out(*)

  integer :: n, t

  n = pn

  ! Initialize
  level_out(1) = series(1)
  trend_out(1) = 0.0d0
  if (n >= 2) trend_out(1) = series(2) - series(1)
  fitted_out(1) = series(1)

  do t = 2, n
    fitted_out(t) = level_out(t - 1) + trend_out(t - 1)
    level_out(t) = alpha * series(t) + (1.0d0 - alpha) * fitted_out(t)
    trend_out(t) = beta * (level_out(t) - level_out(t - 1)) + (1.0d0 - beta) * trend_out(t - 1)
  end do
end subroutine

! --------------------------------------------------------------------------
! Differencing: compute d-th order differences.
!   series: n observations
!   d: order of differencing
!   result: (n - d) output differences
! --------------------------------------------------------------------------
subroutine c_difference(series, pn, pd, result) &
    bind(C, name="fortran_difference")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: series(*)
  integer(c_int), intent(in)  :: pn, pd
  real(c_double), intent(out) :: result(*)

  integer :: n, d, i, j, curr_len
  real(c_double), allocatable :: buf1(:), buf2(:)

  n = pn
  d = pd

  if (d <= 0 .or. n <= d) return

  allocate(buf1(n))
  allocate(buf2(n))

  ! Copy input
  do i = 1, n
    buf1(i) = series(i)
  end do
  curr_len = n

  ! Apply differencing d times
  do j = 1, d
    do i = 1, curr_len - 1
      buf2(i) = buf1(i + 1) - buf1(i)
    end do
    curr_len = curr_len - 1
    do i = 1, curr_len
      buf1(i) = buf2(i)
    end do
  end do

  ! Copy result
  do i = 1, curr_len
    result(i) = buf1(i)
  end do

  deallocate(buf1)
  deallocate(buf2)
end subroutine
