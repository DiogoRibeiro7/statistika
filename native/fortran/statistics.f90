! ==========================================================================
! Numerical routines for statistics modules.
!
! High-performance Fortran implementations for computationally intensive
! operations used by distance, KDE, GMM, and streaming statistics modules.
! ==========================================================================

! --------------------------------------------------------------------------
! Pairwise Euclidean distance matrix.
!   data: n x p matrix (column-major flat)
!   dist: n x n output distance matrix (column-major flat)
! --------------------------------------------------------------------------
subroutine c_pairwise_euclidean(data, dist, pn, pp) &
    bind(C, name="fortran_pairwise_euclidean")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: data(*)
  real(c_double), intent(out) :: dist(*)
  integer(c_int), intent(in)  :: pn, pp

  integer :: n, p, i, j, k, idx_ij, idx_ji
  real(c_double) :: diff, d

  n = pn
  p = pp

  ! Initialize diagonal to 0
  do i = 1, n
    dist((i-1)*n + i) = 0.0d0
  end do

  ! Compute upper triangle and mirror
  do i = 1, n
    do j = i+1, n
      d = 0.0d0
      do k = 1, p
        ! column-major: data[(k-1)*n + i] is element (i, k)
        diff = data((k-1)*n + i) - data((k-1)*n + j)
        d = d + diff * diff
      end do
      d = sqrt(d)
      idx_ij = (j-1)*n + i  ! column-major (i, j)
      idx_ji = (i-1)*n + j  ! column-major (j, i)
      dist(idx_ij) = d
      dist(idx_ji) = d
    end do
  end do
end subroutine

! --------------------------------------------------------------------------
! Batch Gaussian PDF evaluation.
!   x: array of n evaluation points
!   mu: mean
!   sigma2: variance (must be > 0)
!   result: array of n PDF values
! --------------------------------------------------------------------------
subroutine c_gaussian_pdf_batch(x, mu, sigma2, result, pn) &
    bind(C, name="fortran_gaussian_pdf_batch")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: x(*)
  real(c_double), intent(in), value :: mu, sigma2
  real(c_double), intent(out) :: result(*)
  integer(c_int), intent(in)  :: pn

  integer :: n, i
  real(c_double) :: norm_factor, inv_2sigma2

  n = pn
  norm_factor = 1.0d0 / sqrt(2.0d0 * 3.14159265358979323846d0 * sigma2)
  inv_2sigma2 = -0.5d0 / sigma2

  do i = 1, n
    result(i) = norm_factor * exp(inv_2sigma2 * (x(i) - mu) ** 2)
  end do
end subroutine

! --------------------------------------------------------------------------
! Kernel density estimation (Gaussian kernel).
!   data: n input observations
!   eval_points: m evaluation points
!   bandwidth: smoothing bandwidth h
!   density: m output density values
! --------------------------------------------------------------------------
subroutine c_kde_gaussian(data, eval_points, density, pn, pm, bandwidth) &
    bind(C, name="fortran_kde_gaussian")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: data(*)
  real(c_double), intent(in)  :: eval_points(*)
  real(c_double), intent(out) :: density(*)
  integer(c_int), intent(in)  :: pn, pm
  real(c_double), intent(in), value :: bandwidth

  integer :: n, m, i, j
  real(c_double) :: norm_factor, inv_h, u, sum_val
  real(c_double), parameter :: SQRT_2PI = 2.5066282746310005d0

  n = pn
  m = pm
  inv_h = 1.0d0 / bandwidth
  norm_factor = inv_h / (dble(n) * SQRT_2PI)

  do i = 1, m
    sum_val = 0.0d0
    do j = 1, n
      u = (eval_points(i) - data(j)) * inv_h
      sum_val = sum_val + exp(-0.5d0 * u * u)
    end do
    density(i) = norm_factor * sum_val
  end do
end subroutine

! --------------------------------------------------------------------------
! Weighted sum of squares for IRLS (GLM inner loop).
!   X: n x cols design matrix (column-major flat)
!   W: n diagonal weights
!   z: n adjusted response
!   XtWX: cols x cols output (column-major flat)
!   XtWz: cols output
! --------------------------------------------------------------------------
subroutine c_weighted_cross_products(X, W, z, XtWX, XtWz, pn, pcols) &
    bind(C, name="fortran_weighted_cross_products")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: X(*)
  real(c_double), intent(in)  :: W(*)
  real(c_double), intent(in)  :: z(*)
  real(c_double), intent(out) :: XtWX(*)
  real(c_double), intent(out) :: XtWz(*)
  integer(c_int), intent(in)  :: pn, pcols

  integer :: n, cols, i, j, k, idx
  real(c_double) :: wi, xij, xik

  n = pn
  cols = pcols

  ! Zero output
  do i = 1, cols * cols
    XtWX(i) = 0.0d0
  end do
  do i = 1, cols
    XtWz(i) = 0.0d0
  end do

  ! Accumulate X^T W X and X^T W z
  do i = 1, n
    wi = W(i)
    do j = 1, cols
      xij = X((j-1)*n + i)  ! column-major X[i, j]
      XtWz(j) = XtWz(j) + xij * wi * z(i)
      do k = 1, cols
        xik = X((k-1)*n + i)  ! column-major X[i, k]
        idx = (k-1)*cols + j   ! column-major XtWX[j, k]
        XtWX(idx) = XtWX(idx) + xij * wi * xik
      end do
    end do
  end do
end subroutine

! --------------------------------------------------------------------------
! Running statistics update (Welford's algorithm, batch).
!   values: n new values to incorporate
!   count_in: current count (0 for fresh start)
!   mean_in: current mean
!   m2_in: current M2
!   count_out, mean_out, m2_out: updated values
!   min_out, max_out: updated min/max
! --------------------------------------------------------------------------
subroutine c_welford_batch(values, pn, count_in, mean_in, m2_in, min_in, max_in, &
    count_out, mean_out, m2_out, min_out, max_out) &
    bind(C, name="fortran_welford_batch")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: values(*)
  integer(c_int), intent(in)  :: pn
  integer(c_int), intent(in), value :: count_in
  real(c_double), intent(in), value :: mean_in, m2_in, min_in, max_in
  integer(c_int), intent(out) :: count_out
  real(c_double), intent(out) :: mean_out, m2_out, min_out, max_out

  integer :: n, i, cnt
  real(c_double) :: mu, m2, delta, delta2, v, vmin, vmax

  n = pn
  cnt = count_in
  mu = mean_in
  m2 = m2_in
  vmin = min_in
  vmax = max_in

  do i = 1, n
    v = values(i)
    cnt = cnt + 1
    delta = v - mu
    mu = mu + delta / dble(cnt)
    delta2 = v - mu
    m2 = m2 + delta * delta2
    if (v < vmin) vmin = v
    if (v > vmax) vmax = v
  end do

  count_out = cnt
  mean_out = mu
  m2_out = m2
  min_out = vmin
  max_out = vmax
end subroutine
