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

! --------------------------------------------------------------------------
! Coordinate descent for Elastic Net / LASSO regression.
!
!   Solves  min (1/2n)||y - X*beta||^2 + lambda*[alpha*||beta||_1 + (1-alpha)/2*||beta||_2^2]
!
!   X: n x p standardised design matrix (column-major flat)
!   yc: n centred response vector (unused, residuals maintained externally)
!   beta: p coefficient vector (in/out, warm start)
!   residuals: n residual vector (in/out, must match y - X*beta on entry)
!   col_norms: p precomputed column norms (X_j^T X_j)
!   lambda: regularisation strength
!   alpha: L1/L2 mixing (1 = LASSO, 0 = Ridge)
!   max_iter: maximum iterations
!   tol: convergence tolerance
!   n, p: dimensions
!   iters_out: actual iterations performed (out)
! --------------------------------------------------------------------------
subroutine c_elastic_net_cd(X, yc, beta, residuals, col_norms, &
    lambda, alpha, max_iter, tol, pn, pp, iters_out) &
    bind(C, name="fortran_elastic_net_cd")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)    :: X(*)
  real(c_double), intent(in)    :: yc(*)
  real(c_double), intent(inout) :: beta(*)
  real(c_double), intent(inout) :: residuals(*)
  real(c_double), intent(in)    :: col_norms(*)
  real(c_double), intent(in), value :: lambda, alpha, tol
  integer(c_int), intent(in), value :: max_iter
  integer(c_int), intent(in)    :: pn, pp
  integer(c_int), intent(out)   :: iters_out

  integer :: n, p, iter, j, i, idx
  real(c_double) :: old_beta, rho, lambda_l1, lambda_l2, new_beta, diff, max_change

  n = pn
  p = pp
  lambda_l1 = lambda * alpha * dble(n)
  lambda_l2 = lambda * (1.0d0 - alpha) * dble(n)

  do iter = 1, max_iter
    iters_out = iter
    max_change = 0.0d0

    do j = 1, p
      old_beta = beta(j)

      ! Compute partial residual dot product: rho = X_j^T (r + X_j * old_beta)
      rho = 0.0d0
      do i = 1, n
        idx = (j - 1) * n + i   ! column-major X[i, j]
        rho = rho + X(idx) * (residuals(i) + X(idx) * old_beta)
      end do

      ! Soft-thresholding
      if (rho > lambda_l1) then
        new_beta = (rho - lambda_l1) / (col_norms(j) + lambda_l2)
      else if (rho < -lambda_l1) then
        new_beta = (rho + lambda_l1) / (col_norms(j) + lambda_l2)
      else
        new_beta = 0.0d0
      end if

      if (new_beta /= old_beta) then
        diff = new_beta - old_beta
        ! Update residuals
        do i = 1, n
          idx = (j - 1) * n + i
          residuals(i) = residuals(i) - X(idx) * diff
        end do
        beta(j) = new_beta
        if (abs(diff) > max_change) max_change = abs(diff)
      end if
    end do

    if (max_change < tol) return
  end do
end subroutine

! --------------------------------------------------------------------------
! Ridge regression: solve (X^T X + lambda*I) beta = X^T y
!
!   X: n x p standardised design matrix (column-major flat)
!   yc: n centred response vector
!   beta_out: p coefficient vector (out)
!   lambda: regularisation strength
!   n, p: dimensions
!   info: DGESV info (out)
!
! Uses LAPACK DGESV for the p x p system.
! --------------------------------------------------------------------------
subroutine c_ridge_solve(X, yc, beta_out, lambda, pn, pp, info) &
    bind(C, name="fortran_ridge_solve")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: X(*)
  real(c_double), intent(in)  :: yc(*)
  real(c_double), intent(out) :: beta_out(*)
  real(c_double), intent(in), value :: lambda
  integer(c_int), intent(in)  :: pn, pp
  integer(c_int), intent(out) :: info

  integer :: n, p, i, j, k, idx_ij, idx_ik
  real(c_double), allocatable :: XtX(:), Xty(:)
  integer, allocatable :: ipiv(:)

  n = pn
  p = pp

  allocate(XtX(p * p))
  allocate(Xty(p))
  allocate(ipiv(p))

  ! Compute X^T X (column-major p x p)
  do j = 1, p
    do k = 1, p
      XtX((k - 1) * p + j) = 0.0d0
      do i = 1, n
        idx_ij = (j - 1) * n + i
        idx_ik = (k - 1) * n + i
        XtX((k - 1) * p + j) = XtX((k - 1) * p + j) + X(idx_ij) * X(idx_ik)
      end do
    end do
    ! Add lambda to diagonal
    XtX((j - 1) * p + j) = XtX((j - 1) * p + j) + lambda
  end do

  ! Compute X^T y
  do j = 1, p
    Xty(j) = 0.0d0
    do i = 1, n
      idx_ij = (j - 1) * n + i
      Xty(j) = Xty(j) + X(idx_ij) * yc(i)
    end do
    beta_out(j) = Xty(j)
  end do

  ! Solve (X^T X + lambda*I) beta = X^T y via LAPACK DGESV
  call dgesv(p, 1, XtX, p, ipiv, beta_out, p, info)

  deallocate(XtX, Xty, ipiv)
end subroutine
