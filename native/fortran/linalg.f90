! ==========================================================================
! Linear algebra operations backed by LAPACK/BLAS.
!
! All matrices passed as flat column-major arrays.
! All integer dimensions passed by reference (standard Fortran convention).
! ==========================================================================

! --------------------------------------------------------------------------
! Matrix multiplication: C = A * B
!   A is m x k, B is k x n, C is m x n
! --------------------------------------------------------------------------
subroutine c_mat_mul(a, b, c, pm, pk, pn) bind(C, name="fortran_mat_mul")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: a(*)
  real(c_double), intent(in)  :: b(*)
  real(c_double), intent(out) :: c(*)
  integer(c_int), intent(in)  :: pm, pk, pn

  integer :: m, k, n
  m = pm; k = pk; n = pn

  call dgemm('N', 'N', m, n, k, 1.0d0, a, m, b, k, 0.0d0, c, m)
end subroutine

! --------------------------------------------------------------------------
! Solve linear system A*x = b via LU factorization (DGESV).
! --------------------------------------------------------------------------
subroutine c_solve(a, b, x, pn, info) bind(C, name="fortran_solve")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: a(*)
  real(c_double), intent(in)  :: b(*)
  real(c_double), intent(out) :: x(*)
  integer(c_int), intent(in)  :: pn
  integer(c_int), intent(out) :: info

  integer :: n, i
  real(c_double), allocatable :: a_copy(:)
  integer, allocatable :: ipiv(:)

  n = pn
  allocate(a_copy(n * n))
  allocate(ipiv(n))

  do i = 1, n * n
    a_copy(i) = a(i)
  end do
  do i = 1, n
    x(i) = b(i)
  end do

  call dgesv(n, 1, a_copy, n, ipiv, x, n, info)

  deallocate(a_copy)
  deallocate(ipiv)
end subroutine

! --------------------------------------------------------------------------
! Invert a square matrix via LU factorization (DGETRF + DGETRI).
! --------------------------------------------------------------------------
subroutine c_invert(a, inv, pn, info) bind(C, name="fortran_invert")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: a(*)
  real(c_double), intent(out) :: inv(*)
  integer(c_int), intent(in)  :: pn
  integer(c_int), intent(out) :: info

  integer :: n, i, lwork, info2
  integer, allocatable :: ipiv(:)
  real(c_double), allocatable :: work(:)

  n = pn
  allocate(ipiv(n))

  do i = 1, n * n
    inv(i) = a(i)
  end do

  call dgetrf(n, n, inv, n, ipiv, info)
  if (info /= 0) then
    deallocate(ipiv)
    return
  end if

  lwork = n * 64
  allocate(work(lwork))

  call dgetri(n, inv, n, ipiv, work, lwork, info2)
  if (info2 /= 0) info = info2

  deallocate(work)
  deallocate(ipiv)
end subroutine

! --------------------------------------------------------------------------
! Symmetric eigenvalue decomposition (DSYEV).
!   eigenvalues in ascending order, eigenvectors as columns.
! --------------------------------------------------------------------------
subroutine c_sym_eigen(a, eigenvalues, eigenvectors, pn, info) &
    bind(C, name="fortran_sym_eigen")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: a(*)
  real(c_double), intent(out) :: eigenvalues(*)
  real(c_double), intent(out) :: eigenvectors(*)
  integer(c_int), intent(in)  :: pn
  integer(c_int), intent(out) :: info

  integer :: n, i, lwork
  real(c_double), allocatable :: work(:)

  n = pn

  do i = 1, n * n
    eigenvectors(i) = a(i)
  end do

  lwork = n * 64
  allocate(work(lwork))

  call dsyev('V', 'U', n, eigenvectors, n, eigenvalues, work, lwork, info)

  deallocate(work)
end subroutine

! --------------------------------------------------------------------------
! LU factorization with partial pivoting (DGETRF).
! Returns L (unit lower), U (upper), and pivot indices.
! L and U are packed into a single n*n output; caller separates them.
! --------------------------------------------------------------------------
subroutine c_lu(a, lu_out, ipiv_out, pn, info) bind(C, name="fortran_lu")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: a(*)
  real(c_double), intent(out) :: lu_out(*)
  integer(c_int), intent(out) :: ipiv_out(*)
  integer(c_int), intent(in)  :: pn
  integer(c_int), intent(out) :: info

  integer :: n, i

  n = pn

  do i = 1, n * n
    lu_out(i) = a(i)
  end do

  call dgetrf(n, n, lu_out, n, ipiv_out, info)
end subroutine

! --------------------------------------------------------------------------
! QR factorization (DGEQRF + DORGQR).
! A is m x n (m >= n). Returns Q (m x n thin) and R (n x n).
! --------------------------------------------------------------------------
subroutine c_qr(a, q_out, r_out, pm, pn, info) bind(C, name="fortran_qr")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: a(*)
  real(c_double), intent(out) :: q_out(*)
  real(c_double), intent(out) :: r_out(*)
  integer(c_int), intent(in)  :: pm, pn
  integer(c_int), intent(out) :: info

  integer :: m, n, lwork, i, j, info2
  real(c_double), allocatable :: work(:), tau(:), qr_buf(:)

  m = pm; n = pn

  allocate(tau(n))
  allocate(qr_buf(m * n))

  ! Copy A into working buffer (column-major)
  do i = 1, m * n
    qr_buf(i) = a(i)
  end do

  ! Workspace query
  lwork = max(m, n) * 64
  allocate(work(lwork))

  ! QR factorization: qr_buf now contains Householder vectors + R
  call dgeqrf(m, n, qr_buf, m, tau, work, lwork, info)
  if (info /= 0) then
    deallocate(work, tau, qr_buf)
    return
  end if

  ! Extract upper-triangular R (n x n) from qr_buf
  do j = 1, n
    do i = 1, n
      if (i <= j) then
        r_out((j - 1) * n + i) = qr_buf((j - 1) * m + i)
      else
        r_out((j - 1) * n + i) = 0.0d0
      end if
    end do
  end do

  ! Generate Q (m x n thin) from Householder vectors
  call dorgqr(m, n, n, qr_buf, m, tau, work, lwork, info2)
  if (info2 /= 0) then
    info = info2
    deallocate(work, tau, qr_buf)
    return
  end if

  ! Copy thin Q
  do i = 1, m * n
    q_out(i) = qr_buf(i)
  end do

  deallocate(work, tau, qr_buf)
end subroutine

! --------------------------------------------------------------------------
! Cholesky factorization for SPD matrices (DPOTRF).
! Returns lower-triangular factor L such that A = L * L^T.
! --------------------------------------------------------------------------
subroutine c_cholesky(a, l_out, pn, info) bind(C, name="fortran_cholesky")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: a(*)
  real(c_double), intent(out) :: l_out(*)
  integer(c_int), intent(in)  :: pn
  integer(c_int), intent(out) :: info

  integer :: n, i, j

  n = pn

  ! Copy A
  do i = 1, n * n
    l_out(i) = a(i)
  end do

  ! DPOTRF with 'L' for lower triangular
  call dpotrf('L', n, l_out, n, info)

  ! Zero the upper triangle (DPOTRF may leave garbage there)
  if (info == 0) then
    do j = 1, n
      do i = 1, j - 1
        l_out((j - 1) * n + i) = 0.0d0
      end do
    end do
  end if
end subroutine

! --------------------------------------------------------------------------
! Singular Value Decomposition (DGESVD).
! A is m x n. Returns U (m x k), S (k), V^T (k x n) where k = min(m, n).
! --------------------------------------------------------------------------
subroutine c_svd(a, u_out, s_out, vt_out, pm, pn, info) &
    bind(C, name="fortran_svd")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: a(*)
  real(c_double), intent(out) :: u_out(*)
  real(c_double), intent(out) :: s_out(*)
  real(c_double), intent(out) :: vt_out(*)
  integer(c_int), intent(in)  :: pm, pn
  integer(c_int), intent(out) :: info

  integer :: m, n, k, lwork, i
  real(c_double), allocatable :: work(:), a_copy(:)

  m = pm; n = pn
  k = min(m, n)

  allocate(a_copy(m * n))
  do i = 1, m * n
    a_copy(i) = a(i)
  end do

  ! Workspace query
  lwork = max(3 * k + max(m, n), 5 * k) * 2
  allocate(work(lwork))

  ! DGESVD: 'S' for thin U (m x k) and thin V^T (k x n)
  call dgesvd('S', 'S', m, n, a_copy, m, s_out, u_out, m, vt_out, k, &
              work, lwork, info)

  deallocate(work, a_copy)
end subroutine

! --------------------------------------------------------------------------
! Standard normal CDF via erfc: Phi(x) = 0.5 * erfc(-x / sqrt(2))
! --------------------------------------------------------------------------
subroutine c_normal_cdf(x, result) bind(C, name="fortran_normal_cdf")
  use iso_c_binding
  implicit none
  real(c_double), intent(in), value :: x
  real(c_double), intent(out) :: result

  result = 0.5d0 * erfc(-x / sqrt(2.0d0))
end subroutine
