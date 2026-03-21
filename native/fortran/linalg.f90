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
! Standard normal CDF via erfc: Phi(x) = 0.5 * erfc(-x / sqrt(2))
! --------------------------------------------------------------------------
subroutine c_normal_cdf(x, result) bind(C, name="fortran_normal_cdf")
  use iso_c_binding
  implicit none
  real(c_double), intent(in), value :: x
  real(c_double), intent(out) :: result

  result = 0.5d0 * erfc(-x / sqrt(2.0d0))
end subroutine
