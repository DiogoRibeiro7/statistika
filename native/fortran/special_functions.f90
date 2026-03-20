module special_functions
  use iso_c_binding
  implicit none

  integer, parameter :: dp = c_double
  real(dp), parameter :: PI = 3.14159265358979323846_dp
  real(dp), parameter :: EULER_MASCHERONI = 0.5772156649015329_dp
  real(dp), parameter :: SQRT_2PI = 2.5066282746310005_dp
  real(dp), parameter :: EPS = 1.0e-14_dp
  integer, parameter :: MAX_ITER = 200

  ! Lanczos coefficients (g=7, n=9)
  integer, parameter :: LANCZOS_N = 9
  real(dp), parameter :: LANCZOS_G = 7.0_dp
  real(dp), dimension(LANCZOS_N), parameter :: LANCZOS_C = (/ &
    0.99999999999980993_dp,   &
    676.5203681218851_dp,     &
    -1259.1392167224028_dp,   &
    771.32342877765313_dp,    &
    -176.61502916214059_dp,   &
    12.507343278686905_dp,    &
    -0.13857109526572012_dp,  &
    9.9843695780195716e-6_dp, &
    1.5056327351493116e-7_dp  &
  /)

contains

  ! =========================================================================
  ! Log-Gamma function (Lanczos approximation)
  ! =========================================================================
  pure recursive function f_gamma_ln(x) result(res)
    real(dp), intent(in) :: x
    real(dp) :: res
    real(dp) :: xx, a, t
    integer :: i

    if (x < 0.5_dp) then
      ! Reflection formula
      res = log(PI / sin(PI * x)) - f_gamma_ln(1.0_dp - x)
      return
    end if

    xx = x - 1.0_dp
    a = LANCZOS_C(1)
    do i = 2, LANCZOS_N
      a = a + LANCZOS_C(i) / (xx + real(i - 1, dp))
    end do
    t = xx + LANCZOS_G + 0.5_dp
    res = 0.5_dp * log(2.0_dp * PI) + (xx + 0.5_dp) * log(t) - t + log(a)
  end function f_gamma_ln

  ! =========================================================================
  ! Gamma function
  ! =========================================================================
  pure function f_gamma(x) result(res)
    real(dp), intent(in) :: x
    real(dp) :: res
    res = exp(f_gamma_ln(x))
  end function f_gamma

  ! =========================================================================
  ! Log factorial: ln(n!)
  ! =========================================================================
  pure function f_log_factorial(n) result(res)
    integer(c_int), intent(in) :: n
    real(dp) :: res
    res = f_gamma_ln(real(n, dp) + 1.0_dp)
  end function f_log_factorial

  ! =========================================================================
  ! Factorial: n!
  ! =========================================================================
  pure function f_factorial(n) result(res)
    integer(c_int), intent(in) :: n
    real(dp) :: res
    real(dp), parameter :: inf = huge(1.0_dp)
    if (n > 170) then
      res = inf
    else
      res = exp(f_log_factorial(n))
    end if
  end function f_factorial

  ! =========================================================================
  ! Binomial coefficient C(n, k)
  ! =========================================================================
  pure function f_binomial_coeff(n, k) result(res)
    integer(c_int), intent(in) :: n, k
    real(dp) :: res
    if (k < 0 .or. k > n) then
      res = 0.0_dp
    else
      res = exp(f_log_factorial(n) - f_log_factorial(k) - f_log_factorial(n - k))
    end if
  end function f_binomial_coeff

  ! =========================================================================
  ! Beta function B(a, b)
  ! =========================================================================
  pure function f_beta_fn(a, b) result(res)
    real(dp), intent(in) :: a, b
    real(dp) :: res
    res = exp(f_gamma_ln(a) + f_gamma_ln(b) - f_gamma_ln(a + b))
  end function f_beta_fn

  ! =========================================================================
  ! Error function erf(x) - Horner approximation (Abramowitz & Stegun)
  ! =========================================================================
  pure function f_erf(x) result(res)
    real(dp), intent(in) :: x
    real(dp) :: res
    real(dp) :: ax, t, poly, sgn

    if (x >= 0.0_dp) then
      sgn = 1.0_dp
    else
      sgn = -1.0_dp
    end if
    ax = abs(x)
    t = 1.0_dp / (1.0_dp + 0.3275911_dp * ax)
    poly = t * (0.254829592_dp + t * (-0.284496736_dp + &
           t * (1.421413741_dp + t * (-1.453152027_dp + t * 1.061405429_dp))))
    res = sgn * (1.0_dp - poly * exp(-ax * ax))
  end function f_erf

  ! =========================================================================
  ! Complementary error function erfc(x) = 1 - erf(x)
  ! =========================================================================
  pure function f_erfc(x) result(res)
    real(dp), intent(in) :: x
    real(dp) :: res
    res = 1.0_dp - f_erf(x)
  end function f_erfc

  ! =========================================================================
  ! Lower regularized incomplete gamma P(s, x) = gamma(s,x) / Gamma(s)
  ! =========================================================================
  pure function f_regularized_gamma_p(s, x) result(res)
    real(dp), intent(in) :: s, x
    real(dp) :: res

    if (x <= 0.0_dp) then
      res = 0.0_dp
      return
    end if

    if (x < s + 1.0_dp) then
      res = gamma_p_series(s, x)
    else
      res = 1.0_dp - gamma_p_cf(s, x)
    end if
  end function f_regularized_gamma_p

  ! Series expansion for P(s, x)
  pure function gamma_p_series(s, x) result(res)
    real(dp), intent(in) :: s, x
    real(dp) :: res
    real(dp) :: term, total
    integer :: n

    term = 1.0_dp / s
    total = term
    do n = 1, MAX_ITER
      term = term * x / (s + real(n, dp))
      total = total + term
      if (abs(term) < abs(total) * EPS) exit
    end do
    res = total * exp(-x + s * log(x) - f_gamma_ln(s))
  end function gamma_p_series

  ! Continued fraction for Q(s, x) = 1 - P(s, x)
  pure function gamma_p_cf(s, x) result(res)
    real(dp), intent(in) :: s, x
    real(dp) :: res
    real(dp) :: f, c, d, an, bn, delta
    integer :: n

    f = x + 1.0_dp - s
    if (abs(f) < EPS) f = EPS
    c = f
    d = 0.0_dp

    do n = 1, MAX_ITER
      an = real(n, dp) * (s - real(n, dp))
      bn = x + 2.0_dp * real(n, dp) + 1.0_dp - s
      d = bn + an * d
      if (abs(d) < EPS) d = EPS
      c = bn + an / c
      if (abs(c) < EPS) c = EPS
      d = 1.0_dp / d
      delta = c * d
      f = f * delta
      if (abs(delta - 1.0_dp) < EPS) exit
    end do
    res = exp(-x + s * log(x) - f_gamma_ln(s)) / f
  end function gamma_p_cf

  ! =========================================================================
  ! Regularized incomplete beta I_x(a, b)
  ! =========================================================================
  pure recursive function f_regularized_beta(x, a, b) result(res)
    real(dp), intent(in) :: x, a, b
    real(dp) :: res
    real(dp) :: ln_prefactor, prefactor

    if (x <= 0.0_dp) then
      res = 0.0_dp
      return
    end if
    if (x >= 1.0_dp) then
      res = 1.0_dp
      return
    end if

    ! Use symmetry for better convergence
    if (x > (a + 1.0_dp) / (a + b + 2.0_dp)) then
      res = 1.0_dp - f_regularized_beta(1.0_dp - x, b, a)
      return
    end if

    ln_prefactor = f_gamma_ln(a + b) - f_gamma_ln(a) - f_gamma_ln(b) + &
                   a * log(x) + b * log(1.0_dp - x)
    prefactor = exp(ln_prefactor)
    res = prefactor * beta_cf(x, a, b) / a
  end function f_regularized_beta

  ! Continued fraction for incomplete beta (Lentz's algorithm)
  pure function beta_cf(x, a, b) result(res)
    real(dp), intent(in) :: x, a, b
    real(dp) :: res
    real(dp) :: f, c, d, numerator, delta
    integer :: m
    real(dp) :: rm

    f = 1.0_dp
    c = 1.0_dp
    d = 1.0_dp - (a + b) * x / (a + 1.0_dp)
    if (abs(d) < EPS) d = EPS
    d = 1.0_dp / d
    f = d

    do m = 1, MAX_ITER
      rm = real(m, dp)

      ! Even step
      numerator = rm * (b - rm) * x / ((a + 2.0_dp * rm - 1.0_dp) * (a + 2.0_dp * rm))
      d = 1.0_dp + numerator * d
      if (abs(d) < EPS) d = EPS
      c = 1.0_dp + numerator / c
      if (abs(c) < EPS) c = EPS
      d = 1.0_dp / d
      f = f * c * d

      ! Odd step
      numerator = -(a + rm) * (a + b + rm) * x / ((a + 2.0_dp * rm) * (a + 2.0_dp * rm + 1.0_dp))
      d = 1.0_dp + numerator * d
      if (abs(d) < EPS) d = EPS
      c = 1.0_dp + numerator / c
      if (abs(c) < EPS) c = EPS
      d = 1.0_dp / d
      delta = c * d
      f = f * delta

      if (abs(delta - 1.0_dp) < EPS) exit
    end do

    res = f
  end function beta_cf

end module special_functions


! ===========================================================================
! C-callable wrappers using ISO_C_BINDING
! ===========================================================================
subroutine c_gamma_ln(x, result) bind(C, name="fortran_gamma_ln")
  use iso_c_binding
  use special_functions
  real(c_double), intent(in), value :: x
  real(c_double), intent(out) :: result
  result = f_gamma_ln(x)
end subroutine

subroutine c_gamma(x, result) bind(C, name="fortran_gamma")
  use iso_c_binding
  use special_functions
  real(c_double), intent(in), value :: x
  real(c_double), intent(out) :: result
  result = f_gamma(x)
end subroutine

subroutine c_log_factorial(n, result) bind(C, name="fortran_log_factorial")
  use iso_c_binding
  use special_functions
  integer(c_int), intent(in), value :: n
  real(c_double), intent(out) :: result
  result = f_log_factorial(n)
end subroutine

subroutine c_factorial(n, result) bind(C, name="fortran_factorial")
  use iso_c_binding
  use special_functions
  integer(c_int), intent(in), value :: n
  real(c_double), intent(out) :: result
  result = f_factorial(n)
end subroutine

subroutine c_binomial_coeff(n, k, result) bind(C, name="fortran_binomial_coeff")
  use iso_c_binding
  use special_functions
  integer(c_int), intent(in), value :: n, k
  real(c_double), intent(out) :: result
  result = f_binomial_coeff(n, k)
end subroutine

subroutine c_beta_fn(a, b, result) bind(C, name="fortran_beta_fn")
  use iso_c_binding
  use special_functions
  real(c_double), intent(in), value :: a, b
  real(c_double), intent(out) :: result
  result = f_beta_fn(a, b)
end subroutine

subroutine c_erf(x, result) bind(C, name="fortran_erf")
  use iso_c_binding
  use special_functions
  real(c_double), intent(in), value :: x
  real(c_double), intent(out) :: result
  result = f_erf(x)
end subroutine

subroutine c_erfc(x, result) bind(C, name="fortran_erfc")
  use iso_c_binding
  use special_functions
  real(c_double), intent(in), value :: x
  real(c_double), intent(out) :: result
  result = f_erfc(x)
end subroutine

subroutine c_regularized_gamma_p(s, x, result) bind(C, name="fortran_regularized_gamma_p")
  use iso_c_binding
  use special_functions
  real(c_double), intent(in), value :: s, x
  real(c_double), intent(out) :: result
  result = f_regularized_gamma_p(s, x)
end subroutine

subroutine c_regularized_beta(x, a, b, result) bind(C, name="fortran_regularized_beta")
  use iso_c_binding
  use special_functions
  real(c_double), intent(in), value :: x, a, b
  real(c_double), intent(out) :: result
  result = f_regularized_beta(x, a, b)
end subroutine
