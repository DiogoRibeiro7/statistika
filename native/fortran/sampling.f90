module sampling_mod
  use iso_c_binding, only: c_int, c_double
  implicit none
  private
  public :: fortran_uniform_sample_batch
  public :: fortran_normal_sample_batch
  public :: fortran_gamma_sample_batch
  public :: fortran_beta_sample_batch

contains

  subroutine init_state(seed, state)
    integer(c_int), intent(in), value :: seed
    integer(c_int), intent(out) :: state(4)
    integer(c_int) :: value
    integer :: i

    value = seed
    if (value == 0) value = 1
    do i = 1, 4
      value = splitmix32(value)
      state(i) = value
    end do
  end subroutine init_state

  pure function splitmix32(x) result(y)
    integer(c_int), intent(in) :: x
    integer(c_int) :: y
    integer(c_int) :: z

    z = x + int(z'9E3779B9', c_int)
    z = ieor(z, ishft(z, -16))
    z = z * int(z'85EBCA6B', c_int)
    z = ieor(z, ishft(z, -13))
    z = z * int(z'C2B2AE35', c_int)
    z = ieor(z, ishft(z, -16))
    y = z
  end function splitmix32

  function next_uint32(state) result(res)
    integer(c_int), intent(inout) :: state(4)
    integer(c_int) :: res
    integer(c_int) :: x, y, t

    x = state(1)
    y = state(4)
    t = ieor(x, ishft(x, 11))
    state(1) = state(2)
    state(2) = state(3)
    state(3) = state(4)
    state(4) = ieor(ieor(ieor(y, ishft(y, -19)), t), ishft(t, -8))
    res = iand(state(4) + y, int(z'FFFFFFFF', c_int))
  end function next_uint32

  function rand01(state) result(u)
    integer(c_int), intent(inout) :: state(4)
    real(c_double) :: u
    integer(c_int) :: r

    r = next_uint32(state)
    u = real(iand(r, int(z'FFFFFFFF', c_int)), kind=c_double) / 4294967296.0d0
    if (u >= 1.0d0) u = 1.0d0 - 1.0d-16
  end function rand01

  function standard_normal(state) result(z)
    integer(c_int), intent(inout) :: state(4)
    real(c_double) :: z
    real(c_double) :: u1, u2

    u1 = rand01(state)
    if (u1 < 1.0d-300) u1 = 1.0d-300
    u2 = rand01(state)
    z = sqrt(-2.0d0 * log(u1)) * cos(2.0d0 * 3.141592653589793d0 * u2)
  end function standard_normal

  recursive function sample_gamma(state, shape, rate) result(x)
    integer(c_int), intent(inout) :: state(4)
    real(c_double), intent(in), value :: shape, rate
    real(c_double) :: x
    real(c_double) :: d, c, v, u, z

    if (shape < 1.0d0) then
      x = sample_gamma(state, shape + 1.0d0, 1.0d0)
      u = rand01(state)
      x = x * u ** (1.0d0 / shape)
      x = x / rate
      return
    end if

    d = shape - 1.0d0 / 3.0d0
    c = 1.0d0 / sqrt(9.0d0 * d)
    do
      do
        z = standard_normal(state)
        v = 1.0d0 + c * z
        if (v > 0.0d0) exit
      end do
      v = v * v * v
      u = rand01(state)
      if (u < 1.0d0 - 0.0331d0 * (z * z) * (z * z)) exit
      if (log(u) < 0.5d0 * z * z + d * (1.0d0 - v + log(v))) exit
    end do
    x = (d * v) / rate
  end function sample_gamma

  subroutine fortran_uniform_sample_batch(seed, a, b, result, n) bind(C, name="fortran_uniform_sample_batch")
    integer(c_int), intent(in), value :: seed
    real(c_double), intent(in), value :: a, b
    real(c_double), intent(out) :: result(*)
    integer(c_int), intent(in), value :: n
    integer(c_int) :: state(4)
    integer(c_int) :: i

    call init_state(seed, state)
    do i = 1, n
      result(i) = a + (b - a) * rand01(state)
    end do
  end subroutine fortran_uniform_sample_batch

  subroutine fortran_normal_sample_batch(seed, mu, sigma, result, n) bind(C, name="fortran_normal_sample_batch")
    integer(c_int), intent(in), value :: seed
    real(c_double), intent(in), value :: mu, sigma
    real(c_double), intent(out) :: result(*)
    integer(c_int), intent(in), value :: n
    integer(c_int) :: state(4)
    integer(c_int) :: i
    real(c_double) :: z0, z1, u1, u2

    call init_state(seed, state)
    i = 1
    do while (i <= n)
      u1 = rand01(state)
      if (u1 < 1.0d-300) u1 = 1.0d-300
      u2 = rand01(state)
      z0 = sqrt(-2.0d0 * log(u1)) * cos(2.0d0 * 3.141592653589793d0 * u2)
      z1 = sqrt(-2.0d0 * log(u1)) * sin(2.0d0 * 3.141592653589793d0 * u2)
      result(i) = mu + sigma * z0
      if (i + 1 <= n) result(i + 1) = mu + sigma * z1
      i = i + 2
    end do
  end subroutine fortran_normal_sample_batch

  subroutine fortran_gamma_sample_batch(seed, shape, rate, result, n) bind(C, name="fortran_gamma_sample_batch")
    integer(c_int), intent(in), value :: seed
    real(c_double), intent(in), value :: shape, rate
    real(c_double), intent(out) :: result(*)
    integer(c_int), intent(in), value :: n
    integer(c_int) :: state(4)
    integer(c_int) :: i

    call init_state(seed, state)
    do i = 1, n
      result(i) = sample_gamma(state, shape, rate)
    end do
  end subroutine fortran_gamma_sample_batch

  subroutine fortran_beta_sample_batch(seed, alpha, beta, result, n) bind(C, name="fortran_beta_sample_batch")
    integer(c_int), intent(in), value :: seed
    real(c_double), intent(in), value :: alpha, beta
    real(c_double), intent(out) :: result(*)
    integer(c_int), intent(in), value :: n
    integer(c_int) :: state(4)
    integer(c_int) :: i
    real(c_double) :: x, y

    call init_state(seed, state)
    do i = 1, n
      x = sample_gamma(state, alpha, 1.0d0)
      y = sample_gamma(state, beta, 1.0d0)
      result(i) = x / (x + y)
    end do
  end subroutine fortran_beta_sample_batch

end module sampling_mod
