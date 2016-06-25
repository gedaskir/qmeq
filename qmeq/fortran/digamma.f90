subroutine digamma(x, y, psr, psi) bind(c)
    use iso_c_binding, only: c_double
    implicit none
    real(c_double), intent(in) :: x
    real(c_double), intent(in) :: y
    real(c_double), intent(out) :: psr
    real(c_double), intent(out) :: psi
    call CPSI(x, y, psr, psi)
end subroutine
