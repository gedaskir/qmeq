real(kind(0.0d0)) function fermi(x)
    implicit none
    real(kind(0.0d0)), intent(in) :: x
    fermi = 1./(exp(x)+1.)
    return        
end function fermi

subroutine pvalint(a, b, c, epsabs, epsrel, result, ier, limit) bind(c)
    use iso_c_binding, only: c_double, c_int
    implicit none
    real(c_double), intent(in) :: a
    real(c_double), intent(in) :: b
    real(c_double), intent(in) :: c
    
    real(c_double), intent(in) :: epsabs
    real(c_double), intent(in) :: epsrel

    real(c_double), intent(out) :: result
    real(kind(0.0d0)) :: abserr
    
    integer :: neval
    integer(c_int), intent(out) :: ier
    integer(c_int), intent(in) :: limit
    integer :: lenw
    integer :: last
    
    integer, dimension(limit) :: iwork
    real(kind(0.0d0)), dimension(4*limit) :: work

    real(kind(0.0d0)), external :: fermi

    lenw = 4*limit    

    call DQAWC(fermi, a, b, c, epsabs, epsrel, result, abserr, neval, ier, limit, lenw, last, iwork, work)    

end subroutine
