!module for 1D interpolation
module interp_mod
  use omp_lib
  use precision_mod
  implicit none
contains

! note we interpolate the power spectrum in log10-log10 space
function lin_interp(x, y, x0) result(y0)
  implicit none
  ! x and y are the nodes and values at the nodes, x0 is the point at which the interpolating function is to be evaluated
  ! x is assumed to be sorted in ascending order
  real(dp), intent(in) :: x(:), y(:), x0
  real(dp) :: y0
  integer :: il, u, l
  integer :: n 

  n  = size(x)
  l = 1
  u = n

  do while (u > l)
    il = (l+u)/2
    if (x(il) <= x0) then
      l = il + 1
    else
      u = il
    end if
  end do
  il = l

  y0 = y(il-1) + (y(il) - y(il-1)) * (x0 - x(il-1)) / (x(il) - x(il-1))

end function lin_interp

! function cubic_spline_interp(x,y,x0) result(y0)
!   implicit none
!   real(dp), intent(in) :: x(:), y(:), x0
!   real(dp) :: y0
!   integer :: i

!   ! implement cubic spline interpolation
!   y = 0.0_dp

! end function cubic_spline_interp



end module interp_mod