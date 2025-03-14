!module for 1D interpolation
module interp_mod
  use omp_lib
  use precision_mod
  implicit none
contains

function locate(x, x0, n) result(i)
  implicit none
  real(dp), intent(in) :: x(:), x0
  integer, intent(in) :: n
  integer :: i

  i = 1
  ! implement better locate algorithm
  do while (x(i) < x0)
    i = i + 1
  end do

  ! binary search locate
  
end function locate

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

  ! ! ! if out of bounds return value at the closest end
  ! ! if (x0<x(1) .or. x0>x(n)) then
  ! !   if (x0<x(1)) then
  ! !     y0 = y(1)
  ! !   else
  ! !     y0 = y(n)
  ! !   end if  
  ! ! if within bounds first locate x0
  ! else
  !   i = 1
  !   ! implement better locate algorithm
  !   do while (x(i) < x0)
  !     i = i + 1
  !   end do
    ! Linear interpolation
    ! y0 = y(i-1) + (y(i) - y(i-1)) * (x0 - x(i-1)) / (x(i) - x(i-1))
  ! end if
end function lin_interp

! function cubic_spline_interp(x,y,x0) result(y0)
!   implicit none
!   real(dp), intent(in) :: x(:), y(:), x0
!   real(dp) :: y0
!   integer :: i

!   ! implement cubic spline interpolation
!   y = 0.0_dp

! end function cubic_spline_interp



! function lin_interp_array(x,y,x0) result(y0)
!   implicit none
!   real(dp), intent(in) :: x(:), y(:), x0(:)
!   real(dp) :: y0(size(x0))
!   integer :: i

!   do i = 1, size(x0)
!     y0(i) = lin_interp(x,y,x0(i))
!   end do

! end function lin_interp_array

! function lin_interp_loglog_array(x,y,x0) result(y0)
!   implicit none
!   real(dp), intent(in) :: x(:), y(:), x0(:)
!   real(dp) :: y0(size(x0))
!   integer :: i

!   do i = 1, size(x0)
!     y0(i) = 10.0_dp ** lin_interp(log10(x), log10(y), log10(x0(i)))
!   end do

! end function lin_interp_loglog_array



end module interp_mod
