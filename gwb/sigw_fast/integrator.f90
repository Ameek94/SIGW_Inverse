! SIGWfast
! This file includes code derived from [SIGWfast](https://github.com/Lukas-T-W/SIGWfast),
! originally written by [Lukas T. Witkowski].
! MIT License
! Copyright (c) 2022 Lukas T. Witkowski
! Modifications have been made by Ameek Malhotra.

module integrator_mod
  use precision_mod
  use omp_lib
  implicit none
contains

  ! Function for the case with c_s^2 = w
  function sigwint_w(cInt1, cInt2, cd, cs1, cs2, nd, ns1, ns2) result(cI)
    implicit none
    integer, intent(in) :: nd, ns1, ns2
    real(dp), intent(in) :: cInt1(:), cInt2(:), cd(:), cs1(:), cs2(:)
    real(dp) :: cI
    real(dp), allocatable :: cId(:)
    integer :: i, j, start_idx, end_idx

    allocate(cId(nd))
    cI = 0.0d0

    do i = 1, nd
    ! Integration over the first set (cs1)
      start_idx = (i-1)*ns1 + 1
      end_idx   = i*ns1
      cId(i) = SUM((cInt1(start_idx+1:end_idx) + cInt1(start_idx:end_idx-1)) * &
                (cs1(start_idx+1:end_idx) - cs1(start_idx:end_idx-1))) / 2.0d0

      ! Integration over the second set (cs2)
      start_idx = (i-1)*ns2 + 1
      end_idx   = i*ns2
      cId(i) = cId(i) + SUM((cInt2(start_idx+1:end_idx) + cInt2(start_idx:end_idx-1)) * &
                         (cs2(start_idx+1:end_idx) - cs2(start_idx:end_idx-1))) / 2.0d0
    end do

    if (nd > 1) then
      cI = SUM((cId(2:nd) + cId(1:nd-1)) * (cd(2:nd) - cd(1:nd-1))) / 2.0d0
    endif

    ! !$omp parallel do private(j, start_idx, end_idx)
    ! do i = 1, nd
    !    cId(i) = 0.0d0
    !    ! Integration over the first set of intervals (cs1)
    !    start_idx = (i-1)*ns1 + 1
    !    end_idx   = i*ns1
    !    do j = start_idx+1, end_idx
    !       cId(i) = cId(i) + (cInt1(j) + cInt1(j-1))*(cs1(j) - cs1(j-1))/2.0d0
    !    end do
    !    ! Integration over the second set of intervals (cs2)
    !    start_idx = (i-1)*ns2 + 1
    !    end_idx   = i*ns2
    !    do j = start_idx+1, end_idx
    !       cId(i) = cId(i) + (cInt2(j) + cInt2(j-1))*(cs2(j) - cs2(j-1))/2.0d0
    !    end do
    ! !    if (i > 1) then
    ! !       cI = cI + (cId(i) + cId(i-1))*(cd(i) - cd(i-1))/2.0d0
    ! !    end if
    ! end do
    ! !$omp end parallel do

    ! cI = 0.0d0
    ! !$omp parallel do reduction(+:cI)
    ! do i = 2, nd
    !     cI = cI + (cId(i) + cId(i-1)) * (cd(i) - cd(i-1)) / 2.0d0
    ! end do
    ! !$omp end parallel do

    deallocate(cId)
  end function sigwint_w

  ! Function for the case with c_s^2 = 1
  function sigwint_1(cInt, cd, cs, nd, ns) result(cI)
    implicit none
    integer, intent(in) :: nd, ns
    real(dp), intent(in) :: cInt(:), cd(:), cs(:)
    real(dp) :: cI
    real(dp), allocatable :: cId(:)
    integer :: i, j, start_idx, end_idx

    allocate(cId(nd))
    cI = 0.0d0

    do i = 1, nd
       cId(i) = 0.0d0
       start_idx = (i-1)*ns + 1
       end_idx   = i*ns
       do j = start_idx+1, end_idx
          cId(i) = cId(i) + (cInt(j) + cInt(j-1))*(cs(j) - cs(j-1))/2.0d0
       end do
      !  if (i > 1) then
      !     cI = cI + (cId(i) + cId(i-1))*(cd(i) - cd(i-1))/2.0d0
      !  end if
    end do

    cI = 0.0d0
    do i = 2, nd
      cI = cI + (cId(i) + cId(i-1)) * (cd(i) - cd(i-1)) / 2.0d0
    end do

    deallocate(cId)
  end function sigwint_1

end module integrator_mod
