! SIGWfast
! This file includes code derived from [SIGWfast](https://github.com/Lukas-T-W/SIGWfast),
! originally written by [Lukas T. Witkowski].
! MIT License
! Copyright (c) 2022 Lukas T. Witkowski
! Modifications have been made by Ameek Malhotra.

module sigwfast_mod
  use interp_mod
  use omp_lib
  use powerspectrum_mod
  use integrator_mod
  use precision_mod
  implicit none
contains

  function power_spectrum_k(nodes, vals, k) result(y)
  ! A function that computes the power spectrum at a given k, nodes, and vals.
  ! Node locations are given as log10_k and vals are log10_Pk.
    implicit none
    real(dp), intent(in) :: k
    real(dp), intent(in) :: nodes(:), vals(:)
    real(dp) :: y, log10k
    integer :: n

    n = size(nodes)
    log10k = log10(k)

    if (log10k < nodes(1) .or. log10k>nodes(n)) then
      y = 0.0_dp
    else
      y = 10**(lin_interp(nodes, vals, log10k))
    end if

  end function power_spectrum_k

  function power_spectrum_k_array(nodes, vals, k) result(y)
  ! A function that computes the power spectrum for an array of k, given nodes and vals. Nodes are log10_k and vals are log10_Pk, k is in physical space.
    implicit none
    real(dp), intent(in) :: k(:)
    real(dp), intent(in) :: nodes(:), vals(:)
    real(dp) :: y(size(k))
    integer :: i, spline_method = 1

    if (spline_method == 1) then !linear interpolation
      do i = 1, size(k)
        y(i) = power_spectrum_k(nodes, vals, k(i))
      end do
    else if (spline_method == 3) then !cubic spline interpolation
      do i = 1, size(k)
        y(i) = power_spectrum_k(nodes, vals, k(i))
      end do
    end if

  end function power_spectrum_k_array

  ! A function that computes:
  !    Psquared(d, s, k) = Pofk( k/2*(s+d) ) * Pofk( k/2*(s-d) )
  function psquared(nodes, vals, d, s, k) result(y)
    implicit none
    real(dp), intent(in) :: nodes(:), vals(:), d(:), s(:), k
    real(dp) :: y(size(d)) !, ps1(size(d)), ps2(size(d))
    integer :: i


    ! call test_pz_array((k/2.0d0)*(s(:)+d(:)), ps1)
    ! call test_pz_array((k/2.0d0)*(s(:)-d(:)), ps2)

    ! y(:) = lin_interp_loglog_array(nodes, vals, (k/2.0d0)*(s(:)+d(:)) ) * lin_interp_loglog_array(nodes, vals, (k/2.0d0)*(s(i)-d(i)) )

    do i=1, size(d)
      y(i) = power_spectrum_k(nodes, vals, (k/2.0d0)*(s(i)+d(i)) ) * power_spectrum_k(nodes,vals,(k/2.0d0)*(s(i)-d(i)) )   
      !10**(lin_interp(nodes, vals, log10((k/2.0d0)*(s(i)+d(i)))) + &
                !lin_interp(nodes,vals, log10((k/2.0d0)*(s(i)-d(i)))) )
    end do

    ! do i = 1, size(d)
    !   y(i) = test_pz( (k/2.0d0)*(s(i)+d(i)) ) * test_pz( (k/2.0d0)*(s(i)-d(i)) )
    ! ! y(:) = ps1(:) * ps2(:)
    ! end do

  end function psquared

  function compute_w_k(nodes, vals, k, kernel1, kernel2, d1array, &
                      s1array, d2array, s2array, darray, nd, ns1, ns2) result(y_k)
  ! computes Omega_gw for a single k
    implicit none
    ! Input arguments:
    integer, intent(in) :: nd, ns1, ns2
    real(dp), intent(in) :: k
    real(dp), intent(in) :: nodes(:), vals(:)
    real(dp), intent(in) :: kernel1(nd*ns1), kernel2(nd*ns2)
    real(dp), intent(in) :: d1array(nd*ns1), s1array(nd*ns1)
    real(dp), intent(in) :: d2array(nd*ns2), s2array(nd*ns2)
    real(dp), intent(in) :: darray(nd)
    real(dp) :: y_k
    integer :: i
    
    ! Local variables:
    real(dp), allocatable :: Int_ds1(:), Int_ds2(:), ps1(:), ps2(:)

    allocate(Int_ds1(ns1*nd), Int_ds2(ns2*nd))
    allocate(ps1(ns1*nd), ps2(ns1*nd))

    ps1 = psquared(nodes, vals, d1array, s1array, k)
    ps2 = psquared(nodes, vals, d2array, s2array, k)

    ! if (ns1 > ns2) then
    !   do i=1, ns2*nd
    !     Int_ds1(i) = kernel1(i) * ps1(i)
    !     Int_ds2(i) = kernel2(i) * ps2(i)
    !   end do

    !   do i=ns2*nd+1,ns1*nd
    !     Int_ds1(i) = kernel1(i) * ps1(i)
    !   end do
    ! else
    !   do i=1, ns1*nd
    !     Int_ds1(i) = kernel1(i) * ps1(i)
    !     Int_ds2(i) = kernel2(i) * ps2(i)
    !   end do

    !   do i=ns1*nd+1,ns2*nd
    !     Int_ds2(i) = kernel2(i) * ps2(i)
    !   end do
    ! end if

    ! do i=1, ns1*nd
    !   Int_ds1(i) = kernel1(i) * ps1(i)
    ! end do

    ! do i=1,ns2*nd      
    !   Int_ds2(i) = kernel2(i) * ps2(i)
    ! end do

    Int_ds1(1:ns1*nd) = kernel1(1:ns1*nd) * ps1(1:ns1*nd)
    Int_ds2(1:ns2*nd) = kernel2(1:ns2*nd) * ps2(1:ns2*nd)

    y_k = sigwint_w(Int_ds1, Int_ds2, darray, s1array, s2array, nd, ns1, ns2)

    deallocate(Int_ds1, Int_ds2, ps1, ps2)

  end function compute_w_k

  subroutine compute_w_k_array(nodes, vals, nk, komega, kernel1, kernel2, d1array,&
                       s1array, d2array, s2array, darray, nd, ns1, ns2, Integral)
    implicit none
    ! Input arguments:
    integer, intent(in) :: nk, nd, ns1, ns2
    real(dp), intent(in) :: komega(nk)
    real(dp), intent(in) :: nodes(:), vals(:)
    real(dp), intent(in) :: kernel1(nd*ns1), kernel2(nd*ns2)
    real(dp), intent(in) :: d1array(nd*ns1), s1array(nd*ns1)
    real(dp), intent(in) :: d2array(nd*ns2), s2array(nd*ns2)
    real(dp), intent(in) :: darray(nd)
    ! Output argument:
    real(dp), intent(out) :: Integral(nk)
    
    ! Local variables:
    integer ::  i

    !$omp parallel do default(shared) private(i)       
    do i = 1, nk
       Integral(i) = compute_w_k(nodes, vals, komega(i), kernel1, kernel2, d1array,&
                                 s1array, d2array, s2array, darray, nd, ns1, ns2)
    end do
    !$omp end parallel do

  end subroutine compute_w_k_array

  function compute_1_k(nodes, vals, k, kernel, darray, ddarray, ssarray, nd, ns) result(y_k)
  ! computes Omega_gw for a single k
    implicit none
    ! Input arguments:
    integer, intent(in) :: nd, ns
    real(dp), intent(in) :: k
    real(dp), intent(in) :: nodes(:), vals(:)
    real(dp), intent(in) :: kernel(nd*ns)
    real(dp), intent(in) :: ddarray(nd*ns), ssarray(nd*ns)
    real(dp), intent(in) :: darray(nd)
    real(dp) :: y_k
    
    ! Local variables:
    real(dp), allocatable :: Int_ds(:)

    allocate(Int_ds(ns*nd))

    Int_ds(1:ns*nd) = kernel(1:ns*nd) * psquared(nodes, vals, ddarray, ssarray, k)
    y_k = sigwint_1(Int_ds, darray, ssarray, nd, ns)

  end function compute_1_k

  subroutine compute_1_k_array(nodes, vals, nk, komega, kernel, darray, &
                              ddarray, ssarray, nd, ns, Integral)
    implicit none
    ! Input arguments:
    integer, intent(in) :: nk, nd, ns
    real(dp), intent(in) :: nodes(:), vals(:)
    real(dp), intent(in) :: komega(nk)
    real(dp), intent(in) :: kernel(nd*ns)
    real(dp), intent(in) :: ddarray(nd*ns), ssarray(nd*ns)
    real(dp), intent(in) :: darray(nd)
    ! Output argument:
    real(dp), intent(out) :: Integral(nk)
    
    ! Local variables:
    integer ::  i
    


    ! $omp parallel do default(shared) private(i) 
    do i = 1, nk
       Integral(i) = compute_1_k(nodes, vals, komega(i), kernel, darray, ddarray, ssarray, nd, ns)
    end do
    ! $omp end parallel do


  end subroutine compute_1_k_array

end module sigwfast_mod