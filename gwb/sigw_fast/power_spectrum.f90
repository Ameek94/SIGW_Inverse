module powerspectrum_mod
  use precision_mod
  use omp_lib
  implicit none
contains
  elemental function test_pz(p) result(res)
    implicit none
    ! Declare inputs
    real(dp), intent(in) :: p
    real(dp) :: pstar, n1, n2, sigma, nir, nuv, pl1, pl2, res

    n1 = 2.0d0
    n2 = -1.0d0
    pstar = 5.0d-4
    sigma = 2.0d0

    nir = n1
    pl1 = (p / pstar)**nir
    nuv = (n2 - n1) / sigma
    pl2 = (1.0d0 + (p / pstar)**sigma)**nuv
    res = 1.0d-2 * pl1 * pl2

  end function test_pz

  subroutine test_pz_array(p,res)
    implicit none
    real(dp), intent(in) :: p(:)
    real(dp), intent(out) :: res(size(p))
    integer :: i !, num_threads

    !$omp parallel do default(shared) private(i)
    do i = 1, size(p)
       res(i) = test_pz(p(i))
        ! num_threads = OMP_get_num_threads()
        ! print *, 'num_threads running:', num_threads
    end do
    !$omp end parallel do
  end subroutine test_pz_array
  
end module powerspectrum_mod
