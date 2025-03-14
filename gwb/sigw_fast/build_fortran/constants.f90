module precision_mod
    implicit none
    integer, parameter :: dp = selected_real_kind(p=15,r=307) ! kind(1.d0) !selected_real_kind(15, 307)
end module precision_mod

module constants_mod
    use precision_mod
    implicit none
    real(dp), parameter :: pi = 3.1415926535897932384626433832795_dp
    real(dp), parameter :: c = 2.99792458e8_dp
    real(dp), parameter :: Mpc = 3.085677581e22_dp
    real(dp), parameter :: MPC_in_sec = Mpc/c ! Mpc/c = 1.029271250e14 in SI units
end module constants_mod