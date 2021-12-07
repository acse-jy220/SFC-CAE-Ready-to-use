

       program main
       implicit none
       integer :: nonods_l, ndim, nscalar
       integer :: nonods
       parameter(ndim=1,nscalar=1)
       parameter(nonods_l=19,nonods=4)
       real :: x_l(ndim,nonods_l), conc_l(nscalar,nonods_l)
       real :: x(ndim,nonods), conc(nscalar,nonods)
       integer :: i
       
       print *,'me'
       do i=1,nonods
          x(1,i) = real(i)
       end do
       conc=1.0
       conc(1,:)=x(1,:)
       call x_conv_fixed_length(x_l,conc_l, x,conc, nonods_l,nonods,ndim,nscalar)
       print *,'nonods_l,nonods:',nonods_l,nonods
       print *,'x:',x
       print *,'x_l:',x_l
       print *,'conc:'
       print *,'conc:',conc
       print *,'conc_l:',conc_l
! interpolate back
       print *,'interpolate back'
       call x_conv_fixed_length(x,conc, x_l,conc_l, nonods,nonods_l,ndim,nscalar)
       print *,'nonods_l,nonods:',nonods_l,nonods
       print *,'x:',x
       print *,'x_l:',x_l
       print *,'conc:'
       print *,'conc:',conc
       print *,'conc_l:',conc_l
       return
       end program main
! 
! 
! 
! 
! from python call...
! x_l,conc_l = x_conv_fixed_length(x,conc,nonods,nonods_l,ndim,nscalar)
       subroutine x_conv_fixed_length(x_l,conc_l, x,conc, nonods_l,nonods,ndim,nscalar)
! ************************************************************************************
! This subroutine calculates x_l from x & conc_l from conc by linearly interpolating 
! from a regular grid
! ************************************************************************************
! nonods = no of nodes in mesh to be interpolated from. 
! nonods_l = no of nodes in mesh to be interpolated too. 
! ndim= no of dimensions e.g. for 3D problems =3. 
! nscalar=no of concentration fields to to be interpolated. 
! conc_l= the concentration field to be interpolated too.
! conc= the concentration field to be interpolated from.
! x_l= the spatial coordinates to be interpolated too.
! x= the spatial coordinates to be interpolated from.
! 
! coordinates = spatial coordinates
       implicit none
       integer, intent(in) :: nonods_l, ndim, nscalar
       integer, intent(in) :: nonods
       real, intent(out) :: x_l(ndim,nonods_l), conc_l(nscalar,nonods_l)
       real, intent(in) :: x(ndim,nonods), conc(nscalar,nonods)
! 
! local variables...
! optimal_back_inter=0 (original simple interpolation); optimal_back_inter=1(optimal back interpolaiton)
       integer, parameter :: optimal_back_inter=1
       real, parameter :: toler=1.0e-6
       integer :: nod, nod_l, nod_l_last, nod_prev, nod_next, nod_prev_keep
       real :: weight_interp
       real :: w1, w2, rsum
       real, allocatable :: x_regular(:), x_l_regular(:), conc_l_1(:), conc_l_2(:)
       integer, allocatable :: nod_prev_list(:)
! 
       allocate(x_regular(nonods), x_l_regular(nonods_l), conc_l_1(nscalar), conc_l_2(nscalar))
       allocate(nod_prev_list(nonods_l))

       do nod = 1,nonods
          x_regular(nod)=real(nod-1)/real(nonods-1)
       end do
       do nod_l = 1,nonods_l
          x_l_regular(nod_l)=real(nod_l-1)/real(nonods_l-1)
       end do

!       print *,'here1'
! for nod_prev...
       nod_prev_keep=1
       nod_prev_list(1)=1
       do nod_l = 2,nonods_l-1
          nod_prev=nod_prev_keep
          do nod = nod_prev,nonods-1
             if(  (x_l_regular(nod_l)>=x_regular(nod)   ) &
             .and.(x_l_regular(nod_l)<=x_regular(nod+1) )   ) then
                nod_prev_list(nod_l)=nod
                nod_prev_keep=nod
                exit
             endif 
          end do
       end do
       nod_prev_list(nonods_l)=nonods-1
! 
!       print *,'here2'
! Form min value
! 
! Calculate x_l from x conc_l from conc by linearly interpolating from regular grid...
       do nod_l = 1,nonods_l
          nod_prev=nod_prev_list(nod_l) 
          nod_next=nod_prev+1
          weight_interp = (x_l_regular(nod_l)  - x_regular(nod_prev)) &
                        / (x_regular(nod_next) - x_regular(nod_prev))
          weight_interp = max( min(weight_interp,1.0), 0.0)
          x_l(:,nod_l)= &
          (1.0-weight_interp) * x(:,nod_prev)    + weight_interp * x(:,nod_next)
          conc_l(:,nod_l)= &
          (1.0-weight_interp) * conc(:,nod_prev) + weight_interp * conc(:,nod_next)
       end do
       x_l(:,nonods_l)=x(:,nonods)
       conc_l(:,nonods_l)=conc(:,nonods)
! 
!       do nod_l = 1,nonods_l
!          print *,'nod_l,nod_prev_list(nod_l),x_l(:,nod_l):', &
!                   nod_l,nod_prev_list(nod_l),x_l(:,nod_l)
!       end do

! optimal SFC interpolation back from interpolated values to original mesh nodes...
       if(optimal_back_inter==1) then
       if(nonods.ge.2*nonods_l) then
!          print *,'here **** nonods,nonods_l:',nonods,nonods_l
          do nod_l = 2,nonods_l-1
             nod_prev=nod_prev_list(nod_l) 
             nod_next=nod_prev+1
! use inverse distance weighting between extrapolating from the left and the right of the SFC...
             w1 = 1.0/max(toler, abs(x_l_regular(nod_l) - x_regular(nod_next)) )
             w2 = 1.0/max(toler, abs(x_l_regular(nod_l) - x_regular(nod_prev)) )
             rsum = w1+w2
             w1=w1/rsum
             w2=w2/rsum
             call opt_interp_func( conc_l_1(:), x_l_regular(nod_l), &
                                   x_regular(nod_next), x_regular(nod_next+1), &
                                   conc(:,nod_next),    conc(:,nod_next+1), nscalar )
             call opt_interp_func( conc_l_2(:), x_l_regular(nod_l), &
                                   x_regular(nod_prev), x_regular(nod_prev-1), &
                                   conc(:,nod_prev),    conc(:,nod_prev-1), nscalar )
             conc_l(:,nod_l) = w1*conc_l_1(:) + w2*conc_l_2(:)
          end do
       endif
       endif

       return
       end subroutine x_conv_fixed_length
! 
! 
! 
! 
       subroutine opt_interp_func( conc_l_1, x_l_regular_nod_l, &
                                   x_regular_nod_next, x_regular_nod_next_p1, &
                                   conc_nod_next,      conc_nod_next_p1, nscalar )
! extrapolate linearly from x_regular_nod_next,x_regular_nod_next_p1 to the point x_l_regular_nod_l
! to obtain conc_l_1 from conc_nod_next, conc_nod_next_p1 respectively. 
       integer, intent(in) :: nscalar
       real, intent(in) :: x_l_regular_nod_l, x_regular_nod_next,x_regular_nod_next_p1
       real, intent(in) :: conc_nod_next(nscalar), conc_nod_next_p1(nscalar)
       real, intent(out) :: conc_l_1(nscalar)
! local variables...
       real :: w1, w2
       
       w1 = 1.0 - (x_regular_nod_next - x_l_regular_nod_l ) &
                 /(x_regular_nod_next - x_regular_nod_next_p1 )
       w2 =  1.0 - w1 
       conc_l_1(:) = w1 * conc_nod_next(:) + w2 * conc_nod_next_p1(:)
       return 
       end subroutine opt_interp_func
       
! 
! 
! 
! 

      
