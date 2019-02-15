"""
Mixed lubrication solver by Abdullah
translation by mike...
"""
import numpy as np

class UnifiedMixedSolver():
    surface1=Surface()
    surface2=Surface()
    normal_load
    hardness
    fluid# some sort of fluid object with a reynolds equation method?
    
    
    def __init__(self, round_surface, roughness_surface, normal_load=None):
        if type(roughness_surface) is array:
            roughness_surfac=Surface(roughness_surface)
        self.surface1=surface1
        self.surface2=surface2
        if normal_load: self.normal_load=normal_load
        
    def set_fluid(self, fluid_type, *params):
        pass
        
    def set_materials(self, E1, v1, E2, v2, **kwargs):
        pass
        
    def SUBAK(self,MM):
        #round surface
        S(X,Y)=X+SQRT(X**2+Y**2)
        for I in range(MM+1):
            XP=I+0.5
            XM=I-0.5
            for J in range(I+1):
                YP=J+0.5
                YM=J-0.5
                A1=S(YP,XP)/S(YM,XP)
                A2=S(XM,YM)/S(XP,YM)
                A3=S(YM,XM)/S(YP,XM)
                A4=S(XP,YP)/S(XM,YP)
                AK(I,J)=XP*np.log(A1)+YM*np.log(A2)+XM*np.log(A3)+YP*np.log(A4)
                AK(J,I)=AK(I,J)
    return AK

    def _steady_state(self):
        pass
    
    def __call__(self, grid_points, grid_spacing):
        N1=1
        
        if self.surface1.is_descrete:
            pass
        AK=self.SUBAK(grid_points-1)
        self.EHL(grid_points,N1)
    

    
#**********************************************************************

def EHL(N,N1):
#****************** Main def that integrates EHL equations to get solution: START ***************
    PAI=3.14159265
    Z=0.68
    P0=1.96E8
    X,Y = np.zeros((N,1))
    H,RO,EPS,EDA,P,POLD,V = np.zeros((N,N))
    ROU,ROU2,RO0,HO0,GOU,TAUL,TAUC,fcoeff,E1_MAT = np.zeros((N,N))
    TFILMBALL,TFILMDISC,WEAR,WEARBALL,WEARDISC,VPLASTIC,G00temp = np.zeros((N,N))
    TLOCALMB,TLOCALMD,WEARBALLLOCAL,WEARDISCLOCAL,UPDISC,UPBALL = np.zeros((N,N))
    PLCOUNT = np.zeros((N,N), dtype=np.int16)
#    REAL (kind=8) :: DH,H00,ERH
    U1temp=np.zeros((15,1))
    
#    COMMON /COM1/ENDA,A1,A2,A3,Z,HM0,HM0AVG,DH,DW,G00
#    COMMON /COM2/EDA0
#    common /COMER/E1,RX,B,PH
#    COMMON /COM3/U1,U2,UR,US,W
#    COMMON /COMCONT/ICM,ICB,ICP
#    COMMON /COMTEMP/TB(1:65,1:65) #change this
#    COMMON /HARD/Hardness(1:65,1:65),UPLASTIC(1:65,1:65) # change this
    
    # reading INPUT parameters
    
#    OPEN(18,FILE='INPUTS.DAT',STATUS='UNKNOWN')
#    READ(18,*) PH,E1,EDA0,RX,US,X0,XE,CU
#    CLOSE(18)
    
    # Initialization

    RO0 = np.zeros((N,N))
    HO0 = np.zeros((N,N))
    TFILMBALL = np.zeros((N,N))
    TFILMDISC = np.zeros((N,N))
    UPLASTIC = np.zeros((N,N))
    UPDISC = np.zeros((N,N))
    UPBALL = np.zeros((N,N))
    TAUC = np.zeros((N,N))
    TAUL = np.zeros((N,N))
    fcoeff = np.zeros((N,N))
    Hsub =1.150E9/PH
    Hardness=Hsub
   
    MM=N-1
    A1=ALOG(EDA0)+9.67
    A2=5.1E-9*PH
    A3=0.59/(PH*1.E-9)
    
    # speed setting
    
    U_ball= 0.2530        # U1 is speed of ball
    U_disk= 0.2480        # U2 is speed of disc/ring
    UR= US/2.           #(U1+U2)/2.
    CU=2.*(U1-U2)/(U1+U2)
    U=EDA0*UR/(E1*RX)
    
    B=PAI*PH*RX/E1
    W0=2.*PAI*PH/(3.*E1)*(B/RX)**2
    W=W0*E1*RX**2
    
    ALFA = 14.94E-9
    G=ALFA*E1
    HM0=3.63*(RX/B)**2*G**0.49*U**0.68*W0**(-0.073)*(1.-exp(-0.68))
    ENDA=12.*UR*EDA0*RX**2/(B**3*PH)    #12.*U*(E1/PH)*(RX/B)**3
    
#    write(4,*) "W,B,U,G,W0 ", W,B,U,G,W0
#    write(*,*) "W,B,U,G,W0 ", W,B,U,G,W0
#    write(4,*) U1, U2, UR, ABS(U1-U2), CU
#    WRITE(4,*) N,X0,XE,W,PH,E1,EDA0,RX,US
    
    MK=1
    G0=2.0943951

    CALL INITI(N,DX,X0,XE,X,Y,P,POLD)
    
    # ******** Steady state calculation without roughness: START ********
    
    DT=0.0
    DAD=0.
    DAB=0.
    DF=0.0
    CALL ROUGHNESS(N,DAD,ROU,RMSR)
    CALL GEOM(N,DAB,X,Y,GOU,ROU2,RMSRB)
    
    Wrel=0.2  # This is the relaxation parameter
    KK=0
    CALL HREEI(N,DX,DF,DT,ERH,KK,H00,G0,X,Y,H,RO,EPS,EDA,P,V,GOU,ROU)
14  KK=20
    CALL ITERSEMISEPI(N,KK,DX,DT,ERH,H00,G0,X,Y,H,HO0,RO,RO0,EPS,EDA,P,V,GOU,ROU)
    MK=MK+1
    CALL ERP(N,Wrel,ER,P,POLD)
    WRITE(*,*)'ER=',ER,'DW',ABS(DW),ERH,H00   
    IF(ER.GT.1.E-6) THEN 
      IF(MK.GE.15)THEN
        MK=1
        DH=0.75*DH
        IF(DW.LT.1.E-4) DH=DH*0.9
      ENDIF
    GOTO 14
    ENDIF

    # Calculating maximum pressure and minimum film thickness
    
    H2=1.E3
    P2=0.0
    DO I=1,N
      DO J=1,N
        IF(H(I,J).LE.1.E-9*RX/B/B) H(I,J)=1.E-9*RX/B/B # For the unified algorithm
        IF(H(I,J).LT.H2)H2=H(I,J)
        IF(P(I,J).GT.P2)P2=P(I,J)
      ENDDO
    ENDDO
    H3=H2*B*B/RX
    P3=P2*PH
    WRITE(*,*)'P2,H2,P3,H3,HM0(DIM)=',P2,H2,P3,H3,HM0*B*B/RX
    
    # calculating the load sharing
    
    LASPT=0.0
    LBOUT=0.0
    DO I=1,N
      DO J=1,N
        IF(H(I,J).LE.1.E-9*RX/B/B) THEN
            LASPS=P(I,J)
            LASPT=LASPT+LASPS
          ENDIF
      ENDDO
    ENDDO
    
    LBOUT=LBOUT*DX*DX/G0
    LASPT=LASPT*DX*DX/G0
#    write(*,*) "load sharing", LBOUT,LASPT,LBOUT+LASPT
    
    COUNT=0
    SUM1=0.0
    DO I=1,N
      DO J=1,N
        IF(SQRT(X(I)*X(I)+Y(J)*Y(J)).LE.(2./3.)) THEN
          COUNT=COUNT+1
          SUM1=SUM1+H(I,J)
        ENDIF
      ENDDO
    ENDDO
    HM0AVG=SUM1/REAL(COUNT)

    # ******** Steady state calculation without roughness: END ********
    
    CALL OUTPUT(N,DX,X,Y,H,P)  
  
# ******** Rough surface calculation: START ********
    
    # ******** To move roughness use DT > 0     ********
    # ******** DAD = disc roughness amplitude   ********
    # ******** DAB = ball roughness amplitude   ********
    # ******** DF  = single asperity amplitude  ********
    # ******** single asperity calculations not ********
    # ******** active in this simulation. To    ********
    # ******** activate, add asperity equation  ********
    # ******** in the def HREE           ********
    
    DT=0.
    DAD = 0.
    DAB = 0.
    DF  = 0.088
    G00 = 0.
    G001= 0.
    
    CALL ROUGHNESS(N,DAD,ROU,RMSR)
    CALL GEOM(N,DAB,X,Y,GOU,ROU2,RMSRB)

    DO K2=1,5
    A2=5.1E-9*PH                       # these values will update if the load is changed through iterations
    A3=0.59/(PH*1.E-9)                 # these values will update if the load is changed through iterations
    
    # speed setting
    
    U1=0.2530                          # U2 is speed of ball
    U2=0.2480                          # U2 is speed of disc/ring
#    UR=(U1+U2)/2.
    CU=2.*(U1-U2)/(U1+U2)
    U=EDA0*UR/(E1*RX)
    
    W0=2.*PAI*PH/(3.*E1)*(B/RX)**2
    W=W0*E1*RX**2
    
    HM0=3.63*(RX/B)**2*G**0.49*U**0.68*W0**(-0.073)*(1.-exp(-0.68))
    HMC=2.69*(RX/B)**2*G**0.53*U**0.67*W0**(-0.067)*(1.-0.61*exp(-0.73))
    
    ENDA=12.*UR*EDA0*RX**2/(B**3*PH)

    WRITE(4,*)N,X0,XE,W0,PH,E1,EDA0,RX,US
    
    MK=1
    G0=2.0943951
    CALL FZ(N,H,HO0)
    CALL FZ(N,RO,RO0)
    KK=0
    CALL HREEI(N,DX,DF,DT,ERH,KK,H00,G0,X,Y,H,RO,EPS,EDA,P,V,GOU,ROU)
    
    Wrel=0.2 # the relaxation parameter. Wrel means Wrel *100 = %age of new approximation used

38  H00initial=H00
    COUNT=0
18  KK=20
    COUNT=COUNT+1
    H00initial=H00
    CALL ITERSEMISEPI(N,KK,DX,DT,ERH,H00,G0,X,Y,H,HO0,RO,RO0,EPS,EDA,P,V,GOU,ROU)
    
    MK=MK+1
    CALL ERP(N,Wrel,ER,P,POLD)
    WRITE(*,*)'ER=',ER,'DW',ABS(DW),ERH,H00
    IF(ER.GT.1.E-6) THEN
      IF(MK.GE.15)THEN
        MK=1
        DH=0.9*DH
        IF(DW.LT.1.E-5) DH=DH*0.95
      ENDIF
      IF(COUNT.LT.70) GOTO 18
    ENDIF
    
    # Calculating maximum pressure and minimum film thickness
    
    H2=1.E3
    P2=0.0
    DO I=1,N
      DO J=1,N
        IF(H(I,J).LE.1.E-9*RX/B/B) H(I,J)=1.E-9*RX/B/B
        IF(H(I,J).LT.H2)H2=H(I,J)
        IF(P(I,J).GT.P2)P2=P(I,J)
      ENDDO
    ENDDO
    H3=H2*B*B/RX
    P3=P2*PH
    WRITE(*,*)'P2,H2,P3,H3=',P2,H2,P3,H3,HM0*B*B/RX/1.E-9,HMC*B*B/RX/1.E-9
   
   # calculating the load sharing
    LASPT=0.0
    LBOUT=0.0
    DO I=1,N
      DO J=1,N
        IF(H(I,J).LE.1.E-9*RX/B/B) THEN
            LASPS=P(I,J)
            LASPT=LASPT+LASPS
        ENDIF
      ENDDO
    ENDDO
    LBOUT=LBOUT*DX*DX/G0
    LASPT=LASPT*DX*DX/G0
    write(*,*) "load sharing", LBOUT,LASPT,LBOUT+LASPT
   
    COUNT=0
    SUM1=0.0
    DO I=1,N
      DO J=1,N
        IF(SQRT(X(I)*X(I)+Y(J)*Y(J)).LE.(2./3.)) THEN
          COUNT=COUNT+1
          SUM1=SUM1+H(I,J)
        ENDIF
      ENDDO
    ENDDO
    HM0AVG=SUM1/REAL(COUNT)
    
    write(*,*) "U1, U2, UR, US, SRR ", U1, U2, UR, ABS(U1-U2), CU    
    write(*,*)"RMS roughness (nm):Ball,Disc",RMSRB*B*B/RX/1.E-6,RMSR*B*B/RX/1.E-6
    Write(*,*)"Lambda: (OLD)min, (OLD)cen, NEW,HM0AVG",HM0/SQRT(RMSR**2+RMSRB**2)&
    &,HMC/SQRT(RMSR**2+RMSRB**2),HM0AVG/SQRT(RMSR**2+RMSRB**2),HM0AVG*B*B/RX/1.E-9
    WRITE(*,*)"B, W, U, G",B,W0,U,G
   
    CALL OUTPUT(N,DX,X,Y,H,P)
85  CONTINUE
    ENDDO    
90  FORMAT(65(1p,E16.6))
#****************** Main def that integrates EHL equations to get solution: END ***************    
    RETURN
    END
#**********************************************************************
    def ERP(N,W,ER,P,POLD)
    IMPLICIT NONE

    INTEGER :: N
    REAL :: ER,SUM,W,ptemp
    REAL,DIMENSION(N,N) :: P,POLD
    INTEGER :: I,J
    REAL :: E1,RX,B,PH

    common /COMER/E1,RX,B,PH
    
    ER=0.0
    SUM=0.0

    DO 10 I=1,N
    DO 10 J=1,N
    P(I,J)=POLD(I,J) + W*(P(I,J)-POLD(I,J))
    ptemp=P(I,J)    
    ER=ER+ABS(Ptemp-POLD(I,J))
    POLD(I,J)=Ptemp
    SUM=SUM+POLD(I,J)
10  CONTINUE
    ER=ER/SUM
    RETURN
    END def ERP
#**********************************************************************
    def GEOM(N,DAB,X,Y,GOU,ROU2,RMSRB)
    IMPLICIT NONE
    INTEGER :: N,I,J
    REAL,DIMENSION(N) :: X,Y
    REAL,DIMENSION(N,N) :: GOU,ROU2
    REAL :: RAD,W1,DAB,RMSRB,A
    
    CALL ROUGHNESS(N,DAB,ROU2,RMSRB)
    #ROU2=CSHIFT(ROU2,SHIFT=N/2)
    ROU2=TRANSPOSE(ROU2)
    DO I=1,N
      DO J=1,N
        RAD=X(I)*X(I)+Y(J)*Y(J)
        W1=0.5*RAD
        GOU(I,J)=W1-ROU2(I,J)
      ENDDO
    ENDDO
    
110 FORMAT(130(E12.5,1X))

    RETURN
    END def GEOM
    
#**********************************************************************
    def INITI(N,DX,X0,XE,X,Y,P,POLD)
    IMPLICIT NONE

    INTEGER :: N
    REAL :: DX,Y0,X0,XE,C,D
    REAL,DIMENSION(N) ::  X,Y
    REAL,DIMENSION(N,N) :: P,POLD
    INTEGER :: I,J

    DX=(XE-X0)/(N-1.)
    Y0=-0.5*(XE-X0)
    DO I=1,N
     X(I)=X0+(I-1)*DX
     Y(I)=Y0+(I-1)*DX
    ENDDO
    DO I=1,N
     D=1.-X(I)*X(I)
     DO J=1,N
      C=D-Y(J)*Y(J)
      IF(C.LE.0.0)P(I,J)=0.0
      IF(C.GT.0.0)P(I,J)=SQRT(C)
      POLD(I,J)=P(I,J)
     ENDDO
    ENDDO
    RETURN
    END def INITI
#**********************************************************************
    def tdma(K,a,b,c,d,x)
    implicit none
    
    integer, intent(in) :: K
    real, intent(in) :: a(K), c(K)
    real, intent(inout), dimension(K) :: b, d
    real, intent(out) :: x(K)
        #  --- Local variables ---
    integer :: i
    real :: q
        #  --- Elimination ---
    do i = 2,K
       q = a(i)/b(i - 1)
       b(i) = b(i) - c(i - 1)*q
       d(i) = d(i) - d(i - 1)*q
    end do
        # --- Backsubstitution ---
    q = d(K)/b(K)
    x(K) = q
    do i = K - 1,1,-1
       q = (d(i) - c(i)*q)/b(i)
       x(i) = q
    end do
    return
    end def TDMA
#********************************************#
    def VI(N,DX,P,V)
    DIMENSION P(N,N),V(N,N)
    COMMON /COMAK/AK(0:64,0:64) # change this
    PAI1=0.2026423
    DO 40 I=1,N
    DO 40 J=1,N
    H0=0.0
    DO 30 K=1,N
    IK=IABS(I-K)
    DO 30 L=1,N
    JL=IABS(J-L)
30  H0=H0+AK(IK,JL)*P(K,L)
40  V(I,J)=H0*DX*PAI1
    RETURN
    END
#********************************************#

#**********************************************************************
    def OUTPUT(N,DX,X,Y,H,P)
    common /COMER/E1,RX,B,PH

    REAL,DIMENSION(N) ::  X,Y
    REAL,DIMENSION(N,N) ::H,P

    # Fluid film thickness
    A=0.0
    WRITE(8,110)A,(Y(I),I=1,N)
    DO I=1,N
     WRITE(8,110)X(I),(H(I,J),J=1,N)
    ENDDO

    # Pressure distribution
    WRITE(10,110)A,(Y(I),I=1,N)
    DO I=1,N
     WRITE(10,110)X(I),(P(I,J),J=1,N)
    ENDDO
    
110 FORMAT(66(E12.5,1X))  # change this
    RETURN
    END

# ******** TIME And DATE ******** #
    def GetDat(iyr,imon,iday)
    integer, intent(out):: iyr, imon, iday
    character(8):: sdate
    character(10):: stime
    call date_and_time(sdate,stime)
    read(sDate,"(I4,I2,I2)") iyr, imon, iday
    end def GetDat

    def GetTim(ihr,imin,isec,i100th)
    integer, intent(out):: ihr, imin, isec, i100th
    character(8):: sdate
    character(10):: stime
    call date_and_time(sdate,stime)
    read(sTime,"(I2,I2,I2,1x,I3)") ihr, imin, isec, i100th
    end def GetTim

# ******************************* #
    def FZ(N,P,POLD)
    IMPLICIT NONE
    
    INTEGER :: N,I,J
    REAL,DIMENSION(N,N) :: P,POLD
    
    DO I=1,N
      DO J=1,N
        POLD(I,J)=P(I,J)
      ENDDO
    ENDDO

    RETURN
    END def FZ
# ******************************* #
    def ROUGHNESS(N,DA,ROU,RMSA)
        character (len=1)       :: typeofroughness
        integer ::      KR,J,N,I,M
        real    ::      DA,N4,Yi,Yj,RMSR,RMSA,wx,wy,DX,U1,DT,U,Ymean
        real,dimension(N)       ::      X,Y
        real,dimension(N,N)     ::      ROU,DS
        DATA KR/0/
        
        OPEN(12,FILE='rough256.txt',STATUS='old',ACTION='READ') 
        READ(12,*,IOSTAT=M) ((ROU(I,J),J=1,N),I=1,N)
     #  WRITE(*,*) M
     #  READ (*,*)
      # OPEN(12,FILE='ROUGH2.DAT',STATUS='old',ACTION='READ')
      # DO I=1,N
      #   DO J=1,N
      #     READ(12,*) ROU(I,J)
      #   ENDDO
      # ENDDO
       close(12)
            
    #    Write(13,90) ((ROU(I,J),J=1,N),I=1,N)
    100 FORMAT(1X,F10.6)
    #90  FORMAT(64(1p,E16.6))
        N4=SIZE(ROU)
        Ymean=SUM(ROU)/N4
    75  Yi=0.0
        Yj=0.0
        do I=1,N
        do J=1,N
        ROU(I,J)=DA*(ROU(I,J)-Ymean)
       # Ymean=Ymean+ROU(I,J)
        Yj=Yj+ABS(ROU(I,J))
        enddo
        enddo
        RMSA=(1./N4)*Yj
        Ymean=SUM(ROU)/N4
        Yi=0.0
        Yj=0.0  
        DO I=1,N
          DO J=1,N
          #  IF(ABS(ROU(I,J)).GT.2.*RMSA) ROU(I,J)=2.*RMSA
            Yj=Yj+ABS(ROU(I,J))
            Yi=Yi+(ROU(I,J))**2
          ENDDO
        ENDDO
       
        RMSA=(1./N4)*Yj
        RMSR=sqrt((1./N4)*Yi)
        
        write(*,*)"DA,RMSR,RMSA", DA,Ymean,RMSA,RMSR
       # read(*,*)
        RETURN
#******************************************#
    def ITERSEMISEPI(N,KK,DX,DT,ERH,H00,G0,X,Y,H,HO0,RO,RO0,EPS,EDA,P,V,GOU,ROU)
        IMPLICIT NONE
    
        INTEGER :: N,KK
        INTEGER,DIMENSION(N,N) :: ID
        REAL,DIMENSION(N) ::  X,Y
        REAL,DIMENSION(N,N) :: P,POLD,H,RO,EPS,EDA,V,ROU,HO0,RO0,GOU,VPLASTIC
        REAL :: DX,H00,G0,DA
        REAL,DIMENSION(N) :: AP,BP,CP,FP,AW,BW,CW,FW,ATIME,BTIME,CTIME,FTIME,A,BARR,C,F
        REAL,DIMENSION(N)   :: P1D
        REAL :: ENDA,A1,A2,A3,Z,HM0,HM0AVG,DW,DH
       # REAL (kind=8) :: DH
        REAL :: AK
        REAL :: PAI1,AK00,AK10,AK20
        REAL :: DX1,DX2,DX3,DX3RHO,DXT
        INTEGER :: I,J,K,ICB,ICM,ICT,ICP,J0,J1,I0,I1
        INTEGER :: nx,ny,nh
        REAL :: D1,D2,D3,D4,D5
        REAL :: DF,DT
        REAL :: E1,RX,B,PH
        REAL :: Q1,Q2,Hardness,UPLASTIC,ERH,SUM
        
        COMMON /COM1/ENDA,A1,A2,A3,Z,HM0,HM0AVG,DH,DW
        COMMON /COMAK/AK(0:64,0:64) # change this
        COMMON /COMCONT/ICM,ICB,ICP
        common /COMER/E1,RX,B,PH
        COMMON /HARD/Hardness(1:65,1:65),UPLASTIC(1:65,1:65) # change this
        
        PAI1= 0.2026423
        
        AK00=AK(0,0)*PAI1
        AK10=AK(1,0)*PAI1
        AK20=AK(2,0)*PAI1
    
        DXT=0.0
    
        DX1=1./DX
        DX2=DX*DX
        DX3=1./DX2
        DX3RHO=DX3/RO(I,J)
        IF(DT.GT.0.0) DXT=1./DT
       # DXT=0.0
         ID=0
         
    155 DO K=1,KK
          ICB=0
          ICM=0
          ICT=0
          ICP=0
          DO J=N-1,2,-1
            J0=J-1
            J1=J+1
            D2=0.5*(EPS(1,J)+EPS(2,J))
          
          DO I=2,N-1
              I0=I-1
              I1=I+1
              D1=D2
              D2=0.5*(EPS(I1,J)+EPS(I,J))
              D4=0.5*(EPS(I,J0)+EPS(I,J))
              D5=0.5*(EPS(I,J1)+EPS(I,J))
              D3=D1+D2+D4+D5
          
            AP(I)=D1*DX3/RO(I,J)
            BP(I)=-D3*DX3/RO(I,J)
            CP(I)=D2*DX3/RO(I,J)
            FP(I)=-(D5*P(I,J+1)+D4*P(I,J-1))*DX3/RO(I,J)
        
        Q1=AK10*P(I-1,J)+AK00*P(I,J)+AK10*P(I+1,J)
        Q2=AK00*P(I-1,J)+AK10*P(I,J)+AK20*P(I+1,J)
    
         # IF(H(I,J).LE.0.0.AND.Hardness(I,J).LT.3.2E9/PH) ICB=ICB+1
         # IF(H(I,J).LE.0.0.AND.Hardness(I,J).GE.3.2E9/PH) ICM=ICM+1
           IF(H(I,J).LE.0.47E-9*RX/B/B) ICM=ICM+1
           
            AW(I)=-(AK10-AK00)*DX1
            BW(I)=-(AK00-AK10)*DX1
            CW(I)=-(AK10-AK20)*DX1
            FW(I)= ((H(I,J)-Q1)-(H(I-1,J)-Q2))*DX1+H(I,J)*(1.-(RO(I-1,J)/RO(I,J)))*DX1&
            &+(ROU(I,J)-ROU(I-1,J))*DX1#-(X(I))
            
            ATIME(I)=-AK10*DXT
            BTIME(I)=-AK00*DXT
            CTIME(I)=-AK10*DXT
            FTIME(I)= (H(I,J)-Q1)*DXT-(RO0(I,J)/RO(I,J))*HO0(I,J)*DXT
    
            A(I)=AP(I)+AW(I)+ATIME(I)
         BARR(I)=BP(I)+BW(I)+BTIME(I)
            C(I)=CP(I)+CW(I)+CTIME(I)
            F(I)=FP(I)+FW(I)+FTIME(I)
        
          ENDDO
       
          BARR(1)=1.0
          BARR(N)=1.0
          A(N)=0.0
          C(1)=0.0
          F(1)=0.0
          F(N)=0.0
         
          CALL TDMA(N,A,BARR,C,F,P1D)
        
          DO I=1,N
            P(I,J)=P1D(I)
    54      IF(P(I,J).LT.0.0)P(I,J)=0.0
          ENDDO      
        ENDDO
        
        CALL HREEI(N,DX,DF,DT,ERH,KK,H00,G0,X,Y,H,RO,EPS,EDA,P,V,GOU,ROU)    
        ENDDO
    
        write(*,*)"Bndry, Mtl, Total,plastic ",ICB,ICM,ICB+ICM,ICP   
        RETURN
#**********************************************************************
    def HREEI(N,DX,DF,DT,ERH,KK,H00,G0,X,Y,H,RO,EPS,EDA,P,V,GOU,ROU)
        IMPLICIT NONE
        
        INTEGER :: N,KK,nx,ny,nh
        REAL :: DX,DA,G0
        REAL,DIMENSION(N) :: X,Y
        REAL,DIMENSION(N,N) :: P,H,RO,EPS,EDA,V,ROU,GOU,ROU2
        REAL :: PAI,PAI1,KR,HMIN,RAD,W1,H0,DW,EDA1,HM0AVG,G00
        INTEGER :: NN,I,J,IEQ,COUNT
        REAL :: ENDA,A1,A2,A3,Z,HM0,DH,H00,ERH
      #  REAL (kind=8) :: DH,H00,ERH
        REAL :: AK
        REAL :: E1,RX,B,PH
        REAL :: ATEMP1,ATEMP2,ATEMP3,RS,XA,XS,DF
        REAL :: U1,U2,UR,US,W,DT,C3,Hardness,UPLA,UPLASTIC,H00OLD,ptemp
    
        COMMON /COM1/ENDA,A1,A2,A3,Z,HM0,HM0AVG,DH,DW,G00
        COMMON /COMAK/AK(0:64,0:64) # change this
        COMMON /COMER/E1,RX,B,PH
        COMMON /COM3/U1,U2,UR,US,W
        COMMON /HARD/Hardness(1:65,1:65),UPLASTIC(1:65,1:65) # change this
        
        PAI = 3.14159265
        PAI1= 0.2026423
        C3=0.005
        nx = 2*N
        ny = nx
        nh = nx / 2  + 1
        
        NN=(N+1)/2
        KR=0
            
        CALL VI(N,DX,P,V)
    
        HMIN=1.E3
        RS=0.1
        XS=-0.0
        XA=XS+(U1/UR)*DT
     #   write(*,*) DF
        DF=0.0
      #  write(*,*) DF
        DO I=1,N
        ATEMP1=(X(I)-XA)**2
        DO J=1,N
         ATEMP2=ATEMP1+Y(J)**2
         ATEMP3=RS**2-ATEMP2
         IF(SQRT(ATEMP2).GE.RS) ATEMP3=0.0
        # ROU(I,J)=ROU(I,J)+(DF/RS)*SQRT(ATEMP3)
         W1=GOU(I,J)+ROU(I,J)
         W1=W1-(DF/RS)*SQRT(ATEMP3)  #+UPLASTIC(I,J)
         H0=W1+V(I,J)#-UPLASTIC(I,J)
         IF(H0.LT.HMIN)HMIN=H0
         H(I,J)=H0
        ENDDO
        ENDDO
        
        IF(KK.EQ.0)THEN
         KK=1
         DH=0.005*HMIN
         H00=0.
         Write(*,*)"START H00",H00
        ENDIF
        
        W1=0.0
        COUNT=0
        DO I=1,N
          DO J=1,N      
            ptemp=P(I,J)        
            W1=W1+ptemp        
          ENDDO
        ENDDO
        
        H00OLD=H00
        W1=DX*DX*W1/G0
        DW=1.-W1    
        H00=H00-0.1*DW
        
        ERH=H00-H00OLD
    
        DO I=1,N
          DO J=1,N
          
            H(I,J)=H00+H(I,J)
            EDA1=EXP(A1*(-1.+(1.+A2*P(I,J))**Z))
            EDA(I,J)=EDA1
            RO(I,J)=(A3+1.34*P(I,J))/(A3+P(I,J))
            EPS(I,J)=RO(I,J)*H(I,J)**3/(ENDA*EDA(I,J))
    
           IF(H(I,J).LE.0.47E-9*RX/B/B) EPS(I,J)=0.0   # asperity contact    
          ENDDO
        ENDDO
    
        RETURN
    
#******************************************#
