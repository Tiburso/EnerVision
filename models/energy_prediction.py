import numpy as np 
# refer to : https://github.com/pvlib/pvlib-python/blob/main/docs/examples/iv-modeling/plot_singlediode.py

class Energy_prediction():
    def __init__(self,zenith): 
        self.zenith = zenith
        self.irradiance = 1 
        self.E0 = 1000 # reference solar irradiance
        self.Eb = None # Beam irradiance on the plane of array
        self.Ed = None # Diffuse irradiance on the plane of array
        self.Ee = Ee(Isc,Isc_0) # effective irradiance

        self.T0 = 25 # reference cell temperature
        self.TC = None # cell temperature
        self.fd = 1 # fraction of diffuse light used by the module 1 for flat plate module
        self.Am = 1/np.cos(self.zenith)
        self.f1 = f1(a) # air mass modifier
        self.f2 = f2(AOI,b) # angle of Incidence modifier

        self.alpha_isc = None #normalized temperature coefficient for short circuit current. Units are 1/C
        self.alpha_lmp = None #normalized temperature coefficient for maximum power current. Units are 1/c
        self.C = None #Vector of coeffients to update predictions 
        
        self.delta = delta(n)

        def f1(a):
            AM_a = self.AM
            return a[0] + a[1]*AM_a + a[2]*AM_a**2 + a[3]* AM_a **2 + a[4]* AM_a**2

        def f2(AOI, b):
            return b[0] + b[1] *AOI + b[2] *AOI**2 + b[3]* AOI **2 + b[4]* AOI**2 + b[5]* AOI**2
        
        def Ee(Isc,Isc_0):
            return Isc/ (Isc_0*(1+ self.alpha_isc(self.TC-self.T0)))
        
        def delta(n):
            k = 138.066e-23 # boltzmann's constant
            q = 1.60218e-19 # elementary charge constant
            #n =  Empirically determined diode factor
            return n * k (self.Tc +273.15) /q
        
        def beta_voc():
            m_bvoc = 0 # coefficient describing the irradiance dependence for the open circuit voltage temperature coefficient (typically equals zero)
            beta_voc0 = None #is the temperature coefficient for module open circuit voltage at irradiance conditions of 1000 W/m^2
            return beta_voc0 + m_bvoc(1-self.Ee)
    def sandia_prediction():
        # Define SAPM primary points 
        """
        Missing: 
        _0 at beginning,
        self.alpha beta_Vmp
        self.Tc
        Ns
        """
        Isc = Isc_0 * self.f1 (self.Eb *self.f2 + self.fd *self.Ed)* (1 + self.aplha_isc * (self.Tc -self.T0))
        Imp = Imp_0 * (self.C[0]*self.Ee +self.C[1]*self.Ee**2)*(1 + self.aplha_imp * (self.Tc -self.T0))
        Voc = Voc_0 + Ns * self.delta* np.log(self.Ee) + beta_Voc * (self.Tc - self.T0) # log = ln
        Vmp = Vmp_0 + self.C[2]* Ns *self.delta *np.log(self.Ee) + self.C[3] *Ns *(self.delta *np.log(self.Ee))**2 + beta_Vmp* (self.Tc -self.T0)

        


