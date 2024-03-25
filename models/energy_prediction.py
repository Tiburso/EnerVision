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

        self.C = None #Vector of coeffients to update predictions 

        self.Isc= None #3x1
        self.Imp= None #first value is Isc/Imp/Voc/Vmp/_0 second is alpha or beta value third is to be calculated
        self.Voc= None
        self.Vmp= None
        
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

        self.Isc[3] = self.Isc[0] * self.f1 (self.Eb *self.f2 + self.fd *self.Ed)* (1 + self.Isc[2] * (self.Tc -self.T0))
        self.Imp[3] = self.Imp[0] * (self.C[0]*self.Ee +self.C[1]*self.Ee**2)*(1 + self.Imp[2] * (self.Tc -self.T0))
        self.Voc[3] = self.Voc[0] + Ns * self.delta* np.log(self.Ee) + self.Voc[2] * (self.Tc - self.T0) # log = ln
        self.Vmp[3] = self.Vmp[0] + self.C[2]* Ns *self.delta *np.log(self.Ee) + self.C[3] *Ns *(self.delta *np.log(self.Ee))**2 + self.Vmp[2]* (self.Tc -self.T0)

        


