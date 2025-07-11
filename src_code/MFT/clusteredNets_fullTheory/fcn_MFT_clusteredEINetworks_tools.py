
"""
Set of functions for MFT calculations of clustered networks
"""

import numpy as np
import fcn_MFT_general_tools

#%% GENERAL FUNCTIONS


#%%

'''
COMPUTE WEIGHT AND DEGREE MATRICES FOR ALL DYNAMICAL POPULATIONS GIVEN SYNAPTIC
CONNECTIVITY PARAMETERS OF A CLUSTERED NETWORK
'''

def fcn_compute_weight_degree_mat(\
                   Cee, Cei, Cie, Cii, Cext, \
                   Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                   Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                   fE, fEb, fI, fIb, n_dyn_e_pops, n_dyn_i_pops):
    
    
    
    # NUMBER OF DYNAMICAL CLUSTERS/BG POPS OF EACH TYPE
    n_Eclu = n_dyn_e_pops - 1
    n_Iclu = n_dyn_i_pops - 1
    
    # COMPUTE WEIGHT AND DEGREE MATRICES
    Jab = np.zeros((n_dyn_e_pops+n_dyn_i_pops, n_dyn_e_pops+n_dyn_i_pops))
    Cab = np.zeros((n_dyn_e_pops+n_dyn_i_pops, n_dyn_e_pops+n_dyn_i_pops))
    Jab_ext = np.zeros(n_dyn_e_pops+n_dyn_i_pops)
    Cab_ext = np.zeros(n_dyn_e_pops+n_dyn_i_pops)   
    
    
    # inputs to E clusters ---------------------------------------------------
    for i in range(0, n_Eclu):

        # input from E clusters
        for j in range(0, n_Eclu):
            
            # input from same cluster
            if j == i:
                
                Jab[i,j] = Jee_p
                
            # input from different cluster
            elif j!=i:
                
                Jab[i,j] = Jee_m
                
            
            Cab[i,j] = fE*Cee

                
        # input from E background
        j = n_dyn_e_pops - 1
        Jab[i,j] = Jee_m
        Cab[i,j] = fEb*Cee
        
        # input from I clusters
        for j in range(0, n_Iclu):
            
            # input from same cluster
            if j == i:
                
                Jab[i,j+n_dyn_e_pops] = -Jei_p
                
            # input from different cluster
            elif j!=i:
                
                Jab[i,j+n_dyn_e_pops] = -Jei_m
                

            Cab[i,j+n_dyn_e_pops] = fI*Cei  
            

                
        # input from I background
        j = n_dyn_i_pops - 1
        Jab[i,j+n_dyn_e_pops] = -Jei_m
        Cab[i,j+n_dyn_e_pops] = fIb*Cei 
                               
        # input from external sources
        Jab_ext[i] = Jee_ext
        Cab_ext[i] = Cext
                
        
    # inputs to E background---------------------------------------------------
    i = n_dyn_e_pops-1
    
    # input from E clusters
    for j in range(0,n_Eclu):

        Jab[i,j] = Jee_m
        
        Cab[i,j] = fE*Cee
            
        
    # input from E background
    j = n_dyn_e_pops-1
    Jab[i,j] = Jee
    Cab[i,j] = fEb*Cee
    
    
    # input from I clusters
    for j in range(0, n_Iclu):

        Jab[i,j+n_dyn_e_pops] = -Jei_m
        
        Cab[i,j+n_dyn_e_pops] = fI*Cei 
        
            
    # input from I background
    j = n_dyn_i_pops - 1
    Jab[i,j+n_dyn_e_pops] = -Jei
    Cab[i,j+n_dyn_e_pops] = fIb*Cei      
     
    # input from external sources
    Jab_ext[i] = Jee_ext
    Cab_ext[i] = Cext


    # inputs to I clusters---------------------------------------------------
    for i in range(0, n_Iclu):
        
        # input from E clusters
        for j in range(0, n_Eclu):     
            
            # input from same cluster
            if j == i:
                Jab[i+n_dyn_e_pops,j] = Jie_p
                
            # input from different cluster
            elif j!=i:
                Jab[i+n_dyn_e_pops,j] = Jie_m
                

            Cab[i+n_dyn_e_pops,j] = fE*Cie
            
                                
        # input from E background
        j = n_dyn_e_pops - 1
        Jab[i+n_dyn_e_pops,j] = Jie_m
        Cab[i+n_dyn_e_pops,j] = fEb*Cie
                
                
        # input from I clusters
        for j in range(0, n_Iclu):
            
            # input from same cluster
            if j == i:
                Jab[i+n_dyn_e_pops,j+n_dyn_e_pops] = -Jii_p
                            
            # input from different cluster
            elif j!=i:
                Jab[i+n_dyn_e_pops,j+n_dyn_e_pops] = -Jii_m

            
            Cab[i+n_dyn_e_pops,j+n_dyn_e_pops] = fI*Cii


        # input from I background
        j = n_dyn_i_pops - 1
        Jab[i+n_dyn_e_pops,j+n_dyn_e_pops] = -Jii_m
        Cab[i+n_dyn_e_pops,j+n_dyn_e_pops] = fIb*Cii 
        
        # input from external sources
        Jab_ext[i+n_dyn_e_pops] = Jie_ext
        Cab_ext[i+n_dyn_e_pops] = Cext
        
        
        
    # inputs to I background---------------------------------------------------
    i = n_dyn_i_pops-1
    
    # input from E clusters
    for j in range(0,n_Eclu,1):

        Jab[i+n_dyn_e_pops,j] = Jie_m
        
        Cab[i+n_dyn_e_pops,j] = fE*Cie
        
                        
    # input from E background
    j = n_dyn_e_pops-1
    Jab[i+n_dyn_e_pops,j] = Jie
    Cab[i+n_dyn_e_pops,j] = fEb*Cie
    
    # input from I clusters
    for j in range(0, n_Iclu):
        
        Jab[i+n_dyn_e_pops,j+n_dyn_e_pops] = -Jii_m
        
        Cab[i+n_dyn_e_pops,j+n_dyn_e_pops] = fI*Cii
            
    # input from I background
    j = n_dyn_i_pops - 1
    Jab[i+n_dyn_e_pops,j+n_dyn_e_pops] = -Jii
    Cab[i+n_dyn_e_pops,j+n_dyn_e_pops] = fIb*Cii     
      
    # input from external sources
    Jab_ext[i+n_dyn_e_pops] = Jie_ext
    Cab_ext[i+n_dyn_e_pops] = Cext
    
    
    return Jab, Cab, Jab_ext, Cab_ext



#%%

'''

COMPUTE WEIGHT AND DEGREE MATRICES FOR REDUCED SYSTEM OF 6 DYNAMICAL POPULATIONS

this method reduces the full problem to three dynamical populations:
  active clusters, inactive clusters, background

compute Jab, Cab, Jab_ext, Cab_ext for reduced network of three populations

including E and I populations, we have a total of 6 populations
assumes that 0: active, 1: inactive, 2: background

'''


def fcn_compute_weight_degree_mat_reduced(\
                                     Cee, Cei, Cie, Cii, Cext, \
                                         Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                             Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                                 fE, fEb, fI, fIb, nClusters, n_activeClusters):
            
        
        
    # number inactive
    n_inactiveClusters = nClusters - n_activeClusters
  
    
    # input to E active from all E
    Jee_active = np.array([Jee_p, 0, Jee_m, Jee_m, Jee_m])
    Cee_active = np.array([fE*Cee*1, 0, fE*Cee*(n_activeClusters-1), fE*Cee*(n_inactiveClusters), fEb*Cee])
    
    # input to E active from all I
    Jei_active = np.array([-Jei_p, 0, -Jei_m, -Jei_m, -Jei_m])
    Cei_active = np.array([fI*Cei*1, 0, fI*Cei*(n_activeClusters-1), fI*Cei*(n_inactiveClusters), fIb*Cei])
    
    # input to E inactive from all E
    Jee_inactive = np.array([0, Jee_p, Jee_m, Jee_m, Jee_m])
    Cee_inactive = np.array([0, fE*Cee*1, fE*Cee*(n_activeClusters), fE*Cee*(n_inactiveClusters-1), fEb*Cee])
    
    # input to E inactive from all I
    Jei_inactive = np.array([0, -Jei_p, -Jei_m, -Jei_m, -Jei_m])
    Cei_inactive = np.array([0, fI*Cei*1, fI*Cei*(n_activeClusters), fI*Cei*(n_inactiveClusters-1), fIb*Cei])
    
    # input to E background from all E
    Jee_back = np.array([0, 0, Jee_m, Jee_m, Jee])
    Cee_back = np.array([0, 0, fE*Cee*(n_activeClusters), fE*Cee*(n_inactiveClusters), fEb*Cee])
    
    # input to E background from all I
    Jei_back = np.array([0, 0, -Jei_m, -Jei_m, -Jei])
    Cei_back = np.array([0, 0, fI*Cei*(n_activeClusters), fI*Cei*(n_inactiveClusters), fIb*Cei])
    
    
        
    # input to I active from all E
    Jie_active = np.array([Jie_p, 0, Jie_m, Jie_m, Jie_m])
    Cie_active = np.array([fE*Cie*1, 0, fE*Cie*(n_activeClusters-1), fE*Cie*(n_inactiveClusters), fEb*Cie])
    
    # input to I active from all I
    Jii_active = np.array([-Jii_p, 0, -Jii_m, -Jii_m, -Jii_m])
    Cii_active = np.array([fI*Cii*1, 0, fI*Cii*(n_activeClusters-1), fI*Cii*(n_inactiveClusters), fIb*Cii])
    
    # input to I inactive from all E
    Jie_inactive = np.array([0, Jie_p, Jie_m, Jie_m, Jie_m])
    Cie_inactive = np.array([0, fE*Cie*1, fE*Cie*(n_activeClusters), fE*Cie*(n_inactiveClusters-1), fEb*Cie])
    
    # input to I inactive from all I
    Jii_inactive = np.array([0, -Jii_p, -Jii_m, -Jii_m, -Jii_m])
    Cii_inactive = np.array([0, fI*Cii*1, fI*Cii*(n_activeClusters), fI*Cii*(n_inactiveClusters-1), fIb*Cii])
    
    # input to I background from all E
    Jie_back = np.array([0, 0, Jie_m, Jie_m, Jie])
    Cie_back = np.array([0, 0, fE*Cie*(n_activeClusters), fE*Cie*(n_inactiveClusters), fEb*Cie])
    
    # input to I background from all I
    Jii_back = np.array([0, 0, -Jii_m, -Jii_m, -Jii])
    Cii_back = np.array([0, 0, fI*Cii*(n_activeClusters), fI*Cii*(n_inactiveClusters), fIb*Cii])
    
    
    # put it all together
    Je_active = np.hstack((Jee_active, Jei_active))
    Je_inactive = np.hstack((Jee_inactive, Jei_inactive))
    Je_back = np.hstack((Jee_back, Jei_back))
    
    Ce_active = np.hstack((Cee_active, Cei_active))
    Ce_inactive = np.hstack((Cee_inactive, Cei_inactive))
    Ce_back = np.hstack((Cee_back, Cei_back))

    Ji_active = np.hstack((Jie_active, Jii_active))
    Ji_inactive = np.hstack((Jie_inactive, Jii_inactive))
    Ji_back = np.hstack((Jie_back, Jii_back))
    
    Ci_active = np.hstack((Cie_active, Cii_active))
    Ci_inactive = np.hstack((Cie_inactive, Cii_inactive))
    Ci_back = np.hstack((Cie_back, Cii_back))
    
    Jab = np.vstack((Je_active, Je_inactive, Je_back, Ji_active, Ji_inactive, Ji_back))
    Cab = np.vstack((Ce_active, Ce_inactive, Ce_back, Ci_active, Ci_inactive, Ci_back))
    
    Cab_ext = np.ones(6) * Cext
    Jab_ext = np.append(np.ones(3)*Jee_ext, np.ones(3)*Jie_ext)
    
   
    
    return Jab, Cab, Jab_ext, Cab_ext


#%%
'''

COMPUTE WEIGHT AND DEGREE MATRICES FOR EFFECTIVE MFT

this method reduces the full problem to 5 dynamical populations:
  2 in focus clusters, out of focus active clusters, inactive clusters, background

compute Jab, Cab, Jab_ext, Cab_ext for reduced effective network 

including E and I populations, we have a total of 10 populations
assumes that 0,1: in-focus, 2: out of focus active, 3: out of focus inactive, 4: out of focus background
assumes that the two in-focus clusters can have two different rates

'''


def fcn_compute_weight_degree_mat_effective_reduced(\
                                     Cee, Cei, Cie, Cii, Cext, \
                                         Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                             Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                                 fE, fEb, fI, fIb, nClusters, n_activeClusters):
            
        
        
    # number inactive
    n_inactiveClusters = nClusters - n_activeClusters
    
    if n_activeClusters > 1:
    
        # input to in-focus E clusters from all E
        Jee_infocusA = np.array([Jee_p, Jee_m, Jee_m, Jee_m, Jee_m, Jee_m, Jee_m])
        Cee_infocusA = np.array([fE*Cee*1, fE*Cee*1, fE*Cee*(n_activeClusters-1), fE*Cee*(n_inactiveClusters-1), 0, 0, fEb*Cee])
        
        Jee_infocusB = np.array([Jee_m, Jee_p, Jee_m, Jee_m, Jee_m, Jee_m, Jee_m])
        Cee_infocusB = np.array([fE*Cee*1, fE*Cee*1, fE*Cee*(n_activeClusters-1), fE*Cee*(n_inactiveClusters-1), 0, 0, fEb*Cee])
    
        # input to E active from all E
        Jee_active = np.array([Jee_m, Jee_m, Jee_p, Jee_m, Jee_m, Jee_m, Jee_m])
        Cee_active = np.array([fE*Cee*1, fE*Cee*1, fE*Cee*1, fE*Cee*(n_inactiveClusters-1), fE*Cee*(n_activeClusters-2), 0, fEb*Cee])
        
        # input to E inactive from all E
        Jee_inactive = np.array([Jee_m, Jee_m, Jee_m, Jee_p, Jee_m, Jee_m, Jee_m])
        Cee_inactive = np.array([fE*Cee*1, fE*Cee*1, fE*Cee*(n_activeClusters-1), fE*Cee*1, 0, fE*Cee*(n_inactiveClusters-2), fEb*Cee])
        
        # input to E background from all E
        Jee_back = np.array([Jee_m, Jee_m, Jee_m, Jee_m, Jee_m, Jee_m, Jee])
        Cee_back = np.array([fE*Cee, fE*Cee, fE*Cee*(n_activeClusters-1), fE*Cee*(n_inactiveClusters-1), 0, 0, fEb*Cee])   
        
        
        
        # input to in-focus E clusters from all I
        Jei_infocusA = np.array([Jei_p, Jei_m, Jei_m, Jei_m, Jei_m, Jei_m, Jei_m])*(-1)
        Cei_infocusA = np.array([fI*Cei*1, fI*Cei*1, fI*Cei*(n_activeClusters-1), fI*Cei*(n_inactiveClusters-1), 0, 0, fIb*Cei])
        
        Jei_infocusB = np.array([Jei_m, Jei_p, Jei_m, Jei_m, Jei_m, Jei_m, Jei_m])*(-1)
        Cei_infocusB = np.array([fI*Cei*1, fI*Cei*1, fI*Cei*(n_activeClusters-1), fI*Cei*(n_inactiveClusters-1), 0, 0, fIb*Cei])
    
        # input to E active from all I
        Jei_active = np.array([Jei_m, Jei_m, Jei_p, Jei_m, Jei_m, Jei_m, Jei_m])*(-1)
        Cei_active = np.array([fI*Cei*1, fI*Cei*1, fI*Cei*1, fI*Cei*(n_inactiveClusters-1), fI*Cei*(n_activeClusters-2), 0, fIb*Cei])
        
        # input to E inactive from all I
        Jei_inactive = np.array([Jei_m, Jei_m, Jei_m, Jei_p, Jei_m, Jei_m, Jei_m])*(-1)
        Cei_inactive = np.array([fI*Cei*1, fI*Cei*1, fI*Cei*(n_activeClusters-1), fI*Cei*1, 0, fI*Cei*(n_inactiveClusters-2), fIb*Cei])
        
        # input to E background from all I
        Jei_back = np.array([Jei_m, Jei_m, Jei_m, Jei_m, Jei_m, Jei_m, Jei])*(-1)
        Cei_back = np.array([fI*Cei, fI*Cei, fI*Cei*(n_activeClusters-1), fI*Cei*(n_inactiveClusters-1), 0, 0, fIb*Cei])      
        
        
        
        # input to in-focus I clusters from all E
        Jie_infocusA = np.array([Jie_p, Jie_m, Jie_m, Jie_m, Jie_m, Jie_m, Jie_m])
        Cie_infocusA = np.array([fE*Cie*1, fE*Cie*1, fE*Cie*(n_activeClusters-1), fE*Cie*(n_inactiveClusters-1), 0, 0, fEb*Cie])
        
        Jie_infocusB = np.array([Jie_m, Jie_p, Jie_m, Jie_m, Jie_m, Jie_m, Jie_m])
        Cie_infocusB = np.array([fE*Cie*1, fE*Cie*1, fE*Cie*(n_activeClusters-1), fE*Cie*(n_inactiveClusters-1), 0, 0, fEb*Cie])
    
        # input to I active from all E
        Jie_active = np.array([Jie_m, Jie_m, Jie_p, Jie_m, Jie_m, Jie_m, Jie_m])
        Cie_active = np.array([fE*Cie*1, fE*Cie*1, fE*Cie*1, fE*Cie*(n_inactiveClusters-1), fE*Cie*(n_activeClusters-2), 0, fEb*Cie])
        
        # input to I inactive from all E
        Jie_inactive = np.array([Jie_m, Jie_m, Jie_m, Jie_p, Jie_m, Jie_m, Jie_m])
        Cie_inactive = np.array([fE*Cie*1, fE*Cie*1, fE*Cie*(n_activeClusters-1), fE*Cie*1, 0, fE*Cie*(n_inactiveClusters-2), fEb*Cie])
        
        # input to E background from all E
        Jie_back = np.array([Jie_m, Jie_m, Jie_m, Jie_m, Jie_m, Jie_m, Jie])
        Cie_back = np.array([fE*Cie, fE*Cie, fE*Cie*(n_activeClusters-1), fE*Cie*(n_inactiveClusters-1), 0, 0, fEb*Cie])  
    
    
    
        # input to in-focus I clusters from all I
        Jii_infocusA = np.array([Jii_p, Jii_m, Jii_m, Jii_m, Jii_m, Jii_m, Jii_m])*(-1)
        Cii_infocusA = np.array([fI*Cii*1, fI*Cii*1, fI*Cii*(n_activeClusters-1), fI*Cii*(n_inactiveClusters-1), 0, 0, fIb*Cii])
        
        Jii_infocusB = np.array([Jii_m, Jii_p, Jii_m, Jii_m, Jii_m, Jii_m, Jii_m])*(-1)
        Cii_infocusB = np.array([fI*Cii*1, fI*Cii*1, fI*Cii*(n_activeClusters-1), fI*Cii*(n_inactiveClusters-1), 0, 0, fIb*Cii])
    
        # input to I active from all E
        Jii_active = np.array([Jii_m, Jii_m, Jii_p, Jii_m, Jii_m, Jii_m, Jii_m])*(-1)
        Cii_active = np.array([fI*Cii*1, fI*Cii*1, fI*Cii*1, fI*Cii*(n_inactiveClusters-1), fI*Cii*(n_activeClusters-2), 0, fIb*Cii])
        
        # input to I inactive from all E
        Jii_inactive = np.array([Jii_m, Jii_m, Jii_m, Jii_p, Jii_m, Jii_m, Jii_m])*(-1)
        Cii_inactive = np.array([fI*Cii*1, fI*Cii*1, fI*Cii*(n_activeClusters-1), fI*Cii*1, 0, fI*Cii*(n_inactiveClusters-2), fIb*Cii])
        
        # input to E background from all E
        Jii_back = np.array([Jii_m, Jii_m, Jii_m, Jii_m, Jii_m, Jii_m, Jii])*(-1)
        Cii_back = np.array([fI*Cii, fI*Cii, fI*Cii*(n_activeClusters-1), fI*Cii*(n_inactiveClusters-1), 0, 0, fIb*Cii])  
        
        
        # put it all together
        Je_infocusA = np.hstack((Jee_infocusA, Jei_infocusA))
        Je_infocusB = np.hstack((Jee_infocusB, Jei_infocusB))
        Je_active = np.hstack((Jee_active, Jei_active))
        Je_inactive = np.hstack((Jee_inactive, Jei_inactive))
        Je_back = np.hstack((Jee_back, Jei_back))
        
        Ce_infocusA = np.hstack((Cee_infocusA, Cei_infocusA))
        Ce_infocusB = np.hstack((Cee_infocusB, Cei_infocusB))
        Ce_active = np.hstack((Cee_active, Cei_active))
        Ce_inactive = np.hstack((Cee_inactive, Cei_inactive))
        Ce_back = np.hstack((Cee_back, Cei_back))
    
        Ji_infocusA = np.hstack((Jie_infocusA, Jii_infocusA))
        Ji_infocusB = np.hstack((Jie_infocusB, Jii_infocusB))
        Ji_active = np.hstack((Jie_active, Jii_active))
        Ji_inactive = np.hstack((Jie_inactive, Jii_inactive))
        Ji_back = np.hstack((Jie_back, Jii_back))
        
        Ci_infocusA = np.hstack((Cie_infocusA, Cii_infocusA))
        Ci_infocusB = np.hstack((Cie_infocusB, Cii_infocusB))
        Ci_active = np.hstack((Cie_active, Cii_active))
        Ci_inactive = np.hstack((Cie_inactive, Cii_inactive))
        Ci_back = np.hstack((Cie_back, Cii_back))
        
        Jab = np.vstack((Je_infocusA, Je_infocusB, Je_active, Je_inactive, Je_back, \
                         Ji_infocusA, Ji_infocusB, Ji_active, Ji_inactive, Ji_back))
            
        Cab = np.vstack((Ce_infocusA, Ce_infocusB, Ce_active, Ce_inactive, Ce_back, \
                         Ci_infocusA, Ci_infocusB, Ci_active, Ci_inactive, Ci_back))
        
        Cab_ext = np.ones(10) * Cext
        Jab_ext = np.append(np.ones(5)*Jee_ext, np.ones(5)*Jie_ext)
        
    
    else:
        
        
        # input to in-focus E clusters from all E
        Jee_infocusA = np.array([Jee_p, Jee_m, Jee_m, Jee_m, Jee_m])
        Cee_infocusA = np.array([fE*Cee*1, fE*Cee*1, 0, fE*Cee*(nClusters-2), fEb*Cee])
        
        Jee_infocusB = np.array([Jee_m, Jee_p, Jee_m, Jee_m, Jee_m])
        Cee_infocusB = np.array([fE*Cee*1, fE*Cee*1, 0, fE*Cee*(nClusters-2), fEb*Cee])
    
        # input to E out-of-focus clusters from all E 
        Jee_outFocus = np.array([Jee_m, Jee_m, Jee_p, Jee_m, Jee_m])
        Cee_outFocus = np.array([fE*Cee*1, fE*Cee*1, fE*Cee*1, fE*Cee*(nClusters-3), fEb*Cee])*(int(nClusters)-2>0)
        
        # input to E background from all E
        Jee_back = np.array([Jee_m, Jee_m, Jee_m, Jee_m, Jee])
        Cee_back = np.array([fE*Cee, fE*Cee, 0, fE*Cee*(nClusters-2), fEb*Cee])  


        # input to in-focus E clusters from all I
        Jei_infocusA = np.array([Jei_p, Jei_m, Jei_m, Jei_m, Jei_m])*(-1)
        Cei_infocusA = np.array([fI*Cei*1, fI*Cei*1, 0, fI*Cei*(nClusters-2), fIb*Cei])
        
        Jei_infocusB = np.array([Jei_m, Jei_p, Jei_m, Jei_m, Jei_m])*(-1)
        Cei_infocusB = np.array([fI*Cei*1, fI*Cei*1, 0, fI*Cei*(nClusters-2), fIb*Cei])
    
        # input to E out-of-focus clusters from all I
        Jei_outFocus = np.array([Jei_m, Jei_m, Jei_p, Jei_m, Jei_m])*(-1)
        Cei_outFocus = np.array([fI*Cei*1, fI*Cei*1, fI*Cei*1, fI*Cei*(nClusters-3), fIb*Cei])*(int(nClusters)-2>0)
        
        # input to E background from all I
        Jei_back = np.array([Jei_m, Jei_m, Jei_m, Jei_m, Jei])*(-1)
        Cei_back = np.array([fI*Cei, fI*Cei, 0, fI*Cei*(nClusters-2), fIb*Cei])     
    
    
        # input to in-focus I clusters from all E
        Jie_infocusA = np.array([Jie_p, Jie_m, Jie_m, Jie_m, Jie_m])
        Cie_infocusA = np.array([fE*Cie*1, fE*Cie*1, 0, fE*Cie*(nClusters-2), fEb*Cie])
        
        Jie_infocusB = np.array([Jie_m, Jie_p, Jie_m, Jie_m, Jie_m])
        Cie_infocusB = np.array([fE*Cie*1, fE*Cie*1, 0, fE*Cie*(nClusters-2), fEb*Cie])
    
        # input to I out-of-focus clusters from all E 
        Jie_outFocus = np.array([Jie_m, Jie_m, Jie_p, Jie_m, Jie_m])
        Cie_outFocus = np.array([fE*Cie*1, fE*Cie*1, fE*Cie*1, fE*Cie*(nClusters-3), fEb*Cie])*(int(nClusters)-2>0)
        
        # input to I background from all E
        Jie_back = np.array([Jie_m, Jie_m, Jie_m, Jie_m, Jie])
        Cie_back = np.array([fE*Cie, fE*Cie, 0, fE*Cie*(nClusters-2), fEb*Cie]) 
        

        # input to in-focus I clusters from all I
        Jii_infocusA = np.array([Jii_p, Jii_m, Jii_m, Jii_m, Jii_m])*(-1)
        Cii_infocusA = np.array([fI*Cii*1, fI*Cii*1, 0, fI*Cii*(nClusters-2), fIb*Cii])
        
        Jii_infocusB = np.array([Jii_m, Jii_p, Jii_m, Jii_m, Jii_m])*(-1)
        Cii_infocusB = np.array([fI*Cii*1, fI*Cii*1, 0, fI*Cii*(nClusters-2), fIb*Cii])
    
        # input to E out-of-focus clusters from all i 
        Jii_outFocus = np.array([Jii_m, Jii_m, Jii_p, Jii_m, Jii_m])*(-1)
        Cii_outFocus = np.array([fI*Cii*1, fI*Cii*1, fI*Cii*1, fI*Cii*(nClusters-3), fIb*Cii])*(int(nClusters)-2>0)
        
        # input to E background from all i
        Jii_back = np.array([Jii_m, Jii_m, Jii_m, Jii_m, Jii])*(-1)
        Cii_back = np.array([fI*Cii, fI*Cii, 0, fI*Cii*(nClusters-2), fIb*Cii])
        

        # put it all together
        Je_infocusA = np.hstack((Jee_infocusA, Jei_infocusA))
        Je_infocusB = np.hstack((Jee_infocusB, Jei_infocusB))
        Je_outFocus = np.hstack((Jee_outFocus, Jei_outFocus))
        Je_back = np.hstack((Jee_back, Jei_back))
        
        Ce_infocusA = np.hstack((Cee_infocusA, Cei_infocusA))
        Ce_infocusB = np.hstack((Cee_infocusB, Cei_infocusB))
        Ce_outFocus = np.hstack((Cee_outFocus, Cei_outFocus))
        Ce_back = np.hstack((Cee_back, Cei_back))
    
        Ji_infocusA = np.hstack((Jie_infocusA, Jii_infocusA))
        Ji_infocusB = np.hstack((Jie_infocusB, Jii_infocusB))
        Ji_outFocus = np.hstack((Jie_outFocus, Jii_outFocus))
        Ji_back = np.hstack((Jie_back, Jii_back))
        
        Ci_infocusA = np.hstack((Cie_infocusA, Cii_infocusA))
        Ci_infocusB = np.hstack((Cie_infocusB, Cii_infocusB))
        Ci_outFocus = np.hstack((Cie_outFocus, Cii_outFocus))
        Ci_back = np.hstack((Cie_back, Cii_back))
        
        Jab = np.vstack((Je_infocusA, Je_infocusB, Je_outFocus, Je_back, \
                         Ji_infocusA, Ji_infocusB, Ji_outFocus, Ji_back))
            
        Cab = np.vstack((Ce_infocusA, Ce_infocusB, Ce_outFocus, Ce_back, \
                         Ci_infocusA, Ci_infocusB, Ci_outFocus, Ci_back))
        
        Cab_ext = np.ones(8) * Cext
        Jab_ext = np.append(np.ones(4)*Jee_ext, np.ones(4)*Jie_ext)
    
   
    
    return Jab, Cab, Jab_ext, Cab_ext







#%%
'''

COMPUTE WEIGHT AND DEGREE MATRICES FOR EFFECTIVE MFT
SIMPLEST CASE OF ONLY 1 GROUPED OUT OF FOCUS POPULATION

this method reduces the full problem to 4 dynamical populations:
  2 in focus groups (each group corresponding to n clusters w/ same rate)
  1 out of focus group (p-2*n clusters with the same rate)
  1 background

compute Jab, Cab, Jab_ext, Cab_ext for reduced effective network 

including E and I populations, we have a total of 8 populations
assumes that 0,1: in-focus, 2: out of focus cluster, 3: out of focus background
assumes that the two in-focus groups can have two different rates

'''


def fcn_compute_weight_degree_mat_effective_reducedSimple(\
                                     Cee, Cei, Cie, Cii, Cext, \
                                         Jee, Jei, Jie, Jii, Jee_ext, Jie_ext, \
                                             Jee_p, Jee_m, Jei_p, Jei_m, Jie_p, Jie_m, Jii_p, Jii_m, \
                                                 fE, fEb, fI, fIb, nClusters, n):
            
        
        
    # number not in in-focus groups
    n_out_focus_clusters = nClusters - 2*n
        

    # input to in-focus E clusters from all E
    Jee_infocusA = np.array([Jee_p, Jee_m, Jee_m, Jee_m, Jee_m, Jee_m, Jee_m])
    Cee_infocusA = np.array([fE*Cee*1, fE*Cee*(n-1), fE*Cee*1, fE*Cee*(n-1), fE*Cee*1, fE*Cee*(n_out_focus_clusters-1), fEb*Cee])
    
    Jee_infocusB = np.array([Jee_m, Jee_m, Jee_p, Jee_m, Jee_m, Jee_m, Jee_m])
    Cee_infocusB = np.array([fE*Cee*1, fE*Cee*(n-1), fE*Cee*1, fE*Cee*(n-1), fE*Cee*1, fE*Cee*(n_out_focus_clusters-1), fEb*Cee])

    # input to out focus E cluster group from all E
    Jee_outFocus = np.array([Jee_m, Jee_m, Jee_m, Jee_m, Jee_p, Jee_m, Jee_m])
    Cee_outFocus = np.array([fE*Cee*1, fE*Cee*(n-1), fE*Cee*1, fE*Cee*(n-1), fE*Cee*1, fE*Cee*(n_out_focus_clusters-1), fEb*Cee])
    
    # input to out focus E background from all E
    Jee_back = np.array([Jee_m, Jee_m, Jee_m, Jee_m, Jee_m, Jee_m, Jee])
    Cee_back = np.array([fE*Cee*1, fE*Cee*(n-1), fE*Cee*1, fE*Cee*(n-1), fE*Cee*1, fE*Cee*(n_out_focus_clusters-1), fEb*Cee])
    
    
    
    # input to in-focus E clusters from all I
    Jei_infocusA = np.array([Jei_p, Jei_m, Jei_m, Jei_m, Jei_m, Jei_m, Jei_m])*(-1)
    Cei_infocusA = np.array([fI*Cei*1, fI*Cei*(n-1), fI*Cei*1, fI*Cei*(n-1), fI*Cei*1, fI*Cei*(n_out_focus_clusters-1), fIb*Cei])
    
    Jei_infocusB = np.array([Jei_m, Jei_m, Jei_p, Jei_m, Jei_m, Jei_m, Jei_m])*(-1)
    Cei_infocusB = np.array([fI*Cei*1, fI*Cei*(n-1), fI*Cei*1, fI*Cei*(n-1), fI*Cei*1, fI*Cei*(n_out_focus_clusters-1), fIb*Cei])

    # input to E out focus from all I
    Jei_outFocus = np.array([Jei_m, Jei_m, Jei_m, Jei_m, Jei_p, Jei_m, Jei_m])*(-1)
    Cei_outFocus = np.array([fI*Cei*1, fI*Cei*(n-1), fI*Cei*1, fI*Cei*(n-1), fI*Cei*1, fI*Cei*(n_out_focus_clusters-1), fIb*Cei])
  
    # input to E background from all I
    Jei_back = np.array([Jei_m, Jei_m, Jei_m, Jei_m, Jei_m, Jei_m, Jei])*(-1)
    Cei_back = np.array([fI*Cei*1, fI*Cei*(n-1), fI*Cei*1, fI*Cei*(n-1), fI*Cei*1, fI*Cei*(n_out_focus_clusters-1), fIb*Cei])
    
    
    
    # input to in-focus I clusters from all E
    Jie_infocusA = np.array([Jie_p, Jie_m, Jie_m, Jie_m, Jie_m, Jie_m, Jie_m])
    Cie_infocusA = np.array([fE*Cie*1, fE*Cie*(n-1), fE*Cie*1, fE*Cie*(n-1), fE*Cie*1, fE*Cie*(n_out_focus_clusters-1), fEb*Cie])
    
    Jie_infocusB = np.array([Jie_m, Jie_m, Jie_p, Jie_m, Jie_m, Jie_m, Jie_m])
    Cie_infocusB = np.array([fE*Cie*1, fE*Cie*(n-1), fE*Cie*1, fE*Cie*(n-1), fE*Cie*1, fE*Cie*(n_out_focus_clusters-1), fEb*Cie])

    # input to I active from all E
    Jie_outFocus = np.array([Jie_m, Jie_m, Jie_m, Jie_m, Jie_p, Jie_m, Jie_m])
    Cie_outFocus = np.array([fE*Cie*1, fE*Cie*(n-1), fE*Cie*1, fE*Cie*(n-1), fE*Cie*1, fE*Cie*(n_out_focus_clusters-1), fEb*Cie])
    
    # input to E background from all E
    Jie_back = np.array([Jie_m, Jie_m, Jie_m, Jie_m, Jie_m, Jie_m, Jie])
    Cie_back = np.array([fE*Cie*1, fE*Cie*(n-1), fE*Cie*1, fE*Cie*(n-1), fE*Cie*1, fE*Cie*(n_out_focus_clusters-1), fEb*Cie])



    # input to in-focus I clusters from all I
    Jii_infocusA = np.array([Jii_p, Jii_m, Jii_m, Jii_m, Jii_m, Jii_m, Jii_m])*(-1)
    Cii_infocusA = np.array([fI*Cii*1, fI*Cii*(n-1), fI*Cii*1, fI*Cii*(n-1), fI*Cii*1, fI*Cii*(n_out_focus_clusters-1), fIb*Cii])
    
    Jii_infocusB = np.array([Jii_m, Jii_m, Jii_p, Jii_m, Jii_m, Jii_m, Jii_m])*(-1)
    Cii_infocusB = np.array([fI*Cii*1, fI*Cii*(n-1), fI*Cii*1, fI*Cii*(n-1), fI*Cii*1, fI*Cii*(n_out_focus_clusters-1), fIb*Cii])

    # input to I out of focus clusters from all E
    Jii_outFocus = np.array([Jii_m, Jii_m, Jii_m, Jii_m, Jii_p, Jii_m, Jii_m])*(-1)
    Cii_outFocus = np.array([fI*Cii*1, fI*Cii*(n-1), fI*Cii*1, fI*Cii*(n-1), fI*Cii*1, fI*Cii*(n_out_focus_clusters-1), fIb*Cii])
    
    
    # input to I background from all E
    Jii_back = np.array([Jii_m, Jii_m, Jii_m, Jii_m, Jii_m, Jii_m, Jii])*(-1)
    Cii_back = np.array([fI*Cii*1, fI*Cii*(n-1), fI*Cii*1, fI*Cii*(n-1), fI*Cii*1, fI*Cii*(n_out_focus_clusters-1), fIb*Cii])
    
    
    # put it all together
    Je_infocusA = np.hstack((Jee_infocusA, Jei_infocusA))
    Je_infocusB = np.hstack((Jee_infocusB, Jei_infocusB))
    Je_outFocus = np.hstack((Jee_outFocus, Jei_outFocus))
    Je_back = np.hstack((Jee_back, Jei_back))
    
    Ce_infocusA = np.hstack((Cee_infocusA, Cei_infocusA))
    Ce_infocusB = np.hstack((Cee_infocusB, Cei_infocusB))
    Ce_outFocus = np.hstack((Cee_outFocus, Cei_outFocus))
    Ce_back = np.hstack((Cee_back, Cei_back))

    Ji_infocusA = np.hstack((Jie_infocusA, Jii_infocusA))
    Ji_infocusB = np.hstack((Jie_infocusB, Jii_infocusB))
    Ji_outFocus = np.hstack((Jie_outFocus, Jii_outFocus))
    Ji_back = np.hstack((Jie_back, Jii_back))
    
    Ci_infocusA = np.hstack((Cie_infocusA, Cii_infocusA))
    Ci_infocusB = np.hstack((Cie_infocusB, Cii_infocusB))
    Ci_outFocus= np.hstack((Cie_outFocus, Cii_outFocus))
    Ci_back = np.hstack((Cie_back, Cii_back))
    
    Jab = np.vstack((Je_infocusA, Je_infocusB, Je_outFocus, Je_back, \
                     Ji_infocusA, Ji_infocusB, Ji_outFocus, Ji_back))
        
    Cab = np.vstack((Ce_infocusA, Ce_infocusB, Ce_outFocus, Ce_back, \
                     Ci_infocusA, Ci_infocusB, Ci_outFocus, Ci_back))
    
    Cab_ext = np.ones(8) * Cext
    Jab_ext = np.append(np.ones(4)*Jee_ext, np.ones(4)*Jie_ext)


    return Jab, Cab, Jab_ext, Cab_ext



#%% FUNCTIONS FOR LIF NETWORKS WITH NO QUENCHED VARIABILITY



#%%

'''

compute augmented versions of nu vector for reduced 3 population system

inputs:
    
    nu:     (6,) vector of population rates; 
            nu[:3] -- excitatory (active, inactive, background)
            nu[3:] -- inhibitory (active, inactive, background)
            
outputs:
    
    nu_extended: (10,) vector of population rates
                 nu[:5] -- excitatory (active, inactive, active, inactive, background)
                 nu[5:] -- inhibitory (active, inactive, active, inactive, background)
                 

'''


def fcn_compute_augmented_nuVec(nu):

    # extended rate vector for computing mean and variance of inputs
    nu_extended = np.zeros(10)
    nu_extended[0:2] = nu[0:2]
    nu_extended[2:4] = nu[0:2]
    nu_extended[4] = nu[2]
    nu_extended[5:7] = nu[3:5]
    nu_extended[7:9] = nu[3:5]
    nu_extended[9] = nu[5]    
     
    return nu_extended


#%%

'''

compute augmented versions of mean and variance of nu vectors for reduced 3
population system

inputs:
    
    nu_bar:     (6,) vector of population rate average; 
                nu_bar[:3] -- excitatory (active, inactive, background)
                nu_bar[3:] -- inhibitory (active, inactive, background)

    popVar_nu:     (6,) vector of population rate variance; 
                    popVar_nu[:3] -- excitatory (active, inactive, background)
                    popVar_nu[3:] -- inhibitory (active, inactive, background)
            
            
outputs:
    
    nu_bar_extended: (10,) vector of population rate average
                     nu_bar_extended[:5] -- excitatory (active, inactive, active, inactive, background)
                     nu_bar_extended[5:] -- inhibitory (active, inactive, active, inactive, background)
                    
                    
    popVar_nu_extended: (10,) vector of population rate variance
                         popVar_nu_extended[:5] -- excitatory (active, inactive, active, inactive, background)
                         popVar_nu_extended[5:] -- inhibitory (active, inactive, active, inactive, background)
                 

'''
def fcn_compute_augmented_stateVec_quenchedVar(nu_bar, popVar_nu):

    # extended rate, var rate vectors for computing mean and variance of inputs
    nu_bar_extended = np.zeros(10)
    nu_bar_extended[0:2] = nu_bar[0:2]
    nu_bar_extended[2:4] = nu_bar[0:2]
    nu_bar_extended[4] = nu_bar[2]
    nu_bar_extended[5:7] = nu_bar[3:5]
    nu_bar_extended[7:9] = nu_bar[3:5]
    nu_bar_extended[9] = nu_bar[5]    
    
    popVar_nu_extended = np.zeros(10)
    popVar_nu_extended[0:2] = popVar_nu[0:2]
    popVar_nu_extended[2:4] = popVar_nu[0:2]
    popVar_nu_extended[4] = popVar_nu[2]
    popVar_nu_extended[5:7] = popVar_nu[3:5]
    popVar_nu_extended[7:9] = popVar_nu[3:5]
    popVar_nu_extended[9] = popVar_nu[5]  
    
    return nu_bar_extended, popVar_nu_extended


#%% 


'''
from reduced state vector consisting of 1 active E & I, 1 inactive E and I,
and 1 background E and I, extend to full state vector of (nClusters + 1) x 2
populations


inputs:
    input vector:       (6,1) vector
                        [:3] excitatory population
                        [:3] inhibitory population
    nClusters:          scalar denoting # of clusters
    nActiveClusters:    scalar denoting # of active clusters
    
outputs:
    
    output_vector: ( (nClusters + 1)*2, )
                    [:nClusters+1] excitatory
                    [nClusters+1:] inhibitory


'''

def fcn_compute_full_vector_from_reduced(input_vector, nClusters, nActiveClusters):
    
    nPops = (nClusters+1)*2
    nPopsE = (nClusters+1)
    
    # output vector
    output_vector = np.zeros(nPops)
    
    # excitatory
    output_vector[:nActiveClusters] = input_vector[0]
    output_vector[nActiveClusters:nPopsE-1] = input_vector[1]
    output_vector[nPopsE-1] = input_vector[2]
    
    # inhibitory
    output_vector[nPopsE:nPopsE+nActiveClusters] = input_vector[3]
    output_vector[nPopsE+nActiveClusters:-1] = input_vector[4]
    output_vector[-1] = input_vector[5]
    
    # return 
    return output_vector




#%%

'''

computed augmented rate vector for effective MFT

nu :        [inFocusA, inFocusB, outFocusActive, outFocusInactive, outFocusBg]
    
nActive :   number of active clusters
'''

def fcn_compute_augmented_effective_nuVec(nu, nActive):

    if nActive > 1:    

        # extended rate vector for computing mean and variance of inputs
        nu_extended = np.zeros(14)
        
        nu_extended[0:2] = nu[0:2]
        nu_extended[2:4] = nu[2:4]
        nu_extended[4:6] = nu[2:4]
        nu_extended[6] = nu[4]
        
        nu_extended[7:9] = nu[5:7]
        nu_extended[9:11] = nu[7:9]
        nu_extended[11:13] = nu[7:9]
        nu_extended[13] = nu[9]  
        
    else:
        
        # extended rate vector for computing mean and variance of inputs
        nu_extended = np.zeros(10)
        
        nu_extended[0:2] = nu[0:2]
        nu_extended[2:4] = nu[2]
        nu_extended[4] = nu[3]
        
        nu_extended[5:7] = nu[4:6]
        nu_extended[7:9] = nu[6]
        nu_extended[9] = nu[7]
     
    return nu_extended



#%%

'''

computed augmented rate vector for effective MFT with the simplest assumption
of one out-of-focus grouped population

nu :                [inFocusA_same, inFocusA_different, inFocusB_same, inFocusB_different outFocusSame outFocusDifferent, outFocusBg]
    
n_inFocus :         number of clusters in in-focus populations


'''

def fcn_compute_augmented_effectiveSimple_nuVec(nu, n_inFocus):


    # extended rate vector for computing mean and variance of inputs
    nu_extended = np.zeros(14)
    
    nu_extended[0:2] = nu[0]
    nu_extended[2:4] = nu[1]
    nu_extended[4:6] = nu[2]
    nu_extended[6] = nu[3]
    
    nu_extended[7:9] = nu[4]
    nu_extended[9:11] = nu[5]
    nu_extended[11:13] = nu[6]
    nu_extended[13] = nu[7]  

     
    return nu_extended




#%% STABILITY MATRIX CALCULATION ALTERNATE

def fcn_stability_matrix_effective_alternate(nu_fixed_point, Mu_vec, Sigma_vec, \
                                                 tau_m, tau_s, Vr, Vth, \
                                                     Jab, Cab, \
                                                         tau_dynamics):
    
        
    
    # total number of dynamical populations
    n_dynPops = np.size(nu_fixed_point)
   


    # COMPUTE STABILITY MATRIX ELEMENTS
    
    dphi_m_dmu_m = np.zeros((n_dynPops))
    dphi_m_dsig_m = np.zeros((n_dynPops))
    
    dmu_m_dnu_n = np.zeros((n_dynPops, n_dynPops))    
    dsig_m_dnu_n = np.zeros((n_dynPops, n_dynPops))

    dphi_m_dnu_n = np.zeros((n_dynPops, n_dynPops))
    
    delta_m_n = np.zeros((n_dynPops, n_dynPops))
    np.fill_diagonal(delta_m_n,1)
    
    S = np.zeros((n_dynPops, n_dynPops))
    
    
    # LOOP OVER ALL POPULATIONS
    for m in range(0, n_dynPops):

        
        phi = nu_fixed_point[m]
            
        BS = fcn_MFT_general_tools.fcn_BrunelSergi_correction(tau_m[m], tau_s[m])
                    
        bm = (Vth[m] - Mu_vec[m])/Sigma_vec[m] + BS
        am = (Vr[m] - Mu_vec[m])/Sigma_vec[m] + BS
                
        gm_b = fcn_MFT_general_tools.fcn_TF_integrand(bm)
        gm_a = fcn_MFT_general_tools.fcn_TF_integrand(am)
                
        dbm_dsigm = -(Vth[m] - Mu_vec[m])/(Sigma_vec[m]**2)
        dam_dsigm = -(Vr[m] - Mu_vec[m])/(Sigma_vec[m]**2)
        
        dbm_dmum = -1/Sigma_vec[m]
        dam_dmum = -1/Sigma_vec[m]
        
        dphi_m_dmu_m[m] =  -(phi**2)*tau_m[m]*( gm_b*dbm_dmum - gm_a*dam_dmum )
        dphi_m_dsig_m[m] = -(phi**2)*tau_m[m]*( gm_b*dbm_dsigm - gm_a*dam_dsigm )

        
        # take derivatives with respect to all others       
        for n in range(0, n_dynPops):
                        
                
            dmu_m_dnu_n[m,n] = tau_m[m]*Jab[m,n]*Cab[m,n]             
            dsig_m_dnu_n[m,n] = (1 / (2*Sigma_vec[m])) * tau_m[m]*Jab[m,n]*Jab[m,n]*Cab[m,n] 
            
            dphi_m_dnu_n[m,n] = dphi_m_dmu_m[m]*dmu_m_dnu_n[m,n] + dphi_m_dsig_m[m]*dsig_m_dnu_n[m,n]
            

            S[m,n] = (1/tau_dynamics[m])*( dphi_m_dnu_n[m,n] - delta_m_n[m,n] ) 
                
            
    
    # compute eigenvalues
    eigenvals_S = np.linalg.eigvals(S)
    realPart_eigvals_S = np.real(eigenvals_S)
            
    return S, eigenvals_S, realPart_eigvals_S    