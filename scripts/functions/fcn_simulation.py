import numpy as np
import sys
import fcn_stimulation





#%%
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# RUN A SIMULATION OF LIF NETWORK MODEL WITH DELTA SYNAPSES
# DOES NUMERICAL INTEGRATION EXACTLY SINCE EQS ARE LINEAR
# (SEE GOODMAN AND BRETTE 2008)
# THIS IMPLEMENTATION GIVES RESULTS CONSISTENT WITH NEST
# 
# equations from Brunel 2000
# taum*dVi/dt = -Vi + Iir + Iie
# Iie = taum*sum(Jij)*nu_ext = taum*J*p*N*nu
# Iir = taum*sum(Jij*sj(t-D))
# Voltage jumps by an amount sum(Jij*sj(t-D))
# multiplication of currents by taum done here, do not input that way
#
# INPUTS
#   sim_params   class that contains all model parameters
#   J            network connectivity (N x N np array)
#   iV           initial conditions for voltage
#                can be input by the user, but if not, defaults to random
#                number between reset and threshold
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def fcn_simulate_exact_deltaSyn(sim_params, J, rand_seed=-1):

#------------------------------------------------------------------------------
# GET PARAMETERS
#------------------------------------------------------------------------------

    save_voltage = sim_params.save_voltage  # whether or not to save voltage array

    T0 = sim_params.T0                  # simulation start time
    TF = sim_params.TF                  # simulation end time
    dt = sim_params.dt                  # time step

    N = sim_params.N                    # total number of neurons
    Ne = sim_params.N_e                 # number excitatory
    Ni = sim_params.N_i                 # number inhibitory
    
    Vth_e = sim_params.Vth_e            # threshold potential E
    Vth_i = sim_params.Vth_i            # threshold potential I
    
    Vr_e = sim_params.Vr_e              # reset potential E
    Vr_i = sim_params.Vr_i              # reset potential I
    
    tau_r = sim_params.tau_r            # refractory period
    tau_m_e = sim_params.tau_m_e        # membrane time constant E
    tau_m_i = sim_params.tau_m_i        # membrane time constant I
    t_delay = sim_params.t_delay        # synaptic delay
    
    pext = sim_params.pext              # external connection probability
    Jee_ext = sim_params.Jee_ext        # external E to E weight
    Jie_ext = sim_params.Jie_ext        # external E to I weight
    nu_ext_e = sim_params.nu_ext_e          # avg baseline afferent rate to E & I neurons [spk/s]
    nu_ext_i = sim_params.nu_ext_i
       
                    
#------------------------------------------------------------------------------
# SETUP
#------------------------------------------------------------------------------
     
    # THRESHOLD VOLTAGE VECTORS   
    Vth_e_vec = Vth_e*np.ones(Ne)
    Vth_i_vec = Vth_i*np.ones(Ni)
    Vth = np.concatenate((Vth_e_vec, Vth_i_vec))
    # RESET VOLTAGE VECTOR
    Vr_e_vec = Vr_e*np.ones(Ne)
    Vr_i_vec = Vr_i*np.ones(Ni)
    Vr = np.concatenate((Vr_e_vec, Vr_i_vec))
    # MEMBRANE TIME CONSTANT VECTOR
    tau_m_e_vec = tau_m_e*np.ones(Ne)
    tau_m_i_vec = tau_m_i*np.ones(Ni)
    tau_m = np.concatenate((tau_m_e_vec, tau_m_i_vec))
    # EXTERNAL CURRENTS X TAUM
    I_ext_e_vec = (Ne*pext*nu_ext_e*Jee_ext*tau_m_e)*(np.ones((Ne)))
    I_ext_i_vec = (Ne*pext*nu_ext_i*Jie_ext*tau_m_i)*(np.ones((Ni)))
    I_ext = np.concatenate((I_ext_e_vec, I_ext_i_vec))

        
#------------------------------------------------------------------------------
# INITIAL CONDITIONS
#------------------------------------------------------------------------------   

    # if seed not input to function, initial voltage is randomly distributed between reset and threshold
    if rand_seed == -1:
        iV = Vr + (Vth-Vr)*np.random.uniform(size=(N))  
    #otherwise use the seed
    else:
        rng = np.random.default_rng(rand_seed)
        iV = Vr + (Vth-Vr)*rng.uniform(size=(N)) 
    # initial recurrent input is zero
    iIrec = np.zeros(N)
    # time each neuron has left to be refractory
    time_ref = np.zeros(N)

#------------------------------------------------------------------------------
# SIMULATION STUFF
#------------------------------------------------------------------------------  

    timePts = np.arange(T0,TF+dt,dt)       # time points of simulation
    nSteps = np.size(timePts)-1            # number of time steps is one less than number of time points in simulation
    spikes = np.zeros((2,1))*np.nan        # initialize array for spike times
    
    if save_voltage == 1:
        pltNode = 0
        v = np.zeros((N,nSteps+1))             # initialize voltage array
        Irec = np.zeros((N,nSteps+1))          # initialize recurrent input array
        Irec_e = np.zeros((nSteps+1))
        Irec_i = np.zeros((nSteps+1))
        Iext_e = np.zeros((nSteps+1))
        # set initial values
        v[:,0] = iV
        Irec[:,0] = iIrec 
        Ejump=0
        Ijump=0
        Irec_e[0] = 0
        Irec_i[0] = 0
        Iext_e[0] = iV[pltNode]
        
    else:
        Irec = iIrec
        v = iV

    

#------------------------------------------------------------------------------
# STIMULATION SETUP [THIS COULD GO INTO A FUNCTION WITH SIM_PARAMS AS INPUT]
#------------------------------------------------------------------------------

    # stimulation parameters
    stim_shape = sim_params.stim_shape
    stim_onset = sim_params.stim_onset
    stim_Ecells = sim_params.stim_Ecells
    stim_Icells = sim_params.stim_Icells
    
    # stimulated cells
    stim_cells = np.concatenate((stim_Ecells, stim_Icells))
    
    # initialize
    Istim = np.zeros((N, nSteps+1))

    if ( (stim_shape == 'box') ):
        
        stim_duration = sim_params.stim_duration
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext*Jee_ext*tau_m_e)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext*Jie_ext*tau_m_i)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        
        for tInd in range(0,nSteps+1,1):
        
            t = timePts[tInd]
            Istim_at_t = fcn_stimulation.fcn_box_stimulus(stim_onset, stim_duration, stim_amplitude, t)
            Istim[:,tInd] = Istim_at_t*stim_cells
            
    elif ( (stim_shape == 'linear') ):
        
        stim_duration = sim_params.stim_duration
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext*Jee_ext*tau_m_e)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext*Jie_ext*tau_m_i)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        
        for tInd in range(0,nSteps+1,1):
        
            t = timePts[tInd]
            Istim_at_t = fcn_stimulation.fcn_linear_stimulus(stim_onset, stim_duration, stim_amplitude, t)
            Istim[:,tInd] = Istim_at_t*stim_cells
        
    elif stim_shape == 'diff2exp':
        
        taur = sim_params.stim_taur
        taud = sim_params.stim_taud
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext*Jee_ext*tau_m_e)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext*Jie_ext*tau_m_i)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        
        for tInd in range(0,nSteps+1,1):
        
            t = timePts[tInd]
            Istim_at_t = fcn_stimulation.fcn_diff2exp_stimulus(stim_onset, stim_amplitude, taur, taud, t)
            Istim[:,tInd] = Istim_at_t*stim_cells
        
    else:
        sys.exit('unspecified stimulus type')


#------------------------------------------------------------------------------
# PRINT
#------------------------------------------------------------------------------ 
    print('I_ext_e = %0.3f' % I_ext_e_vec[0])
    print('I_ext_i = %0.3f'% I_ext_i_vec[0])
    print('Vr_e = %0.3f' % Vr_e_vec[0])
    print('Vr_i = %0.3f' % Vr_i_vec[0])
    print('Vth_e = %0.3f' % Vth_e_vec[0])
    print('Vth_i = %0.3f' % Vth_i_vec[0])
    print('tau_m_e = %0.5f s' % tau_m_e_vec[0])
    print('tau_m_i = %0.5f s' % tau_m_i_vec[0])
    print('tau_r = %0.5f s' % tau_r)
    print('t_delay = %0.5f s' % t_delay)

#------------------------------------------------------------------------------
# INTEGRATE
#------------------------------------------------------------------------------     
    
    # initial sensitive IDS
    sensitive_id = np.where(time_ref <= 0)[0]
    non_sensitive_id = []
    
    # time loop
    for tInd in range(0,nSteps,1):
                      
        if save_voltage == 1:
            
            # update currents
            Irec_e[tInd+1]  = np.exp(-dt/tau_m[pltNode])*Irec_e[tInd] + Ejump
            Irec_i[tInd+1]  = np.exp(-dt/tau_m[pltNode])*Irec_i[tInd] + Ijump
            Iext_e[tInd+1]  = I_ext[pltNode] + (Iext_e[tInd] - I_ext[pltNode])*np.exp(-dt/tau_m[pltNode])

            
            # total external is background + stimulation
            I_ext_total = I_ext + Istim[:,tInd]
            
            # update voltage of non-refractory neurons (exponential decay + spikes)
            # exact integration between t and t+dt
            # you can check that this is correct by examining the taylor expansion of 
            # exp(-dt/tau_m) to first order and recover the Euler Method :)
            if ( np.size(sensitive_id) != 0) :
                v[sensitive_id,tInd + 1] = (np.exp(-dt/tau_m[sensitive_id])* \
                                            ( v[sensitive_id,tInd] - \
                                              I_ext_total[sensitive_id] ) + \
                                              I_ext_total[sensitive_id])  +  \
                                              Irec[sensitive_id,tInd]
                                               
            # update any non-sensitive ones
            if ( np.size(non_sensitive_id) != 0) :
                v[non_sensitive_id,tInd+1] = Vr[non_sensitive_id]
                  
            # find neurons who fired at the current time index
            fired_ind = np.where(v[:,tInd+1]>=Vth)[0]
            
        else:
            
            # total external is background + stimulation
            I_ext_total = I_ext + Istim[:,tInd]
        
            # update voltage of non-refractory neurons (exponential decay + spikes)
            # exact integration between t and t+dt
            if ( np.size(sensitive_id) != 0) :
                v[sensitive_id] = (np.exp(-dt/tau_m[sensitive_id])* \
                                       ( v[sensitive_id] - I_ext_total[sensitive_id] ) + \
                                        I_ext_total[sensitive_id]) +  Irec[sensitive_id] 
 
            # update any non-sensitive ones
            if ( np.size(non_sensitive_id) != 0) :
                v[non_sensitive_id] = Vr[non_sensitive_id]
                  
            # find neurons who fired at the current time index
            fired_ind = np.where(v>=Vth)[0]
        
            
        # store spikes (row 1 = times, row 2 = indices)
        new_spikes = np.vstack((timePts[tInd+1]*np.ones(np.size(fired_ind)), fired_ind))
        spikes = np.append(spikes, new_spikes, 1)
        
        # get delayed spikes
        # these determine curent at tInd+1
        # NOTE: tInd + 2 here gives results consistent with NEST
        # means: a spike that ocurred in (tInd+1-delay,tInd+2-delay] will affect 
        #        voltage update for v[tInd+1] --> v[tInd+2]
        tDelay = timePts[tInd + 2 - round(t_delay/dt)]
        if tInd + 2 - round(t_delay/dt) < 0:
            spikes_for_input = np.zeros(N)
        else:
            spikes_for_input = np.zeros(N)
            fired_delay_ind = np.where(spikes[0,:]==tDelay)[0]
            fired_delay = spikes[1,fired_delay_ind]
            spikes_for_input[fired_delay.astype(int)] = 1
             
        # spike propagation
        if save_voltage == 1:
            
            # total recurrent
            Irec[:,tInd+1] = np.matmul(J,(spikes_for_input))
            
            # jumps in membrane potential of 1 node due to E and I spikes
            Ejump = np.matmul(J[pltNode,:Ne],(spikes_for_input[:Ne]*1))
            Ijump = np.matmul(J[pltNode,Ne:],(spikes_for_input[Ne:]*1))
            
            # reset neurons who fired
            v[fired_ind,tInd+1] = Vr[fired_ind]
            
            
        else:
            # total recurrent
            Irec = np.matmul(J,(spikes_for_input))
            
            # reset neurons who fired
            v[fired_ind] = Vr[fired_ind]
        
        # reset time remaining refractory
        time_ref[fired_ind] = tau_r
        # update time remaining refractory at each step
        time_ref = time_ref - dt
        
        # update sensitive ids
        sensitive_id = np.where(time_ref <= 0)[0]
        non_sensitive_id = np.where(time_ref > 0)[0]
                
        # update status
        if ((tInd % 5000) == 0):
            print('%0.3f pct complete' % (100*tInd/nSteps) )
                                  
    # delete first column used to initialize spikes    
    spikes = np.delete(spikes, 0, 1)
    
    # return
    if save_voltage == 1:
        return timePts, spikes, v, Irec_e, Irec_i, Iext_e, I_ext, Istim
    else:
        return spikes




#%%

'''
assumes that baseline external inputs are either
    homogeneous poisson spikes, filtered through an exponential synapse
    constant, goes directly to voltage equation
    
assumes that external stimulus enters voltage equation, not poisson

this function yields results that are exactly consistent with NEST when the 
external inputs are constant (i.e., deterministic case)

equations take the form of Brunel & Sergi 1998


'''



def fcn_simulate_exact_poisson(sim_params, J, rand_seed='random'):

#------------------------------------------------------------------------------
# GET PARAMETERS
#------------------------------------------------------------------------------

    save_voltage = sim_params.save_voltage # whether or not to save voltage array


    T0 = sim_params.T0                  # simulation start time
    TF = sim_params.TF                  # simulation end time
    dt = sim_params.dt                  # time step

    N = sim_params.N                    # total number of neurons
    Ne = sim_params.N_e                 # number excitatory
    Ni = sim_params.N_i                 # number inhibitory
    
    Vth_e = sim_params.Vth_e            # threshold potential E
    Vth_i = sim_params.Vth_i            # threshold potential I
    
    Vr_e = sim_params.Vr_e              # reset potential E
    Vr_i = sim_params.Vr_i              # reset potential E
    
    tau_r = sim_params.tau_r            # refractory period
    tau_m_e = sim_params.tau_m_e        # membrane time constant E
    tau_m_i = sim_params.tau_m_i        # membrane time constant I
    tau_s_e = sim_params.tau_s_e        # synaptic time constant E
    tau_s_i = sim_params.tau_s_i        # synaptic time constant I
    
    t_delay = sim_params.t_delay        # synaptic delay

    
    pext = sim_params.pext              # external connection probability
    Jee_ext = sim_params.Jee_ext        # external E to E weight
    Jie_ext = sim_params.Jie_ext        # external E to I weight
    nu_ext_e = sim_params.nu_ext_e      # avg baseline afferent rate to E & I neurons [spk/s]
    nu_ext_i = sim_params.nu_ext_i      
       
    extCurrent_poisson = sim_params.extCurrent_poisson # whether or not external current is Poisson
    
#------------------------------------------------------------------------------
# SETUP
#------------------------------------------------------------------------------
     
    # THRESHOLD VOLTAGE VECTORS   
    Vth_e_vec = Vth_e*np.ones(Ne)
    Vth_i_vec = Vth_i*np.ones(Ni)
    Vth = np.concatenate((Vth_e_vec, Vth_i_vec))
    # RESET VOLTAGE VECTOR
    Vr_e_vec = Vr_e*np.ones(Ne)
    Vr_i_vec = Vr_i*np.ones(Ni)
    Vr = np.concatenate((Vr_e_vec, Vr_i_vec))
    # MEMBRANE TIME CONSTANT VECTOR
    tau_m_e_vec = tau_m_e*np.ones(Ne)
    tau_m_i_vec = tau_m_i*np.ones(Ni)
    tau_m = np.concatenate((tau_m_e_vec, tau_m_i_vec))    
    # SYNAPTIC WEIGHTS  
    Jij = J
    
    
        
#------------------------------------------------------------------------------
# INITIAL CONDITIONS
#------------------------------------------------------------------------------ 

    # if seed not input to function, initial voltage is randomly distributed between reset and threshold
    if rand_seed == 'random':
        rng = np.random.default_rng(np.random.choice(100000))
        iV = Vr + (Vth-Vr)*np.random.uniform(size=(N))  
    #otherwise use the seed
    else:
        rng = np.random.default_rng(rand_seed)
        iV = Vr + (Vth-Vr)*rng.uniform(size=(N)) 

    # time each neuron has left to be refractory
    time_ref = np.zeros(N)

#------------------------------------------------------------------------------
# SIMULATION STUFF
#------------------------------------------------------------------------------  

    timePts = np.arange(T0,TF+dt,dt)  # time points of simulation
    nSteps = np.size(timePts)-1       # number of time steps is one less than number of time points in simulation
    spikes = np.zeros((2,1))*np.nan   # initialize array for spike times
        
#------------------------------------------------------------------------------
# SET UP PROPAGATOR FOR SUBTHRESHOLD DYNAMICS
#------------------------------------------------------------------------------ 

    propagator = np.zeros((N, 4, 4))
    
    propagator[:,0,0] = np.exp(-dt/tau_m)
    propagator[:,0,1] = (tau_s_e*tau_m)/(tau_s_e - tau_m) * ( np.exp(-dt/tau_s_e) - np.exp(-dt/tau_m) )
    propagator[:,0,2] = (tau_s_i*tau_m)/(tau_s_i - tau_m) * ( np.exp(-dt/tau_s_i) - np.exp(-dt/tau_m) )
    propagator[:,0,3] = tau_m*(1 - np.exp(-dt/tau_m))
    
    propagator[:,1,0] = 0
    propagator[:,1,1] = np.exp(-dt/tau_s_e)
    propagator[:,1,2] = 0
    propagator[:,1,3] = 0
    
    propagator[:,2,0] = 0
    propagator[:,2,1] = 0
    propagator[:,2,2] = np.exp(-dt/tau_s_i)
    propagator[:,2,3] = 0
    
    propagator[:,3,0] = 0
    propagator[:,3,1] = 0
    propagator[:,3,2] = 0
    propagator[:,3,3] = 1
    
        
#------------------------------------------------------------------------------
# EXTERNAL CURRENTS [POISSON WITH RATE C_ext*nu_ext]
#------------------------------------------------------------------------------

    if extCurrent_poisson == True:
        
        extPoisson_e_vec = np.zeros((Ne, nSteps+1))
        extPoisson_i_vec = np.zeros((Ni, nSteps+1))
        for i in range(0, Ne):
           extPoisson_e_vec[i,:] = rng.poisson(nu_ext_e[i]*Ne*pext*dt, nSteps+1)
        for i in range(0, Ni):
           extPoisson_i_vec[i,:] = rng.poisson(nu_ext_i[i]*Ne*pext*dt, nSteps+1)
        
        # poisson spike trains for all cells
        extPoisson_vec = np.vstack((extPoisson_e_vec, extPoisson_i_vec))
        
    else:
        
        # no external poisson
        extPoisson_vec = np.zeros((N, nSteps+1))
    
    
               
#------------------------------------------------------------------------------
# STIMULATION SETUP [THIS COULD GO INTO A FUNCTION WITH SIM_PARAMS AS INPUT]
#------------------------------------------------------------------------------

    # stimulation parameters
    stim_shape = sim_params.stim_shape
    stim_onset = sim_params.stim_onset
    stim_Ecells = sim_params.stim_Ecells
    stim_Icells = sim_params.stim_Icells
    
    # stimulated cells
    stim_cells = np.concatenate((stim_Ecells, stim_Icells))
    
    # initialize
    Istim = np.zeros((N, nSteps+1))
    
    print(np.shape(Istim))

    if ( (stim_shape == 'box') ):
        
        stim_duration = sim_params.stim_duration
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext*Jee_ext)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext*Jie_ext)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        
        for tInd in range(0,nSteps+1,1):
        
            t = timePts[tInd]
            Istim_at_t = fcn_stimulation.fcn_box_stimulus(stim_onset, stim_duration, stim_amplitude, t)
            Istim[:,tInd] = Istim_at_t*stim_cells
            
    elif ( (stim_shape == 'linear') ):
        
        stim_duration = sim_params.stim_duration
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext*Jee_ext)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext*Jie_ext)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        
        for tInd in range(0,nSteps+1,1):
        
            t = timePts[tInd]
            Istim_at_t = fcn_stimulation.fcn_linear_stimulus(stim_onset, stim_duration, stim_amplitude, t)
            Istim[:,tInd] = Istim_at_t*stim_cells
        
    elif stim_shape == 'diff2exp':
        
        taur = sim_params.stim_taur
        taud = sim_params.stim_taud
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext*Jee_ext)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext*Jie_ext)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        
        for tInd in range(0,nSteps+1,1):
        
            t = timePts[tInd]
            Istim_at_t = fcn_stimulation.fcn_diff2exp_stimulus(stim_onset, stim_amplitude, taur, taud, t)
            Istim[:,tInd] = Istim_at_t*stim_cells
        
    else:
        sys.exit('unspecified stimulus type')
        
    
        
#------------------------------------------------------------------------------
# STATE VARIABLES
#------------------------------------------------------------------------------ 

    if save_voltage == True:
    
        V = np.zeros((N, nSteps+1))
        I_exc = np.zeros((N, nSteps+1))
        I_inh = np.zeros((N, nSteps+1))
        I_o = np.zeros((N, nSteps+1))
    
        # initial conditions
        V[:, 0] = iV
        I_exc[:, 0] = 0.
        I_inh[:, 0] = 0.
        
        # external current for straight to voltage equation
        if extCurrent_poisson == True:
            I_o = Istim.copy()
        else:
            I_const = np.zeros(N)
            I_const[:Ne] = nu_ext_e*Ne*pext*Jee_ext
            I_const[Ne:] = nu_ext_i*Ne*pext*Jie_ext
            for i in range(0,N):
                I_o[i, :] = Istim[i, :] + I_const[i]
                
    else:
        
        V = iV
        I_exc = np.zeros(N)
        I_inh = np.zeros(N)
        I_o = np.zeros((N, nSteps+1))

        # external current for straight to voltage equation
        if extCurrent_poisson == True:
            I_o = Istim.copy()
        else:
            I_const = np.zeros(N)
            I_const[:Ne] = nu_ext_e*Ne*pext*Jee_ext
            I_const[Ne:] = nu_ext_i*Ne*pext*Jie_ext
            for i in range(0,N):
                I_o[i, :] = Istim[i, :] + I_const[i]
                
#------------------------------------------------------------------------------
# PRINT
#------------------------------------------------------------------------------ 
    print('stim amp = %0.3f' % stim_amplitude[0])
    print('Vr_e = %0.3f' % Vr_e_vec[0])
    print('Vr_i = %0.3f' % Vr_i_vec[0])
    print('Vth_e = %0.3f' % Vth_e_vec[0])
    print('Vth_i = %0.3f' % Vth_i_vec[0])
    print('tau_m_e = %0.5f s' % tau_m_e_vec[0])
    print('tau_m_i = %0.5f s' % tau_m_i_vec[0])
    print('tau_s_e = %0.5f s' % tau_s_e)
    print('tau_s_i = %0.5f s' % tau_s_i)
    print('tau_r = %0.5f s' % tau_r)
        
#------------------------------------------------------------------------------
# INTEGRATE
#------------------------------------------------------------------------------     

    
    # time loop
    for tInd in range(0,nSteps,1):
        
                
        
        #------------------ SUBTHRESHOLD STATE UPDATE ------------------------#
        
        if save_voltage == True:
            
            #------------------ REFRACTORY CONDITION -----------------------------#
        
            # update sensitive ids
            sensitive_id = np.nonzero(np.round(time_ref,5) == 0)[0]
            non_sensitive_id = np.setdiff1d(np.arange(0,N), sensitive_id)
        
            # update time remaining refractory at each step
            time_ref[non_sensitive_id] = time_ref[non_sensitive_id] - dt
        
            # update the voltage of non-refractory
            if np.size(sensitive_id) > 0:
                
                V[sensitive_id, tInd+1] = propagator[sensitive_id, 0, 0] * V[sensitive_id, tInd] + \
                                          propagator[sensitive_id, 0, 1] * I_exc[sensitive_id, tInd] + \
                                          propagator[sensitive_id, 0, 2] * I_inh[sensitive_id, tInd] + \
                                          propagator[sensitive_id, 0, 3] * I_o[sensitive_id, tInd]
                               
                                        
            # update voltage of refractory
            if np.size(non_sensitive_id) > 0:
                
                V[non_sensitive_id, tInd+1] = Vr[non_sensitive_id]
              
        
            # update excitatory synaptic current
            I_exc[:, tInd+1] = propagator[:, 1, 1] * I_exc[:, tInd]
            
            # update inhibitory synaptic current
            I_inh[:, tInd+1] = propagator[:, 2, 2] * I_inh[:, tInd]
            
    
    
            #------------------ GET CELLS THAT FIRED AT CURRENT INDEX -------------#
            
            fired_ind = np.nonzero(V[:, tInd+1]>=Vth)[0]
            
            
            #------------------ STORE THESE NEW SPIKES ----------------------------#      
            
            # store spikes (row 1 = times, row 2 = neuron index)
            new_spikes = np.vstack((timePts[tInd + 1]*np.ones(np.size(fired_ind)), fired_ind))
            spikes = np.append(spikes, new_spikes, 1)
    
    
    
            #------------------ UPDATE SYNAPTIC WEIGHTS --> JUMP ------------------#
            
            # get delayed spikes --> update curent at tInd+1
            # note that with this definition, if t_delay = dt, then a spike at time 
            # t will affect psc at time t+1 and voltage at time t+2
            
            indDelay = tInd + 1 - round(t_delay/dt)
            tDelay = timePts[indDelay]
            if indDelay < 0:
                spikes_for_input = np.zeros(N)
            else:
                spikes_for_input = np.zeros(N)
                fired_delay_ind = np.nonzero(spikes[0,:] == tDelay)[0]
                fired_delay = spikes[1,fired_delay_ind]
                spikes_for_input[fired_delay.astype(int)] = 1
            
            
            # update synaptic current
            
            # external excitatory current 
            # [since homogeneous poisson, don't need to worry about delays]
            I_exc_ext = np.zeros(N)
            I_exc_ext[:Ne] = Jee_ext*extPoisson_vec[:Ne, tInd+1]/tau_s_e
            I_exc_ext[Ne:] = Jie_ext*extPoisson_vec[Ne:, tInd+1]/tau_s_e
            
            # recurrent excitatory
            I_exc_rec = np.matmul(Jij[:,:Ne],spikes_for_input[:Ne]*1)/tau_s_e
            
            # total excitatory
            I_exc[:, tInd + 1] = I_exc[:, tInd + 1] + I_exc_ext + I_exc_rec
            
            
            # recurrent inhibitory
            I_inh_rec = np.matmul(Jij[:,Ne:],spikes_for_input[Ne:]*1)/tau_s_i
            
            # total inhibitory
            I_inh[:, tInd + 1] = I_inh[:, tInd + 1] + I_inh_rec
    
            
            
            #------------------ RESET REFRACTORY NEURONS --------------------------#
    
            
            # reset neurons who fired
            V[fired_ind, tInd + 1] = Vr[fired_ind]        
            
            
            # set their time remaining refractory to tau_r
            time_ref[fired_ind] = tau_r
            
            
            
            
        #------------------ SUBTHRESHOLD STATE UPDATE ------------------------#

            
        else:
            
            #------------------ REFRACTORY CONDITION -----------------------------#
        
            # update sensitive ids
            sensitive_id = np.nonzero(np.round(time_ref,5) == 0)[0]
            non_sensitive_id = np.setdiff1d(np.arange(0,N), sensitive_id)
        
            # update time remaining refractory at each step
            time_ref[non_sensitive_id] = time_ref[non_sensitive_id] - dt
            
            # update the voltage of non-refractory
            if np.size(sensitive_id) > 0:
                
                V[sensitive_id] = propagator[sensitive_id, 0, 0] * V[sensitive_id] + \
                                  propagator[sensitive_id, 0, 1] * I_exc[sensitive_id] + \
                                  propagator[sensitive_id, 0, 2] * I_inh[sensitive_id] + \
                                  propagator[sensitive_id, 0, 3] * I_o[sensitive_id, tInd]
                               
                                        
            # update voltage of refractory
            if np.size(non_sensitive_id) > 0:
                
                V[non_sensitive_id] = Vr[non_sensitive_id]
              
        
            # update excitatory synaptic current
            I_exc = propagator[:, 1, 1] * I_exc
            
            # update inhibitory synaptic current
            I_inh = propagator[:, 2, 2] * I_inh

            
    
    
            #------------------ GET CELLS THAT FIRED AT CURRENT INDEX -------------#
            
            fired_ind = np.nonzero(V>=Vth)[0]
            
            
            #------------------ STORE THESE NEW SPIKES ----------------------------#      
            
            # store spikes (row 1 = times, row 2 = neuron index)
            new_spikes = np.vstack((timePts[tInd + 1]*np.ones(np.size(fired_ind)), fired_ind))
            spikes = np.append(spikes, new_spikes, 1)
    
    
    
            #------------------ UPDATE SYNAPTIC WEIGHTS --> JUMP ------------------#
            
            # get delayed spikes --> update curent at tInd+1
            # note that with this definition, if t_delay = dt, then a spike at time 
            # t will affect psc at time t+1 and voltage at time t+2
            
            indDelay = tInd + 1 - round(t_delay/dt)
            tDelay = timePts[indDelay]
            if indDelay < 0:
                spikes_for_input = np.zeros(N)
            else:
                spikes_for_input = np.zeros(N)
                fired_delay_ind = np.nonzero(spikes[0,:] == tDelay)[0]
                fired_delay = spikes[1,fired_delay_ind]
                spikes_for_input[fired_delay.astype(int)] = 1
            
            
            # update synaptic current
            
            # external excitatory current 
            # [since homogeneous poisson, don't need to worry about delays]
            I_exc_ext = np.zeros(N)
            I_exc_ext[:Ne] = Jee_ext*extPoisson_vec[:Ne, tInd+1]/tau_s_e
            I_exc_ext[Ne:] = Jie_ext*extPoisson_vec[Ne:, tInd+1]/tau_s_e
            
            # recurrent excitatory
            I_exc_rec = np.matmul(Jij[:,:Ne],spikes_for_input[:Ne]*1)/tau_s_e
            
            # total excitatory
            I_exc = I_exc + I_exc_ext + I_exc_rec
            
            
            # recurrent inhibitory
            I_inh_rec = np.matmul(Jij[:,Ne:],spikes_for_input[Ne:]*1)/tau_s_i
            
            # total inhibitory
            I_inh = I_inh + I_inh_rec
    
            
            
            #------------------ RESET REFRACTORY NEURONS --------------------------#
    
            
            # reset neurons who fired
            V[fired_ind] = Vr[fired_ind]       
            
        
            # set their time remaining refractory to tau_r
            time_ref[fired_ind] = tau_r
        
        
        
    # delete first column used to initialize spikes    
    spikes = np.delete(spikes, 0, 1)


    # output
    if save_voltage == True:

        return timePts, spikes, V, I_exc, I_inh, I_o
    
    else:
        
        return spikes





#%%
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# RUN A SIMULATION OF LIF NETWORK MODEL WITH WHITE NOISE EXTERNAL INPUTS
#
# Equations take the following form:
#
# taum dV/dt = -V + taum*Iext + taum*Irec
# taus dIrec/dt = -Irec + \sum Jij*sj(t-d)
#
# Where the external input is
#
# Iext = \mu_ext + \sigma_ext \xi(t)
# \mu_ext = Ne*pext*nu_ext_e*J_ext
# \sigma_ext = \sqrt(taum)*sd_nu_ext_white_pert*mu_ext
#
# NOTES
#
# In this function, we do not include the membrane time constant in the defintion of the currents
# Note the definition of the white noise term [includes \sqrt(taum) factor such that sd_nu_ext_white_pert*mu_ext is in units of current]
# Stochastic integration is Euler-Maruyama
# Might be clearer to define sigma_ext outside of this script and then feed it in
#
# Not sure Iext has been defined correctly for the case save_voltage = 1
# White noise process not well defined
# Also note that current can go negative [would be like having an overall inhibitory input at times]
#
#
# INPUTS
#   sim_params   class that contains all model parameters
#   J            network connectivity (N x N np array)
#   rand_seed    random seed for voltage initial conditions
#                can be input by the user, but if not, defaults to random
#                number between reset and threshold
#
#

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def fcn_simulate_whitenoise(sim_params, J, rand_seed='random', whitenoiseSeed = 'random'):

#------------------------------------------------------------------------------
# GET PARAMETERS
#------------------------------------------------------------------------------

    save_voltage = sim_params.save_voltage # whether or not to save voltage array
    

    T0 = sim_params.T0                  # simulation start time
    TF = sim_params.TF                  # simulation end time
    dt = sim_params.dt                  # time step

    N = sim_params.N                    # total number of neurons
    Ne = sim_params.N_e                 # number excitatory
    Ni = sim_params.N_i                 # number inhibitory
    
    Vth_e = sim_params.Vth_e            # threshold potential E
    Vth_i = sim_params.Vth_i            # threshold potential I
    
    Vr_e = sim_params.Vr_e              # reset potential E
    Vr_i = sim_params.Vr_i              # reset potential E
    
    tau_r = sim_params.tau_r            # refractory period
    tau_m_e = sim_params.tau_m_e        # membrane time constant E
    tau_m_i = sim_params.tau_m_i        # membrane time constant I
    tau_s_e = sim_params.tau_s_e        # synaptic time constant E
    tau_s_i = sim_params.tau_s_i        # synaptic time constant I
    
    t_delay = sim_params.t_delay        # synaptic delay

    
    pext = sim_params.pext              # external connection probability
    Jee_ext = sim_params.Jee_ext        # external E to E weight
    Jie_ext = sim_params.Jie_ext        # external E to I weight
    nu_ext_e = sim_params.nu_ext_e      # avg baseline afferent rate to E & I neurons [spk/s]
    nu_ext_i = sim_params.nu_ext_i 
    
    sd_nu_ext_e_white_pert = sim_params.sd_nu_ext_e_white_pert
    sd_nu_ext_i_white_pert = sim_params.sd_nu_ext_i_white_pert
    

#------------------------------------------------------------------------------
# SETUP
#------------------------------------------------------------------------------
     
    # THRESHOLD VOLTAGE VECTORS   
    Vth_e_vec = Vth_e*np.ones(Ne)
    Vth_i_vec = Vth_i*np.ones(Ni)
    Vth = np.concatenate((Vth_e_vec, Vth_i_vec))
    # RESET VOLTAGE VECTOR
    Vr_e_vec = Vr_e*np.ones(Ne)
    Vr_i_vec = Vr_i*np.ones(Ni)
    Vr = np.concatenate((Vr_e_vec, Vr_i_vec))
    # MEMBRANE TIME CONSTANT VECTOR
    tau_m_e_vec = tau_m_e*np.ones(Ne)
    tau_m_i_vec = tau_m_i*np.ones(Ni)
    tau_m = np.concatenate((tau_m_e_vec, tau_m_i_vec))
    # EXTERNAL CURRENTS [WITHOUT MEMBRANE TIME CONSTANT HERE]
    I_ext_e_vec = (Ne*pext*nu_ext_e*Jee_ext)*(np.ones((Ne)))
    I_ext_i_vec = (Ne*pext*nu_ext_i*Jie_ext)*(np.ones((Ni)))
    I_ext = np.concatenate((I_ext_e_vec, I_ext_i_vec))   
    # STANDARD DEVIATION OF WHITE NOISE INPUT AS FRACTION OF MEAN INPUT
    sd_nu_ext_e_white_pert_vec = sd_nu_ext_e_white_pert*np.ones((Ne))
    sd_nu_ext_i_white_pert_vec = sd_nu_ext_i_white_pert*np.ones((Ni))
    sd_nu_ext_white_pert = np.concatenate((sd_nu_ext_e_white_pert_vec,sd_nu_ext_i_white_pert_vec))
    # SYNAPTIC WEIGHTS  
    Jij = J
    
        
#------------------------------------------------------------------------------
# INITIAL CONDITIONS
#------------------------------------------------------------------------------ 

    # if seed not input to function, initial voltage is randomly distributed between reset and threshold
    if rand_seed == 'random':
        rng = np.random.default_rng(np.random.choice(10000))
        iV = Vr + (Vth-Vr)*rng.uniform(size=(N))  
    #otherwise use the seed
    else:
        rng = np.random.default_rng(rand_seed)
        iV = Vr + (Vth-Vr)*rng.uniform(size=(N)) 
    # initial recurrent input is zero
    iIrec = np.zeros(N)
    # time each neuron has left to be refractory
    time_ref = np.zeros(N)
    
#------------------------------------------------------------------------------
# NOISE
#------------------------------------------------------------------------------

    if whitenoiseSeed == 'random':
        
        rng_noise = np.random.default_rng(np.random.choice(10000))
        
    else:
        
        rng_noise = np.random.default_rng(whitenoiseSeed)


#------------------------------------------------------------------------------
# SIMULATION STUFF
#------------------------------------------------------------------------------  

    timePts = np.arange(T0,TF+dt,dt)  # time points of simulation
    nSteps = np.size(timePts)-1       # number of time steps is one less than number of time points in simulation
    spikes = np.zeros((2,1))*np.nan   # initialize array for spike times

    if save_voltage == 1:
        
        v = np.zeros((N,nSteps+1))          # initialize voltage array
        Iext_e = np.zeros((N,nSteps+1))     # initialize external input
        Irec_e = np.zeros((N,nSteps+1))     # initialize recurrent E input array 
        Irec_i = np.zeros((N,nSteps+1))     # initialize recurrent I input array
        Irec = np.zeros((N,nSteps+1))       # initialize total recurrent input array

        # set initial values
        # recurrent    
        Irec[:,0] = iIrec
        Irec_e[:,0] = iIrec
        Irec_i[:,0] = iIrec
        # voltage
        v[:,0] = iV
        
    else:
        
        # recurrent
        Irec = iIrec
        Irec_e = iIrec
        Irec_i = iIrec
        # voltage
        v = iV
               
#------------------------------------------------------------------------------
# STIMULATION SETUP [THIS COULD GO INTO A FUNCTION WITH SIM_PARAMS AS INPUT]
#------------------------------------------------------------------------------

    # stimulation parameters
    stim_shape = sim_params.stim_shape
    stim_onset = sim_params.stim_onset
    stim_Ecells = sim_params.stim_Ecells
    stim_Icells = sim_params.stim_Icells
    
    # stimulated cells
    stim_cells = np.concatenate((stim_Ecells, stim_Icells))
    
    # initialize
    Istim = np.zeros((N, nSteps+1))

    if ( (stim_shape == 'box') ):
        
        stim_duration = sim_params.stim_duration
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext*Jee_ext)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext*Jie_ext)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        
        for tInd in range(0,nSteps+1,1):
        
            t = timePts[tInd]
            Istim_at_t = fcn_stimulation.fcn_box_stimulus(stim_onset, stim_duration, stim_amplitude, t)
            Istim[:,tInd] = Istim_at_t*stim_cells
            
    elif ( (stim_shape == 'linear') ):
        
        stim_duration = sim_params.stim_duration
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext*Jee_ext)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext*Jie_ext)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        
        for tInd in range(0,nSteps+1,1):
        
            t = timePts[tInd]
            Istim_at_t = fcn_stimulation.fcn_linear_stimulus(stim_onset, stim_duration, stim_amplitude, t)
            Istim[:,tInd] = Istim_at_t*stim_cells
        
    elif stim_shape == 'diff2exp':
        
        taur = sim_params.stim_taur
        taud = sim_params.stim_taud
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext*Jee_ext)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext*Jie_ext)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        
        for tInd in range(0,nSteps+1,1):
        
            t = timePts[tInd]
            Istim_at_t = fcn_stimulation.fcn_diff2exp_stimulus(stim_onset, stim_amplitude, taur, taud, t)
            Istim[:,tInd] = Istim_at_t*stim_cells
        
    else:
        sys.exit('unspecified stimulus type')

       
#------------------------------------------------------------------------------
# PRINT
#------------------------------------------------------------------------------ 
    print('I_ext_e = %0.3f' % I_ext_e_vec[0])
    print('I_ext_i = %0.3f'% I_ext_i_vec[0])
    print('stim amp = %0.3f' % stim_amplitude[0])
    print('Vr_e = %0.3f' % Vr_e_vec[0])
    print('Vr_i = %0.3f' % Vr_i_vec[0])
    print('Vth_e = %0.3f' % Vth_e_vec[0])
    print('Vth_i = %0.3f' % Vth_i_vec[0])
    print('tau_m_e = %0.5f s' % tau_m_e_vec[0])
    print('tau_m_i = %0.5f s' % tau_m_i_vec[0])
    print('tau_s_e = %0.5f s' % tau_s_e)
    print('tau_s_i = %0.5f s' % tau_s_i)
    print('tau_r = %0.5f s' % tau_r)
        
#------------------------------------------------------------------------------
# INTEGRATE
#------------------------------------------------------------------------------     

    sensitive_id = np.where(time_ref <= 0)[0]
    non_sensitive_id = []
    
    # time loop
    for tInd in range(0,nSteps,1):
        
        if save_voltage == 1:
            
            # generate noise
            dW = rng_noise.normal(loc=0.0, scale=1., size=N)*np.sqrt(dt)
                
        
            # update voltage of non-refractory neurons              
            if ( np.size(sensitive_id) != 0) :
                                                   
                # voltage update
                v[sensitive_id,tInd + 1] = v[sensitive_id,tInd] - \
                                           dt*v[sensitive_id,tInd]/tau_m[sensitive_id] + \
                                           dt*I_ext[sensitive_id] + dt*Istim[sensitive_id, tInd] + \
                                           dt*Irec[sensitive_id,tInd] + \
                                           dW[sensitive_id]*np.sqrt(tau_m[sensitive_id])*sd_nu_ext_white_pert[sensitive_id]*I_ext[sensitive_id]

        
        
            # update voltage of refractory neurons
            if ( np.size(non_sensitive_id) != 0) :
                
                v[non_sensitive_id, tInd + 1] = Vr[non_sensitive_id]  
        
        
            # find neurons who fired at the current time index
            fired_ind = np.where(v[:,tInd+1]>=Vth)[0]
            
            # store spikes (row 1 = times, row 2 = indices)
            new_spikes = np.vstack((timePts[tInd+1]*np.ones(np.size(fired_ind)), fired_ind))
            spikes = np.append(spikes, new_spikes, 1)
            
            # get delayed spikes
            # these determine curent at tInd+1
            tDelay = timePts[tInd + 1 - round(t_delay/dt)]
            if tInd + 1 - round(t_delay/dt) < 0:
                spikes_for_input = np.zeros(N)
            else:
                spikes_for_input = np.zeros(N)
                fired_delay_ind = np.where(spikes[0,:]==tDelay)[0]
                fired_delay = spikes[1,fired_delay_ind]
                spikes_for_input[fired_delay.astype(int)] = 1
            
            # update synaptic current
            # exponential decay + spike propagation
            # exact integration between t and t+dt
                    
            # external exctitatory contribution
            Iext_e[:,tInd + 1] = (I_ext + np.sqrt(tau_m)*sd_nu_ext_white_pert*I_ext*dW/np.sqrt(dt))*tau_m
            
            # recurrent excitatory current to each neuron
            Irec_e[:,tInd + 1] = Irec_e[:,tInd]*np.exp(-dt/tau_s_e)
            Irec_e[:,tInd + 1] = Irec_e[:,tInd+1] + np.matmul(Jij[:,:Ne],spikes_for_input[:Ne]*1)/tau_s_e
            
            # recurrent inhibitory current to each neuron
            Irec_i[:,tInd + 1] = Irec_i[:,tInd]*np.exp(-dt/tau_s_i)
            Irec_i[:,tInd + 1] = Irec_i[:,tInd+1] + np.matmul(Jij[:,Ne:],spikes_for_input[Ne:]*1)/tau_s_i
                                            
            # total recurrent input
            Irec[:,tInd + 1] = Irec_e[:,tInd+1] + Irec_i[:,tInd+1] 
            
            
            # reset neurons who fired
            v[fired_ind,tInd + 1] = Vr[fired_ind]
            
        else: 
                        
            # generate noise
            dW = rng_noise.normal(loc=0.0, scale=1., size=N)*np.sqrt(dt)

                
            # update voltage of non-refractory neurons              
            if ( np.size(sensitive_id) != 0) :
            
                # voltage update
                v[sensitive_id] = v[sensitive_id] - \
                                  dt*v[sensitive_id]/tau_m[sensitive_id] + \
                                  dt*I_ext[sensitive_id] + dt*Istim[sensitive_id, tInd] + \
                                  dt*Irec[sensitive_id] + \
                                  dW[sensitive_id]*np.sqrt(tau_m[sensitive_id])*sd_nu_ext_white_pert[sensitive_id]*I_ext[sensitive_id]
                              
                                           
            # update refractory
            if ( np.size(non_sensitive_id) != 0) :
                
                v[non_sensitive_id] = Vr[non_sensitive_id] 
                
            # find neurons who fired at the current time index
            fired_ind = np.where(v>=Vth)[0]

            # store spikes (row 1 = times, row 2 = indices)
            new_spikes = np.vstack((timePts[tInd+1]*np.ones(np.size(fired_ind)), fired_ind))
            spikes = np.append(spikes, new_spikes, 1)

            # get delayed spikes
            # these determine curent at tInd+1
            tDelay = timePts[tInd + 1 - round(t_delay/dt)]
            if tInd + 1 - round(t_delay/dt) < 0:
                spikes_for_input = np.zeros(N)
            else:
                spikes_for_input = np.zeros(N)
                fired_delay_ind = np.where(spikes[0,:]==tDelay)[0]
                fired_delay = spikes[1,fired_delay_ind]
                spikes_for_input[fired_delay.astype(int)] = 1


            # update synaptic current
            # exponential decay
            # exact integration between t and t+dt
                      
            # recurrent excitatory current to each neuron
            Irec_e = Irec_e*np.exp(-dt/tau_s_e)
            Irec_e = Irec_e + np.matmul(Jij[:,:Ne],spikes_for_input[:Ne]*1)/tau_s_e
            
            # recurrent inhibitory current to each neuron
            Irec_i = Irec_i*np.exp(-dt/tau_s_i)
            Irec_i = Irec_i + np.matmul(Jij[:,Ne:],spikes_for_input[Ne:]*1)/tau_s_i
                                            
            # total recurrent input
            Irec = Irec_e + Irec_i

            # reset neurons who fired
            v[fired_ind] = Vr[fired_ind]

        
        # set their time remaining refractory to tau_r
        time_ref[fired_ind] = tau_r

        # update time remaining refractory at each step
        time_ref = time_ref - dt
        
        # update sensitive ids
        sensitive_id = np.where(time_ref <= 0)[0]
        non_sensitive_id = np.where(time_ref > 0)[0]

        
    # delete first column used to initialize spikes    
    spikes = np.delete(spikes, 0, 1)

    if save_voltage == 1:
       
        # multiply currents by tau_m to get into right units 
        for i in range(0,N,1):
            I_ext = I_ext*tau_m
            Iext_e = Iext_e
            Irec[i,:]=Irec[i,:]*tau_m[i]
            Irec_e[i,:]=Irec_e[i,:]*tau_m[i]
            Irec_i[i,:]=Irec_i[i,:]*tau_m[i]
        
         # output
        return timePts, spikes, v, Irec, I_ext, Irec_e, Irec_i, Iext_e, Istim
    else:
         # output spike time array
        return spikes
