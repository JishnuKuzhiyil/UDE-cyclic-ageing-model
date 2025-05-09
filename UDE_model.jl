"""

This code runs the UDE based cyclic ageing model. Set condition index to the desired condition to simulate.
The code simulates corresponding cyclic ageing experiment and plots the results.

Model details: 
- Single particle model (SPM) with thermal dynamics
- Degradation mechsnisms include (i) SEI growth, (ii)pore blockage, (iii) lithium plating, (iv) particle cracking and SEI on cracks, (v) mechsnical damage
- Physics-based goverining equations are used for degradation mechsnisms (i)-(iv)
- Neural networks are used to model the mechanical damage (v) and its effect on the pore blockage

Code is writtern by Jishnu Ayyangatu Kuzhiyil, PhD student, University of Warwick, UK

"""
#Comment out the following lines to instantiate the project environment

# using Pkg #Package manager for Julia
# Pkg.activate(".") # Activate the project environment
# Pkg.instantiate() # install packages if not already installed


#Import required packages
using LinearAlgebra, DifferentialEquations, Plots, SparseArrays
using PreallocationTools, DelimitedFiles, Statistics, Revise
using Lux,ComponentArrays,StableRNGs

#import custom modules 
include("LGM50_parameters_SPMDeg.jl"); import .Para_set # import electrochemical parameter set based on Chen2020
include("SPM_Model_Aux.jl") ; import .SPM_Model_Aux # import auxilary functions for SPM model
include("Experiment.jl"); import .Experiment  # import functions for defining the experiment
include("Deg_model_Aux.jl"); import .Deg_model_Aux # import auxilary functions for degradation models
include("ExperimentConditions.jl"); import .ExperimentConditions # import functions for defining the experiment conditions

#ageing conditions
condition_names = [
    "0p3C_1C_100_0_25degC",    "0p3C_0p3C_100_0_25degC", "0p5C_0p3C_100_0_25degC",
     "0p3C_2C_100_0_25degC",   "0p3C_1C_4p2V_3p1V_25degC", "0p3C_0p3C_100_15_25degC",
     "0p3C_2C_4p2V_3p1V_25degC", "0p3C_1C_85_0_25degC",     "0p3C_1C_4V_2p5V_25degC", 
     "0p3C_1C_100_15_25degC",    "0p3C_1C_95_5_25degC",     "0p3C_2C_4V_2p5V_25degC",
    "0p3C_2C_100_0_0degC",     "0p3C_0p3C_100_0_0degC", "0p3C_1C_100_0_0degC",
      "0p3C_0p3C_100_0_45degC", "0p3C_1C_100_0_45degC",  "0p3C_2C_100_0_45degC",
     "0p3C_1C_85_0_45degC",      "0p3C_0p3C_100_0_10degC", "0p3C_1C_100_0_10degC", 
]

Condition_index =1   #Index of the condition to simulate, choose 1-21

#Define the neural networks and load its optimal parameters
rng = StableRNG(1111) #random number generator, for neural network initialisation (but it is replaced with optimal weights after that)

#Negative electrode
NN_LAM_Neg = Lux.Chain(Lux.Dense(4, 20, tanh),Lux.Dense(20, 20, tanh),  Lux.Dense(20, 1)) # Neural network for LAM degradation at negative electrode
NN_LAM_Neg_para, NN_LAM_Neg_st = Lux.setup(rng,NN_LAM_Neg) #initialise the neural network weights
NN_LAM_Neg_para = ComponentVector(NN_LAM_Neg_para) #converting the weights to a component array
NN_LAM_Neg_para[1:end] = Deg_model_Aux.neg_NN_para_weights #assigning the optimal weights to the neural network

#Positive electrode
NN_LAM_pos = Lux.Chain(Lux.Dense(4, 20, tanh),Lux.Dense(20, 20, tanh),  Lux.Dense(20, 1)) # Neural network for LAM degradation at positive electrode
NN_LAM_pos_para, NN_LAM_pos_st = Lux.setup(rng,NN_LAM_pos) #initialise the neural network weights
NN_LAM_pos_para = ComponentVector(NN_LAM_pos_para) #converting the weights to a component array
NN_LAM_pos_para[1:end] = Deg_model_Aux.pos__NN_para_weights #assigning the optimal weights to the neural network


Condition_name = condition_names[Condition_index] #Name of the condition to simulate
V_min, Volt_hold, V_max, Dis_SOC, Ch_SOC, tol, N_cyc_bet_RPT, N_RPTs, Ch_I, Dis_I, T∞ = ExperimentConditions.get_test_condition(Condition_name) # defining features of the cyclic ageing experiment
title_cond=ExperimentConditions.get_title_for_condition(Condition_name) # title for the condition, as used in the article
dis_voltage_cycle_num=2

#Optimal Physics Degradation parameters

# SEI parameters
K_SEI_ref = 1.098e-16
Ea_K_SEI = 74521.8/8.314
D_SEI_ref = 2.784e-22
Ea_D_SEI = 83202.0/8.314
β_pore_ref = 7.45e-10
Ea_β_pore = 26584.3/8.314

# LiP Model parameters
K_LiP_ref=1.03e-10   
Ea_K_LiP = 1590.0/8.314

#Cracking Parameters
k_cr = 3.77e-21
m_cr = 2.2

#Stress based LAM parameters
Ea_NN_pos=2187.0
Ea_NN_neg=1050.0


#Define experiment

#1 cycle of cyclic ageing experiment
Exp_cycle = Experiment.cyclic_ageing_experiment(;I_ch=Ch_I,I_dis=Dis_I,dt_max_factor=1.2,Rest_period=100.0,V_max=V_max,V_min=V_min,Volt_hold=Volt_hold,Dis_SOC=Dis_SOC,Ch_SOC = Ch_SOC) # 1 cycle
Exp_RPT = Experiment.Experiment.RPT_cycle # RPT cycle 
if Condition_index ==11 
    Ch_to5per = Experiment.Experiment_step(false,false,-5.0/3,"Ch_SOC",0.05,3600.0,50.0,nothing)
    Exp_RPT = [Exp_RPT;Ch_to5per]
end 
Dis_step=Experiment.Experiment_step(false,false,3.0,"Voltage",3.2,2*3600.0,100.0,nothing) #intial discharge step
Exp  = [Dis_step;repeat([Exp_RPT;repeat(Exp_cycle, N_cyc_bet_RPT)],N_RPTs);Exp_RPT] # Total experiment

# Struct to store simulation result from RPTs
mutable struct SimulationState
    RPT_num::Int
    Q_RPT::Vector{Float64}
    Time_RPT_end::Vector{Float64}
    LAM_anode::Vector{Float64}
    LAM_cathode::Vector{Float64}
    SEI_thickness::Vector{Float64}
    LiP_thickness::Vector{Float64}
    Crack_length::Vector{Float64}
    θ_soc0_neg::Vector{Float64}
    θ_soc0_pos::Vector{Float64}
    Resistance::Vector{Float64}
    Total_lithium::Vector{Float64}
    LAM_Neg_pore::Vector{Float64}
end

cycle_num=[0]
Step_nu = 1; t_last =0.0 # initial step number and time
ΔC = [0.0,0.0,0.0,0.0]  #ΔC_max_pos,ΔC_max_neg,ΔC_min_pos,ΔC_min_neg on a cycle
theta_soc0=[0.0,0.0] # stoichiomatries at end of the cycle at negative and positive electrodes
Deg_Para = (K_SEI_ref,Ea_K_SEI,D_SEI_ref,Ea_D_SEI,β_pore_ref,Ea_β_pore,K_LiP_ref,Ea_K_LiP,k_cr,m_cr,Ea_NN_pos,Ea_NN_neg) #Degradation parameters
sim_state = SimulationState(0, Float64[],Float64[],Float64[],Float64[],Float64[],Float64[],Float64[],Float64[],Float64[],Float64[],Float64[],Float64[])
para = ([Step_nu],[t_last],Exp,Deg_Para,ΔC,cycle_num,theta_soc0,sim_state)
T∞_curr = T∞


# Model state function
function TSPMD_model!(du, u, p, t)

    step_nu = p[1][1] #current experiment step number 
    I = p[3][step_nu].I_val - u[end] # Applied current, u[end] represents current during voltage hold

    @views c_n, dc_n = u[1:6], du[1:6]
    @views c_p, dc_p = u[7:12], du[7:12]
    δ_SEI = u[13];δ_LIP=u[14]; bn=u[15];bp=u[16];T=u[17]

    #Unpacking parameter tuple
    K_SEI_ref,Ea_K_SEI,D_SEI_ref,Ea_D_SEI,β_pore_ref,Ea_β_pore,K_LiP_ref,Ea_K_LiP,k_cr,m_cr,Ea_NN_pos,Ea_NN_neg = p[4]
    #Thermal dependancy of SEI, LIP and pore blockage parameters    
    D_SEI = D_SEI_ref * exp(-Ea_D_SEI * (1/T - 1/298.15))
    K_SEI = K_SEI_ref * exp(-Ea_K_SEI * (1/T - 1/298.15))
    β_pore = β_pore_ref * exp(Ea_β_pore * (1/T - 1/298.15))
    K_LiP = K_LiP_ref * exp(-Ea_K_LiP * (1/T - 1/298.15))

    #Anode and cathode potential calculation
    c_n_surf = 1.5 * u[6] - 0.5 * u[5]   # surface concentration at anode
    c_p_surf = 1.5 * u[12] - 0.5 * u[11] #surface concentration at cathode
    c_n_avg = SPM_Model_Aux.FVM_vol_avg(c_n)
    c_p_avg = SPM_Model_Aux.FVM_vol_avg(c_p)
    ΔC_n = c_n_surf  - c_n_avg
    ΔC_p = c_p_surf - c_p_avg

    j_n = I / (Para_set.A * Para_set.Ln) / bn
    j_n_0 = Para_set.Negative_exchange_current_density(c_n_surf, T)
    ηₕ = Para_set.Negative_OCP(c_n_surf) + (2 * Para_set.R * T / Para_set.F) * asinh(j_n / j_n_0)
    ηₕ_over = (2 * Para_set.R * T / Para_set.F) * asinh(j_n / j_n_0)

    j_p = -I / (Para_set.A * Para_set.Lp) / bp
    j_p_0 = Para_set.Positive_exchange_current_density(c_p_surf, T)
    ηₖ_over = (2 * Para_set.R * T / Para_set.F) * asinh(j_p / j_p_0)

    η_reaction =  ηₖ_over - ηₕ_over
    R0 = 2.2e-4
    R0_SEI =δ_SEI/(Para_set.A*Para_set.Ln*bn*SPM_Model_Aux.κ_SEI)
    η_ohmic = I * (R0 + R0_SEI)

    #SEI calculations
    η_SEI =  ηₕ - Deg_model_Aux.U_SEI
    J_SEI = Deg_model_Aux.J_SEI_eval(δ_SEI,η_SEI,T,D_SEI,K_SEI)
    du[13] = 0.5*J_SEI*Deg_model_Aux.V_SEI

    #LiP calculations
    J_LiP = Deg_model_Aux.J_LIP_eval(ηₕ,T,ΔC_n,K_LiP)
    du[14] = J_LiP*Deg_model_Aux.V_LiP 

    #LAM calculations
    du[15] =  - β_pore * (Para_set.F*J_SEI *bn )*3/Para_set.Rn
    du[16] = 0.0

    #Thermal dynamics
    Q_gen = abs(I*( η_reaction + η_ohmic) )       
    du[17] = (Para_set.h*Para_set.Cooling_Area*(T∞_curr - T) + Q_gen)/(Para_set.Vol*Para_set.Cp)

    #Particle cracking
    l_cr,a_cr= u[18],u[19]
    du[18],du[19] = Deg_model_Aux.Particle_crack_propogation(l_cr, a_cr,bn,ΔC_n,k_cr,m_cr)

    #SEI growth on cracks
    L_SEI_cr = u[20]
    J_SEI_cr = Deg_model_Aux.J_SEI_eval(L_SEI_cr,η_SEI,T,D_SEI,K_SEI)
    du[20] = 0.5*J_SEI_cr*Deg_model_Aux.V_SEI - du[19]*(L_SEI_cr-Para_set.L_SEI_cr0)/a_cr

    #Additional state variables
    du[21] = abs(I)/3600.0
    du[22] =  β_pore * (Para_set.F*J_SEI *bn )/Para_set.bn_0 # state variable representing Δϵ/ϵ_0

    #Flux at particle surface.
    N_surf_neg = j_n/Para_set.F +J_SEI + J_LiP + J_SEI_cr* (a_cr/bn)
    N_surf_pos = j_p/Para_set.F

    #Ficks law
    SPM_Model_Aux.neg_particle_diffusion_derivative!(dc_n, c_n, N_surf_neg, T)
    SPM_Model_Aux.pos_particle_diffusion_derivative!(dc_p, c_p, N_surf_pos, T)

    #Constant voltage control
    if p[3][step_nu].is_voltage_hold
        du[end] = SPM_Model_Aux.SPM_voltage(c_n_surf,c_p_surf, bn, bp, u[13],u[14], I, T) - p[3][step_nu].extra
    else
        du[end]=u[end]
    end

    return nothing
end


function condition(u, t, integrator)

    Step_nu = integrator.p[1][1]# current step number
    t_last = integrator.p[2][1] # end time of last step (start time of ongoing step)
    I = integrator.p[3][Step_nu].I_val - u[end] #current
    Exp_step = integrator.p[3][Step_nu] #Tuple containing details of ongoing experimental step 
    δ_SEI = u[13];δ_LIP=u[14]; bn=u[15];bp=u[16];T=u[17]

    if Exp_step.is_voltage_hold #If the ongoing step is voltage hold

        if integrator.p[3][Step_nu-1].Termination_condition == "Ch_SOC" # Termination condition for charging was based on SOC and it wasn't met during CC charging phase

            SOC_condn =  integrator.p[3][Step_nu-1].Termination_val - u[21]/integrator.p[end].Q_RPT[end]  # positive until condition is met
            if Exp_step.Termination_condition == "Time"
                Time_condn = Exp_step.Termination_val-(t - t_last) #positive until condition is met.
                return Time_condn*SOC_condn
            elseif Exp_step.Termination_condition == "Abs_Current"
                Curr_condition = abs(I) - Exp_step.Termination_val #positive until condition is met
                return SOC_condn*Curr_condition      
            end
        end

        if Exp_step.Termination_condition == "Time"
            return t - t_last - Exp_step.Termination_val
        elseif Exp_step.Termination_condition == "Abs_Current"
            return abs(I) - Exp_step.Termination_val       
        end

    else #ongoing experimental step is current controlled

        if Exp_step.Termination_condition == "Voltage"

            c_n_surf = 1.5 * u[6] - 0.5 * u[5]   
            c_p_surf = 1.5 * u[12] - 0.5 * u[11]

            return SPM_Model_Aux.SPM_voltage(c_n_surf,c_p_surf, bn, bp, δ_SEI,δ_LIP, I, T) - Exp_step.Termination_val

        elseif Exp_step.Termination_condition == "Time"

            return t - t_last - Exp_step.Termination_val

        elseif Exp_step.Termination_condition == "Dis_SOC"

            Q_RPT_latest = integrator.p[end].Q_RPT[end]
            SOC_condn =  u[21]/Q_RPT_latest-(1-Exp_step.Termination_val) #positive until condition is met
            c_n_surf = 1.5 * u[6] - 0.5 * u[5]   
            c_p_surf = 1.5 * u[12] - 0.5 * u[11] 
            vol_condition= SPM_Model_Aux.SPM_voltage(c_n_surf,c_p_surf, bn, bp, δ_SEI,δ_LIP, I, T) - 2.5 # positive until lower cut off voltage is reached
            return SOC_condn*vol_condition

        elseif Exp_step.Termination_condition == "Ch_SOC" #CC charging step with SOC termination

            Q_RPT_latest = integrator.p[end].Q_RPT[end]
            SOC_condn =  u[21]/Q_RPT_latest-(Exp_step.Termination_val) #positive until condition is met
            c_n_surf = 1.5 * u[6] - 0.5 * u[5]   
            c_p_surf = 1.5 * u[12] - 0.5 * u[11] 
            vol_condition= SPM_Model_Aux.SPM_voltage(c_n_surf,c_p_surf, bn, bp, δ_SEI,δ_LIP, I,T) - 4.2 #positive until upper cut off voltage is reached

            return vol_condition*SOC_condn
        end

    end

end


function LAM_pos_NN_update(b_p_old,θ_soc0_neg,ΔC_min_pos,ΔC_max_pos,NN_LAM_pos_para=NN_LAM_pos_para,NN_LAM_pos_st=NN_LAM_pos_st)

    LAM_per_old = (1-b_p_old/Para_set.bp_0)*100.0
    input1 = LAM_per_old
    input2 = Deg_model_Aux.positive_else_zero(0.25-θ_soc0_neg)*100.0
    input3 = -ΔC_min_pos/1000.0 *(1.21/1.36)
    input4 = ΔC_max_pos/1000.0 *(1.21/1.36)

    input = [input1,input2,input3,input4]    
    ΔLAM_per =   NN_LAM_pos(input,NN_LAM_pos_para,NN_LAM_pos_st)[1][1]^2/(1+LAM_per_old) 

  return  -(ΔLAM_per/100.0) * Para_set.bp_0 

end 

function LAM_neg_NN_update(b_n_old,θ_soc0_neg,ΔC_min_neg,ΔC_max_neg,NN_LAM_Neg_para=NN_LAM_Neg_para,NN_LAM_Neg_st=NN_LAM_Neg_st)

    LAM_per_old = (1-b_n_old/Para_set.bn_0)*100.0
    input1 = LAM_per_old
    input2 = Deg_model_Aux.positive_else_zero(0.25-θ_soc0_neg)*100.0
    input3 = -ΔC_min_neg/1000.0 
    input4 = ΔC_max_neg/1000.0
  
    input = [ input1 ,input2,input3,input4]    
    ΔLAM_per = NN_LAM_Neg(input,NN_LAM_Neg_para,NN_LAM_Neg_st)[1][1]^2/(1+LAM_per_old)   #30

    return  -(ΔLAM_per/100.0)*Para_set.bn_0 

end 

function affect!(integrator)

    Erase_sol = true
    skip_nu=0
    Step_nu = integrator.p[1][1]
    t_step_start = integrator.p[2][1]
    t_step_end = integrator.t
    last_Ah=integrator.u[21]
    integrator.u[21]=0.0

    if (integrator.p[3][Step_nu].extra == "Charge_step") && (integrator.p[3][Step_nu].Termination_condition == "Ch_SOC")
        if abs(last_Ah/integrator.p[end].Q_RPT[end] - integrator.p[3][Step_nu].Termination_val) < 0.001
            skip_nu=1
        else 
            integrator.u[21]=last_Ah
        end
    end

    if ((integrator.p[3][Step_nu].extra == "start_of_cycle") || (integrator.p[3][Step_nu].extra == "start_of_RPT")) 
        cycle_num = integrator.p[6][1]
        integrator.p[6][1] +=1
        T = integrator.u[17]

        ΔC_max_neg= integrator.p[5][2]
        ΔC_min_neg = integrator.p[5][4]
        θ_soc0_neg = integrator.p[7][1]       
        integrator.u[15] +=  LAM_neg_NN_update(integrator.u[15],θ_soc0_neg, ΔC_min_neg, ΔC_max_neg)* exp(-Ea_NN_neg*(1/T - 1/298.15))

        ΔC_max_pos = integrator.p[5][1]
        ΔC_min_pos = integrator.p[5][3]     
        integrator.u[16] += LAM_pos_NN_update(integrator.u[16],θ_soc0_neg,ΔC_min_pos,ΔC_max_pos) *exp(-Ea_NN_pos*(1/T - 1/298.15))
    end

    if integrator.p[3][Step_nu].extra == "Charge_step"
        C_n_array = @view(integrator.sol[1:6,:])
        C_p_array = @view(integrator.sol[7:12,:])

        weights = [zeros(4,1)...,-0.5,1.5] .- ([i^3 - (i-1)^3 for i in 1:6]./6^3)        
        ΔC_n_max = maximum([sum(weights .*col) for col in eachcol(C_n_array)])
        ΔC_p_min = minimum([sum(weights .* col) for col in eachcol(C_p_array)])

        integrator.p[5][2] = ΔC_n_max
        integrator.p[5][3] = ΔC_p_min
    end

    if integrator.p[3][Step_nu].extra == "Discharge_step"
        C_n_array = @view(integrator.sol[1:6,:])
        C_p_array = @view(integrator.sol[7:12,:])

        weights = [zeros(4,1)...,-0.5,1.5] .- ([i^3 - (i-1)^3 for i in 1:6]./6^3)   
        ΔC_n_min = minimum([sum(weights .* col) for col in eachcol(C_n_array)])
        ΔC_p_max = maximum([sum(weights .* col) for col in eachcol(C_p_array)])

        integrator.p[5][4] = ΔC_n_min
        integrator.p[5][1] = ΔC_p_max

        integrator.p[7][1] = sum(([i^3 - (i-1)^3 for i in 1:6]./6^3).*C_n_array[1:6,end])/Para_set.c_s_max_neg
        integrator.p[7][2] = sum(([i^3 - (i-1)^3 for i in 1:6]./6^3).*C_p_array[1:6,end])/Para_set.c_s_max_pos

        cycle_num = integrator.p[6][1]
        if cycle_num==dis_voltage_cycle_num
            global Time_vec = Array(integrator.sol.t).-integrator.sol.t[1]
            Dis_sol = Array(integrator.sol)
            C_n_surf_vec = 1.5.*Dis_sol[6,:] .- 0.5.*Dis_sol[5,:]
            C_p_surf_vec = 1.5.*Dis_sol[12,:] .- 0.5.*Dis_sol[11,:]
            bn_vec = Dis_sol[15,:] ; bp_vec = Dis_sol[16,:]
            δ_SEI_vec = Dis_sol[13,:]; global T_vec = Dis_sol[17,:]
            δ_LIP_vec = Dis_sol[14,:]
            I = integrator.p[3][Step_nu].I_val
            V=similar(T_vec)

            for i in eachindex(T_vec)
                V[i] = SPM_Model_Aux.SPM_voltage(C_n_surf_vec[i],C_p_surf_vec[i],bn_vec[i],bp_vec[i],δ_SEI_vec[i],δ_LIP_vec[i],I,T_vec[i])
            end

            global Voltage_vec = V
        end
    end

    if integrator.p[3][Step_nu].extra=="Capacity_mes_step"
        Soln_50per=integrator.sol((t_step_end + t_step_start)/2)
        Cn_surf_50per = 1.5*Soln_50per[6] - 0.5*Soln_50per[5]
        Cp_surf_50per = 1.5*Soln_50per[12] - 0.5*Soln_50per[11]
        bn_50per,bp_50per = Soln_50per[15],Soln_50per[16]
        δ_SEI_50per = Soln_50per[13]
        δ_LIP_50per = Soln_50per[14]
        T_50per = 298.15
        I = integrator.p[3][Step_nu].I_val
        Res=SPM_Model_Aux.Resistance_calc(Cn_surf_50per, Cp_surf_50per, bn_50per, bp_50per, δ_SEI_50per,δ_LIP_50per, 2.5, T_50per)
        Q_mes = (t_step_end - t_step_start)/3600.0 * (5/3)

        Cn_avg = SPM_Model_Aux.FVM_vol_avg(integrator.u[1:6])
        Cp_avg = SPM_Model_Aux.FVM_vol_avg(integrator.u[7:12])
        bn,bp = integrator.u[15],integrator.u[16]
        Ntot_anode  = Para_set.F*Para_set.A*Para_set.Ln*(bn*Para_set.Rn/3)*Cn_avg/3600.0
        Ntot_cathode = Para_set.F*Para_set.A*Para_set.Lp*(bp*Para_set.Rp/3)*Cp_avg/3600.0
        Ntot = Ntot_anode + Ntot_cathode

        integrator.p[end].RPT_num += 1
        push!(integrator.p[end].Q_RPT, Q_mes) 
        push!(integrator.p[end].Time_RPT_end, t_step_end) 
        push!(integrator.p[end].LAM_anode, 1.0 - integrator.u[15]./Para_set.bn_0)
        push!(integrator.p[end].LAM_cathode,1.0 - integrator.u[16]./Para_set.bp_0)
        push!(integrator.p[end].SEI_thickness, integrator.u[13]*1e9)
        push!(integrator.p[end].LiP_thickness, integrator.u[14]*1e9)
        push!(integrator.p[end].Crack_length, integrator.u[18]*1e9) 
        push!(integrator.p[end].θ_soc0_neg, integrator.p[7][1])
        push!(integrator.p[end].θ_soc0_pos, integrator.p[7][2])  
        push!(integrator.p[end].Resistance, Res)
        push!(integrator.p[end].Total_lithium, Ntot)
        println("Running RPT number : $(integrator.p[end].RPT_num)") 
    end

    if Step_nu==length(integrator.p[3])
        terminate!(integrator)
        println("Simulation ended successfully")
        return nothing  
    end

    if integrator.p[3][Step_nu+1].extra=="start_of_RPT"
        integrator.u[17]=298.15
        global T∞_curr = 298.15
    elseif integrator.p[3][Step_nu].extra=="Capacity_mes_step"
        integrator.u[17]=T∞
        global T∞_curr = T∞
    end

    if integrator.p[3][Step_nu].is_voltage_hold
        integrator.u[end]=0.0
    end 

    next_step_duration = integrator.p[3][Step_nu+1].duration + t_step_end
    next_step_dt_max = integrator.p[3][Step_nu+1].dt_max
    integrator.p[1][1] += 1+skip_nu
    integrator.p[2][1] = t_step_end
    integrator.opts.dtmax = next_step_dt_max
    reinit!(integrator, integrator.u, t0=t_step_end, tf=next_step_duration,erase_sol=Erase_sol)
end


cb = ContinuousCallback(condition, affect!)


u0 = [29866.0 .* ones(6); 17038.0 .* ones(6);Para_set.L_SEI0;Para_set.L_LIP0;Para_set.bn_0;Para_set.bp_0;298.15; Deg_model_Aux.l_cr_0;Deg_model_Aux.a_cr_0;Para_set.L_SEI_cr0;0.0;0.0;0.0]
tspan = (0.0, 2*3600.0)

Inertia_matrix=spdiagm([ones(length(u0)-1);0])
DAE_system  = ODEFunction(TSPMD_model!, mass_matrix=Inertia_matrix)
prob = ODEProblem(DAE_system, u0, tspan, para)
sol = solve(prob, Rosenbrock23(), reltol=tol , abstol=tol ,callback=cb);


Capacity = sim_state.Q_RPT
LAM_anode = sim_state.LAM_anode
LAM_cathode = sim_state.LAM_cathode
SEI_thickness = sim_state.SEI_thickness
LiP_thickness = sim_state.LiP_thickness
x = [(i-1)*N_cyc_bet_RPT  for i in 1:length(Capacity)]


"""___________________________Importing Experiment data_________________________________________________________________________________________"""

gr() #GR backend for Plots.jl

#Import capacity resistance and DM experiment data
Capacity_data_file_name = "Exp_Data_Processed/$(Condition_name)_capacity.txt"
DM_file_name = "Exp_Data_Processed/$(Condition_name)_DM.txt"
Res_file_name = "Exp_Data_Processed/$(Condition_name)_resistance.txt"

#extracting capacity exp data
data_cap = readdlm(Capacity_data_file_name, ',', Float64, '\n', header=true)[1]
cycles_capacity = data_cap[:, 1] ; Capacity_mean_data = data_cap[:, 2] ; Capacity_std_data = data_cap[:, 3]

#extracting resistance data
data_res = readdlm(Res_file_name, ',', Float64, '\n', header=true)[1]
cycles_res = data_res[:, 1] ; Resistance_mean_data = data_res[:, 2] ; Resistance_std_data = data_res[:, 3]

#extracting DM data
data_DM = readdlm(DM_file_name, ',', Float64, '\n', header=true)[1]
cycles_DM = data_DM[:, 1] ; Anode_lam = 100.0 .- data_DM[:, 4]/data_DM[1,4]*100.0 ; Anode_Cap_std = data_DM[:, 5]/data_DM[1,4]*100.0
Anode_lam = 100.0 .- data_DM[:, 4]/data_DM[1,4]*100.0 ; Anode_Cap_std = data_DM[:, 5]/data_DM[1,4]*100.0
Cathode_lam = 100.0 .- data_DM[:, 6]/data_DM[1,6]*100.0 ; Cathode_Cap_std = data_DM[:, 7]/data_DM[1,6]*100.0
Total_lithium  = data_DM[:, 8] ; Total_lithium_std = data_DM[:, 9]



"""___________________________________________Plotting data_________________________________________________________________________________________"""

fs = 13
del_fs = 4

P_capacity = plot(x, 100 .*Capacity./Capacity[1], m=:circle, mc=:orange, ms=5, legend=false, lw=3, xlabel="Cycle number", ylabel="Capacity [Ah]", framestyle=:box, tickfontsize=fs, labelfontsize=fs+del_fs, size=(400,400), title=title_cond)
plot!(P_capacity,cycles_capacity,100 .*Capacity_mean_data./Capacity_mean_data[1],m=:square,ms=5,ribbon=100 .*Capacity_std_data./Capacity_mean_data[1],fillalpha=0.2,label="Experiment",mc=:black,color=:black,fillcolor=:black,dpi=300)


P_capacity = plot(x, Capacity, m=:circle, mc=:orange, ms=5, legend=false, lw=3, xlabel="Cycle number", ylabel="Capacity [Ah]", framestyle=:box, tickfontsize=fs, labelfontsize=fs+del_fs, size=(400,400), title=title_cond)
plot!(P_capacity,cycles_capacity,Capacity_mean_data,m=:square,ms=5,ribbon=Capacity_std_data,fillalpha=0.2,label="Experiment",mc=:black,color=:black,fillcolor=:black,dpi=300)
plot!(P_capacity,ylim=(3.0,4.9),yticks = 3.0:0.5:4.9)

P_resistance = plot(x, 1e3 .* sim_state.Resistance, m=:circle, mc=:orange, ms=5, legend=false, lw=3, xlabel="Cycle number", ylabel="Resistance [mΩ]", framestyle=:box, tickfontsize=fs, labelfontsize=fs+del_fs, size=(400,400), title=title_cond)
plot!(P_resistance,cycles_res,Resistance_mean_data,m=:square,ms=5,ribbon=Resistance_std_data,fillalpha=0.4,fillcolor=:black,label="Experiment",mc=:black,color=:black,dpi=300)
plot!(P_resistance,ylim=(24,44),yticks = 24:5:44)

p_LAM_cathode = plot(x, 100 .* LAM_cathode, m=:circle, mc=:orange, ms=5, legend=false, lw=3, xlabel="Cycle number", ylabel="LAM PE [%]", framestyle=:box, tickfontsize=fs, labelfontsize=fs+del_fs, size=(400,400), title=title_cond)
 scatter!(p_LAM_cathode,cycles_DM,Cathode_lam,yerr=Cathode_Cap_std,fillalpha=0.4,fillcolor=:black,label="Experiment",m=:square,ms=7,mc=:black,dpi=300)
 plot!(p_LAM_cathode,ylim=(0,33),yticks = 0:10:30)

p_LAM_anode = plot(x, 100 .* LAM_anode, m=:circle, mc=:orange, ms=5, legend=false, lw=3, xlabel="Cycle number", ylabel="LAM NE [%]", framestyle=:box, tickfontsize=fs, labelfontsize=fs+del_fs, size=(400,400), title=title_cond)
 scatter!(p_LAM_anode,cycles_DM,Anode_lam,yerr=Anode_Cap_std,fillalpha=0.4,fillcolor=:black,label="Experiment",m=:square,ms=7,color=:black,dpi=300)
 plot!(p_LAM_anode,ylim=(0,33),yticks = 0:10:30)

 p_LLI = plot(x, 100.0 .- 100.0 .* sim_state.Total_lithium./sim_state.Total_lithium[1], m=:circle, mc=:orange, ms=5, legend=false, lw=3, xlabel="Cycle number", ylabel="LLI [%]", framestyle=:box, tickfontsize=fs, labelfontsize=fs+del_fs, size=(400,400), title=title_cond)
scatter!(p_LLI,cycles_DM,100.0 .- 100.0 .* Total_lithium./Total_lithium[1],yerr=100.0 .*Total_lithium_std./Total_lithium[1],fillalpha=0.4,fillcolor=:black,label="Experiment",m=:square,ms=7,color=:black,dpi=300)
plot!(p_LLI,ylim=(0,40),yticks = 0:10:40)
 P=plot(P_capacity,P_resistance,p_LAM_cathode,p_LAM_anode,p_LLI,layout=(3,2),size=(800,800))





# """___________________________________________Saving data_________________________________________________________________________________________"""

function save_model_results(x, Capacity, LAM_anode, LAM_cathode, sim_state, Condition_name )
    # Calculate the quantities to save
    LAM_NE = 100 .* LAM_anode
    LAM_PE = 100 .* LAM_cathode
    LLI = 100.0 .- 100.0 .* sim_state.Total_lithium ./ sim_state.Total_lithium[1]
    Resistance = 1e3 .* sim_state.Resistance

    # Define the file name
    file_name = "Output_data/UDE/results$(Condition_name ).txt"

    # Open the file in write mode
    open(file_name, "w") do file
        # Write the headers
        println(file, "cycles, capacity, LAM_NE, LAM_PE, LLI, Resistance")

        # Write the data
        for i in 1:length(x)
            println(file, "$(x[i]), $(Capacity[i]), $(LAM_NE[i]), $(LAM_PE[i]), $(LLI[i]), $(Resistance[i])")
        end
    end
end

# save_model_results(x, Capacity, LAM_anode, LAM_cathode, sim_state, Condition_name )


display(P)

