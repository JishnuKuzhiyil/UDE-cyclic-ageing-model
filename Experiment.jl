module Experiment

using MAT
# Define the custom struct contsaining information needed for each step of the experiment
struct Experiment_step

    is_voltage_hold::Bool  # Boolean to indicate if the step is a voltage hold step
    is_Q_mes_step::Bool    # Boolean to indicate if the step is a capacity measurement step in the RPT cycle
    I_val::Float64         # Current value for the step [A]
    Termination_condition::String  # Termination condition for the step (eg: Voltage, Time, etc)
    Termination_val::Float64  # Termination value for the step 
    duration::Float64 # Duration of the step [s]
    dt_max::Float64 # Maximum solver time step for the experiment step [s]
    extra::Union{Float64, Nothing,String} # Extra information for the step (eg: Voltage value for voltage hold step, etc)
    
end
# Define the RPT cycle
RPT_cycle = [
    Experiment_step(false,false,0.0,"Time",3600.0,5000.0,1000.0,"start_of_RPT"); # Rest for 1 hr
    Experiment_step(false,false,-5.0/3,"Voltage",4.2,3600.0*5.0,500.0,nothing);  # CC Charging at C/3 to 4.2V
    Experiment_step(true,false,-5.0/3.0,"Abs_Current",0.25,50000.0,500.0,4.2);   # CV Charging at 4.2V until current falls below 0.25A
    Experiment_step(false,false,0.0,"Time",3600.0,5000.0,1000.0,nothing);  # Rest for 1 hr
    Experiment_step(false,true,5.0/3.0,"Voltage",2.5,3600.0*5.0,500.0,"Capacity_mes_step") # CC Discharging at C/3 to 2.5V to measure capacity
]

function cyclic_ageing_experiment(;I_ch=2.0,I_dis=2.0,V_max=4.2,V_min=2.5,CV_curr_lim=0.25,Rest_period=3600.0,dt_max_factor=1.0,Volt_hold=true,Dis_SOC=nothing,Ch_SOC=nothing)

    I_1C = 5.0
    dur_ch = 1.9*3600.0*I_1C/abs(I_ch)*1.2; dt_max_ch = 500.0*dt_max_factor
    dur_dis = 1.9*3600.0*I_1C/abs(I_dis); dt_max_dis = 500.0*dt_max_factor
    
    # Define the experiment steps
    Cycle_exp = [
        Experiment_step(false, false, 0.0, "Time", Rest_period, 1.5 * Rest_period, Rest_period / 3, "start_of_cycle"), #Rest for "Rest period" seconds
        
    ]

    if Ch_SOC === nothing
        push!(Cycle_exp,Experiment_step(false, false, -abs(I_ch), "Voltage", V_max, dur_ch, dt_max_ch, "Charge_step"))  # Charge at I_ch Amps to V_max
    else
        push!(Cycle_exp, Experiment_step(false, false, -abs(I_ch), "Ch_SOC", Ch_SOC, dur_ch, dt_max_ch, "Charge_step")) # Charge at I_ch Amps until SOC reaches Ch_SOC
    end
    
    if Volt_hold
        push!(Cycle_exp, Experiment_step(true, false, -abs(I_ch), "Abs_Current", CV_curr_lim, 7200.0, 500.0 * dt_max_factor, V_max)) # Hold at V_max until current falls below CV_curr_lim
    end
    
    push!(Cycle_exp, Experiment_step(false, false, 0.0, "Time", Rest_period, 1.5 * Rest_period, Rest_period / 3, nothing)) # Rest for "Rest period" seconds

    if (Dis_SOC !== nothing)
        push!(Cycle_exp,Experiment_step(false, false, abs(I_dis), "Dis_SOC", Dis_SOC, dur_dis, dt_max_dis, "Discharge_step")) # Discharge at I_dis Amps until SOC reaches Dis_SOC
    else 
        push!(Cycle_exp,Experiment_step(false, true, abs(I_dis), "Voltage", V_min, dur_dis, dt_max_dis, "Discharge_step")) # Discharge at I_dis Amps to V_min
    end
    
    return Cycle_exp
end


function get_calendar_data(SOC,Temperature)

    file_path ="M://Calendar_ageing_data/calAnal_$(SOC)Per_$(Temperature)degC.mat"
    mat =matread(file_path)["calResults"]
    days=mat["Time_d"]
    Capacity=mat["Capacity"]["meanCap"]
    Capacity_std=mat["Capacity"]["meanCapStdErr"]
    return days,Capacity,Capacity_std
end


function storage_step(storage_SOC,Storage_duration_btn_RPT_days)

    Time_to_storage_SOC = (3)*3600.0*(1.0-storage_SOC)
    Storage_duration_btn_RPT  = Storage_duration_btn_RPT_days*24*3600.0
    Ageing_exp = [Experiment_step(false,false,-5.0/3.0,"Voltage",4.2,3600.0*5.0,500.0,nothing);
                  Experiment_step(true,false,-5.0/3.0,"Abs_Current",0.25,50000.0,500.0,4.2);
                  Experiment_step(false,false,0.0,"Time",3600.0,5000.0,1000.0,nothing);
                  Experiment_step(false,true,5.0/3.0,"Time",Time_to_storage_SOC,Time_to_storage_SOC*2,500.0,nothing);
                 Experiment_step(false,false,0.0,"Time",Storage_duration_btn_RPT,Storage_duration_btn_RPT*2,Storage_duration_btn_RPT/5.0,nothing)]
    return Ageing_exp

end


function calendar_ageing_experiment(start_day;SOC=100,Temperature=25)

    calendar_ageing_days,_,_= get_calendar_data(SOC,Temperature)
    calendar_ageing_days =calendar_ageing_days[start_day:end].-calendar_ageing_days[start_day]
    Exp=[]

    for i in 1:length(calendar_ageing_days)-1

        push!(Exp,RPT_cycle...)
        storage_duration_btw_RPTs_days = (calendar_ageing_days[i+1]-calendar_ageing_days[i])
        push!(Exp,storage_step(SOC/100.0,storage_duration_btw_RPTs_days)...)       
    end
    push!(Exp,RPT_cycle...)
    
    return Exp
end




end

 


