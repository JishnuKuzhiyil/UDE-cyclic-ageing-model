module ExperimentConditions

function get_test_condition(condition_name)
    # Default values
    params = Dict(
        "V_min" => 2.5,
        "Volt_hold" => true,
        "V_max" => 4.2,
        "Dis_SOC" => nothing,
        "Ch_SOC" => nothing,
        "tol" => 1e-5,
        "N_cyc_bet_RPT" => 25,
        "N_RPTs" => 40,
        "Ch_I" => 0.3 * 5.0,
        "Dis_I" => 5.0,
        "T∞" => 298.15
    )

    # Condition-specific modifications
    if condition_name == "0p3C_0p3C_100_0_25degC"
        params["N_RPTs"] = 27
        params["Dis_I"] = 0.3 * 5.0
    elseif condition_name == "0p3C_1C_100_0_25degC"
        params["N_RPTs"] = 41
    elseif condition_name == "0p3C_1C_100_0_0degC"
        params["N_cyc_bet_RPT"] = 50
        params["N_RPTs"] = 50
        params["T∞"] = 298.15 - 25.0
    elseif condition_name == "0p3C_1C_100_0_45degC"
        params["N_RPTs"] = 14
        params["T∞"] = 298.15 + 20.0
    elseif condition_name == "0p3C_1C_100_15_45degC"
        params["N_RPTs"] = 13
        params["T∞"] = 298.15 + 20.0
        params["Dis_SOC"] = 0.15
    elseif condition_name == "0p3C_1C_85_0_45degC"
        params["N_RPTs"] = 13
        params["T∞"] = 298.15 + 20.0
        params["Ch_SOC"] = 0.83

    elseif condition_name == "0p3C_0p3C_100_0_45degC"
        params["N_RPTs"] = 9
        params["Dis_I"] = 0.3 * 5.0
        params["T∞"] = 298.15 + 20.0
    elseif condition_name == "0p3C_2C_100_0_45degC"
        params["N_RPTs"] = 20
        params["Dis_I"] = 10.0
        params["T∞"] = 298.15 + 20.0
    elseif condition_name == "0p3C_2C_100_0_0degC"
        params["N_cyc_bet_RPT"] = 50
        params["N_RPTs"] = 45
        params["Dis_I"] = 10.0
        params["T∞"] = 298.15 - 25.0
    elseif condition_name == "0p3C_0p3C_100_0_0degC"
        params["N_cyc_bet_RPT"] = 50
        params["N_RPTs"] = 22
        params["Dis_I"] = 5.0 * 0.3
        params["T∞"] = 298.15 - 25.0
    elseif condition_name == "0p5C_0p3C_100_0_0degC"
        params["N_RPTs"] = 25
        params["Ch_I"] = 5.0 * 0.5
        params["Dis_I"] = 5.0 * 0.3
        params["T∞"] = 298.15 - 25.0
    elseif condition_name == "0p5C_0p3C_100_0_25degC"
        params["N_RPTs"] = 32
        params["Ch_I"] = 5.0 / 2.0
        params["Dis_I"] = 0.3 * 5.0
        params["tol"] = 1e-7
    elseif condition_name ==  "0p5C_0p3C_100_0_10degC"
        params["N_RPTs"] = 32
        params["Ch_I"] = 5.0 * 0.5
        params["Dis_I"] = 5.0 * 0.3
        params["T∞"] = 298.15 - 15.0
    elseif condition_name == "0p3C_2C_100_0_25degC"
        params["N_RPTs"] = 67
        params["Dis_I"] = 10
    elseif condition_name == "0p3C_1C_4p2V_3p1V_25degC"
        params["N_RPTs"] = 25
        params["V_min"] = 3.1
    elseif condition_name == "0p3C_1C_85_0_25degC"
        params["N_RPTs"] = 48
        params["Ch_SOC"] = 0.825
    elseif condition_name == "0p3C_1C_4V_2p5V_25degC"
        params["N_RPTs"] = 40
        params["V_max"] = 4.0
    elseif condition_name == "0p3C_2C_4V_2p5V_25degC"
        params["N_RPTs"] = 40
        params["Dis_I"] = 10.0
        params["V_max"] = 4.0
    elseif condition_name == "0p7C_0p3C_100_0_25degC"
        params["N_RPTs"] = 10
        params["Ch_I"] = 0.7 * 5.0
        params["Dis_I"] = 0.3 * 5.0
        params["tol"] = 1e-7
    elseif condition_name == "0p3C_1C_100_15_25degC"
        params["Dis_SOC"] = 0.15
        params["N_RPTs"] = 44
    elseif condition_name == "0p3C_0p3C_100_15_25degC"
        params["N_RPTs"] = 30
        params["Dis_I"] = 5.0 * 0.3
        params["Dis_SOC"] = 0.15
    elseif condition_name == "0p3C_1C_95_5_25degC"
        params["Dis_SOC"] = 0.05
        params["Ch_SOC"] = 0.90 #For this condition, the cell will be charged to 5% after RPT discharge. 
        params["N_RPTs"] = 45
    elseif condition_name == "0p3C_2C_4p2V_3p1V_25degC"
        params["Dis_I"] = 10.0
        params["V_min"] = 3.1
        params["N_RPTs"] = 43

    elseif condition_name == "0p3C_1C_100_0_10degC"
        params["N_RPTs"] = 42
        params["T∞"] = 298.15 + -15.0
        params["N_cyc_bet_RPT"] = 40

    elseif condition_name == "0p3C_0p3C_100_0_10degC"
        params["N_RPTs"] = 22
        params["Dis_I"] = 0.3 * 5.0
        params["N_cyc_bet_RPT"] = 40
        params["T∞"] = 298.15 + -15.0

    elseif condition_name == "0p3C_2C_100_0_10degC"
        params["N_RPTs"] = 60
        params["N_cyc_bet_RPT"] = 80
        params["Dis_I"] = 10.0
        params["T∞"] = 298.15 + -15.0

    end

    # Unpack the dictionary into individual variables
    V_min, Volt_hold, V_max, Dis_SOC, Ch_SOC, tol, N_cyc_bet_RPT, N_RPTs, Ch_I, Dis_I, T∞ = 
        params["V_min"], params["Volt_hold"], params["V_max"], params["Dis_SOC"], params["Ch_SOC"], 
        params["tol"], params["N_cyc_bet_RPT"], params["N_RPTs"], params["Ch_I"], params["Dis_I"], params["T∞"]

    return V_min, Volt_hold, V_max, Dis_SOC, Ch_SOC, tol, N_cyc_bet_RPT, N_RPTs, Ch_I, Dis_I, T∞
end



function get_title_for_condition(condition_name)
    condition_names = [
        "0p3C_1C_100_0_25degC",    "0p3C_0p3C_100_0_25degC", "0p5C_0p3C_100_0_25degC",
         "0p3C_2C_100_0_25degC",   "0p3C_1C_4p2V_3p1V_25degC", "0p3C_0p3C_100_15_25degC",
         "0p3C_2C_4p2V_3p1V_25degC", "0p3C_1C_85_0_25degC",     "0p3C_1C_4V_2p5V_25degC", 
         "0p3C_1C_100_15_25degC",    "0p3C_1C_95_5_25degC",     "0p3C_2C_4V_2p5V_25degC",
        "0p3C_2C_100_0_0degC",     "0p3C_0p3C_100_0_0degC", "0p3C_1C_100_0_0degC",
          "0p3C_0p3C_100_0_45degC", "0p3C_1C_100_0_45degC",  "0p3C_2C_100_0_45degC",
         "0p3C_1C_85_0_45degC",      "0p3C_0p3C_100_0_10degC", "0p3C_1C_100_0_10degC", 
    ]

    titles = [
        "0.3C-1C | 100%-0% | 25°C", "0.3C-0.3C | 100%-0% | 25°C", "0.5C-0.3C | 100%-0% | 25°C",
        "0.3C-2C | 100%-0% | 25°C", "0.3C-1C | 4.2V-3.1V | 25°C", "0.3C-0.3C | 100%-15% | 25°C",
        "0.3C-2C | 4.2V-3.1V | 25°C", "0.3C-1C | 85%-0% | 25°C", "0.3C-1C | 4V-2.5V | 25°C",
        "0.3C-1C | 100%-15% | 25°C", "0.3C-1C | 95%-5% | 25°C", "0.3C-2C | 4V-2.5V | 25°C",
        "0.3C-2C | 100%-0% | 0°C", "0.3C-0.3C | 100%-0% | 0°C", "0.3C-1C | 100%-0% | 0°C",
        "0.3C-0.3C | 100%-0% | 45°C", "0.3C-1C | 100%-0% | 45°C", "0.3C-2C | 100%-0% | 45°C",
        "0.3C-1C | 85%-0% | 45°C", "0.3C-0.3C | 100%-0% | 10°C", "0.3C-1C | 100%-0% | 10°C"]

    index = findfirst(x -> x == condition_name, condition_names)
    if isnothing(index)
        return "Unknown condition"
    else
        return titles[index]
    end
end




end