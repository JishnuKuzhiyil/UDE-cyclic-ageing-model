module Para_set

    export F, R, T_ref, Ln, Lp, A, Rn, Rp, c_e, c_s_max_neg, c_s_max_pos, bn_0, bp_0, ϵn_0, ϵp_0, Ds_n, Ds_p
    export Negative_exchange_current_density, Positive_exchange_current_density, Positive_OCP, Negative_OCP

    # Constants and reference values
    const F = 96485.33289
    const R = 8.3144598
    const T_ref = 298.15
    const c_e = 1000.0  # electrolyte concentration (const in SPM model) [mol/m^3]

    # Geometrical parameters
    const Ln = 8.52e-05  # Negative electrode thickness [m]
    const Lp = 7.56e-05  # Positive electrode thickness [m]
    const Ls = 12.0e-06   # Separator thickness [m]
    const Rn = 5.86e-06  # Negative particle Radius [m]
    const Rp = 5.22e-06  # Positive particle Radius [m]
    const A = 0.1        # Electrode surface area [m^2]
    const bn_0 = 3.84e5  # BOL Area to volume ratio of negative particles [m⁻¹]
    const bp_0 = 3.82e5  # BOL Area to volume ratio of positive particles [m⁻¹]
    const ϵn_0 = 0.25    # Initial porosity of the negative electrode [-]
    const ϵp_0 = 0.335   # Initial porosity of the positive electrode [-]


    # Maximum solid concentration
    const c_s_max_neg = 33133.0   # Maximum solid concentration in the negative electrode [mol/m^3]
    const c_s_max_pos = 63104.0  # Maximum solid concentration in the positive electrode [mol/m^3]

    # Diffusivity
    const Ds_n = 5.94e-15 # Diffusion coefficient of lithium in negative solid phase [m^2/s]
    const Ds_p = 4e-15    # Diffusion coefficient of lithium in positive solid phase [m^2/s]

    # SEI parameters
    const L_SEI0 = 5e-9
    const L_LIP0 = 0.0
    const L_SEI_cr0 = 5e-13


    #Thermal Parameters
    const h = 192.0 # Heat transfer coefficient [W/m²/K]
    const Vol = 2.43e-5 # Total volume of cell [m³]
    const Cooling_Area = 0.0053 # Cooling area [m²]
    const Cp = 2.85e6  # Volume specific heat capacity [J/m³/K]

    # Helper functions
    @inline function safe_sqrt(x)
        return  x > 0 ? sqrt(x) : -sqrt(-x)
    end

    # Exchange current density functions
    @inline function Negative_exchange_current_density(c_s_surf, T)
        T=298.15
        m_ref = 5.25e-6 # (A/m2)(m3/mol)^1.5 - includes ref concentrations
        E_r = 35000
        arrhenius = exp(E_r / R * (1 / T_ref - 1 / T))
        c_s_term = safe_sqrt(c_s_surf) * safe_sqrt(c_s_max_neg - c_s_surf) # the safesqrt is to ensure solver crashes when solver jump over the termination event
        return m_ref * arrhenius * sqrt(c_e) * c_s_term
    end

    @inline function Positive_exchange_current_density(c_s_surf, T)
        T=298.15
        m_ref =8.55e-7  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
        E_r = 17800
        arrhenius = exp(E_r / R * (1 / T_ref - 1 / T))
        c_s_term = safe_sqrt(c_s_surf*(c_s_max_pos - c_s_surf))
        return m_ref * arrhenius * sqrt(c_e) * c_s_term
    end

    # Open Circuit Potential functions
    @inline function Positive_OCP(surf_conc)
        sto = surf_conc / c_s_max_pos
        return -0.8090 * sto + 4.4875 - 
            0.0428 * tanh(18.5138 * (sto - 0.5542)) - 
            17.7326 * tanh(15.7890 * (sto - 0.3117)) + 
            17.5842 * tanh(15.9308 * (sto - 0.3120))
    end 

    @inline function Negative_OCP(surf_conc)
        sto = surf_conc / c_s_max_neg
        return 1.9793 * exp(-39.3631 * sto) + 0.2482 - 
            0.0909 * tanh(29.8538 * (sto - 0.1234)) - 
            0.04478 * tanh(14.9159 * (sto - 0.2769)) - 
            0.0205 * tanh(30.4444 * (sto - 0.6103))
    end 

end  # end module