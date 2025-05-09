module SPM_Model_Aux

    using SparseArrays, LinearAlgebra
    import ..Para_set

    export neg_particle_diffusion_derivative!, pos_particle_diffusion_derivative!, SPM_voltage

    const F = Para_set.F
    const R = Para_set.R
    const A = Para_set.A
    const Ln = Para_set.Ln
    const Lp = Para_set.Lp
    const bn_0 = Para_set.bn_0
    const ϵn_0 = Para_set.ϵn_0


    # area to volume ratio of outermost control volume (10-th shell)
    const  A_BY_V_pos = 1.3641530882910194e6
    const A_BY_V_neg = 1.2151670854742527e6



    # FVM discretization with 6 control volumes
    const PARTICLE_DIFFUSION_MATRIX = Tridiagonal(
        [15.428571428571429, 22.736842105263158, 26.27027027027027, 28.327868852459016, 29.67032967032967],
        [-108.0, -77.14285714285714, -73.89473684210526, -72.97297297297297, -72.59016393442623, -29.67032967032967],
        [108.0, 61.714285714285715, 51.1578947368421, 46.7027027027027, 44.26229508196721]
    )

    function pos_particle_diffusion_derivative!(dc, c, N_surf, T)

        T_ref = 298.15
        D_T_corr = Para_set.Ds_p * exp(-2490/8.3*(1/T - 1/T_ref))  #note that no Ea/R
        mul!(dc, (D_T_corr / Para_set.Rp^2) .* PARTICLE_DIFFUSION_MATRIX, c)
        dc[end] -= N_surf * A_BY_V_pos

        nothing
    end

    function neg_particle_diffusion_derivative!(dc, c, N_surf, T)

        T_ref = 298.15
        D_T_corr = Para_set.Ds_n * exp(-1245.0/8.3* (1/T - 1/T_ref))
        mul!(dc, (D_T_corr / Para_set.Rn^2) .* PARTICLE_DIFFUSION_MATRIX, c)
        dc[end] -= N_surf * A_BY_V_neg

        nothing

    end

   

    @inline function FVM_vol_avg(conc)

        #sum([i^3 - (i-1)^3 for i in 1:6]./6^3 ⋅* conc)
        sum([1,7,19,37,61,91] ./ 216.0 .* conc)

    end 

    

    κ_SEI = 3.75e-6

    function SPM_voltage(c_n_surf, c_p_surf, bn, bp, L_SEI,L_LIP, I, T)

    

        OCP = Para_set.Positive_OCP(c_p_surf) - Para_set.Negative_OCP(c_n_surf)

        j_n = I / (A * Ln) / bn
        j_n_0 =Para_set.Negative_exchange_current_density(c_n_surf, T)
        j_p = -I / (A * Lp) / bp
        j_p_0 = Para_set.Positive_exchange_current_density(c_p_surf, T)
        Φ_reaction = (2 * R * T / F )* (asinh(j_p / j_p_0) - asinh(j_n / j_n_0)) 

        R0 =0.00122
        R0_SEI = L_SEI/(Para_set.A*Para_set.Ln*bn*κ_SEI)
        
        V = OCP - I * R0 - I * R0_SEI + Φ_reaction 

        return V
    end 


    

    function Resistance_calc(c_n_surf, c_p_surf, bn, bp, L_SEI,L_LIP, I, T)

       
        j_n = I / (A * Ln) / bn
        j_n_0 =Para_set.Negative_exchange_current_density(c_n_surf, T)
        j_p = -I / (A * Lp) / bp
        j_p_0 = Para_set.Positive_exchange_current_density(c_p_surf, T)
        Φ_reaction = (2 * R * T / F )* (asinh(j_p / j_p_0) - asinh(j_n / j_n_0)) 

        R0 = 0.00122
        R0_SEI = L_SEI/(Para_set.A*Para_set.Ln*bn*κ_SEI)  

        return  R0 + R0_SEI + abs(Φ_reaction/I)
    end 

end
