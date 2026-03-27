using CairoMakie
using Colors

# --- 1. Thermodynamic & Physical Constants ---
const R = 8.314             # Gas Constant (J/mol·K)
const Ea = 91000.0          # Activation Energy (J/mol)
const T_ref_K = 303.15      # 30°C Reference (Kelvin)
const TSS_LIMIT = 300.0     # Empirical Boundary Cap (mg/L)

# --- 2. Exact Coefficients from Table 4 ---
const c_int   = -0.02216
const c_org   =  0.03754
const c_phase = -0.00847
const c_temp  =  0.02765
const c_tss   =  0.00068
const c_pxs   = -0.03259
const c_pxtss =  0.02137
const c_txtss = -0.00461
const c_txs   = -0.00056
const c_pxt   =  0.00015
const c_sxtss = -0.00006

# --- 3. Full Domain Coupled Function ---
function get_k_exact_extended(temp, tss, is_e11, is_liquid)
    Strain = is_e11 ? 1.0 : 0.0  
    Phase = is_liquid ? 1.0 : 0.0 
    
    # Apply Empirical TSS Boundary 
    # (Prevents non-physical polynomial runaway at extreme TSS)
    tss_eff = min(tss, TSS_LIMIT)
    TSS_s = tss_eff / 100.0
    
    # Inner closure to calculate the exact linear 11-term state
    function calc_linear(T_s_val)
        return c_int + (c_org * Strain) + (c_phase * Phase) + 
               (c_temp * T_s_val) + (c_tss * TSS_s) + 
               (c_pxs * Phase * Strain) + (c_pxtss * Phase * TSS_s) + 
               (c_txtss * T_s_val * TSS_s) + (c_txs * T_s_val * Strain) + 
               (c_pxt * Phase * T_s_val) + (c_sxtss * Strain * TSS_s)
    end

    # Calculate the specific anchor state at exactly 30°C
    k_30 = calc_linear(3.0)

    if temp <= 30.0
        # Validated Experimental Regime
        k = calc_linear(temp / 10.0)
    else
        # Arrhenius Thermodynamic Extension (> 30°C)
        t_k = temp + 273.15
        arrhenius_factor = exp((Ea / R) * (1/T_ref_K - 1/t_k))
        k = k_30 * arrhenius_factor
    end
    
    return max(k, 0.005) # Floor at 0.005 to cap max half-life at ~138 days
end

# Convert to Half-Life (Days)
get_thalf(temp, tss, is_e11, is_liquid) = log(2) / get_k_exact_extended(temp, tss, is_e11, is_liquid)

# --- 4. Macro-Scale Grid & Plotting Setup ---
temps = 0.0:0.5:50.0      
tss_vals = 0.0:5.0:600.0   
strains = ["E11", "CVB5", "Δ T½ (CVB5 - E11)"] 
phase_flags = [true, false] # true = Suspended (Liquid), false = Attached (Solid)
col_labels = ["Suspended Phase (Liquid)", "Attached Phase (Solid)"]

fig = Figure(size = (1400, 1400), font = "Arial", fontsize = 20)

for (i, strain_label) in enumerate(strains)
    for (j, is_liquid) in enumerate(phase_flags)
        ax = Axis(fig[i, j], 
                  xlabel = i == 3 ? "Temperature (°C)" : "", 
                  ylabel = j == 1 ? "TSS (mg/L)" : "",
                  xticks = 0:10:50, yticks = 0:100:600) 
        
        if i < 3
            # --- Rows 1 & 2: Absolute Half-Life ---
            is_e11 = (i == 1)
            data = [get_thalf(t, s, is_e11, is_liquid) for t in temps, s in tss_vals]
            
            # Bright/Yellow = Long Survival (50+ days), Black = Fast Death (<12 hours)
            hm = heatmap!(ax, temps, tss_vals, data, 
                          colorscale = log10, 
                          colormap = :glasgow, 
                          colorrange = (2.0, 140.0), 
                          lowclip = :yellow,
                          highclip = :green
                        )

            if j == 2
                Colorbar(fig[i, 3], hm, label = "Half-Life (Days)", 
                         ticks = ([0.5, 1, 5, 10, 50], ["0.5", "1", "5", "10", "50+"]))
            end
        else
            # --- Row 3: Survival Deficit (CVB5 - E11) ---
            # Yields POSITIVE days of survival that E11 loses.
            data_diff = [(get_thalf(t, s, false, is_liquid) - get_thalf(t, s, true, is_liquid)) for t in temps, s in tss_vals]
            
            # Red = E11 dies faster (Large deficit in days)
            hm_diff = heatmap!(ax, temps, tss_vals, data_diff, 
                               colormap = :glasgow,
                               #colorrange = (0.001, 160.0), 
                               #colorscale = log10,
                               lowclip = :yellow,
                               highclip = :green
            )
            
            if j == 2
                Colorbar(fig[i, 3], hm_diff, label = "Survival Deficit (Days Lost by E11)", 
                         ticks = ([0, 10, 20, 30], ["0", "10", "20", "30+"]))
            end
        end

        # Universal Boundary Safety Rails (Dashed Lines)
        # White for the dark magma plots, black for the brighter RdBu difference plot
        line_col = i < 3 ? :white : :black
        vlines!(ax, [30.0], color = line_col, linestyle = :dash, linewidth = 2)
        hlines!(ax, [300.0], color = line_col, linestyle = :dash, linewidth = 2)
    end
end

# --- 5. Outer Margin Labels ---
for (j, label) in enumerate(col_labels)
    Label(fig[0, j], label, font = :bold, padding = (0, 0, 10, 0), tellwidth = false, tellheight = true)
end

for (i, label) in enumerate(strains)
    Label(fig[i, 0], label, font = :bold, rotation = pi/2, padding = (0, 20, 0, 0), tellwidth = true, tellheight = false)
end

fig

save("virus_half_life__expansion_heatmaps.png", fig)