using CairoMakie
using Colors

# --- 1. Exact Coefficients from Table 4 (Delta AICc <= 4) ---
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

# --- 2. Master Model Function ---
function get_k_exact(temp, tss, is_e11, is_liquid)
    # Map booleans to your numeric dummy structure
    Strain = is_e11 ? 1.0 : 0.0  # E11 = 1.0, CVB5 = 0.0
    Phase = is_liquid ? 1.0 : 0.0 # Liquid = 1.0, Solid = 0.0
    
    T_s = temp / 10.0
    TSS_s = tss / 100.0
    
    # The exact 11-term multimodel average
    k = c_int + 
        (c_org * Strain) + 
        (c_phase * Phase) + 
        (c_temp * T_s) + 
        (c_tss * TSS_s) + 
        (c_pxs * Phase * Strain) + 
        (c_pxtss * Phase * TSS_s) + 
        (c_txtss * T_s * TSS_s) + 
        (c_txs * T_s * Strain) + 
        (c_pxt * Phase * T_s) + 
        (c_sxtss * Strain * TSS_s)
        
    return max(k, 0.005) # Floor at 0.005 to prevent infinite half-lives
end

# Half-Life Conversion
get_thalf(temp, tss, is_e11, is_liquid) = log(2) / get_k_exact(temp, tss, is_e11, is_liquid)

# --- 3. Grid Generation & Matrix Setup ---
temps = 4.0:0.25:30.0      
tss_vals = 0.0:2.5:300.0   
strains = ["E11", "CVB5", "Δ T½ (CVB5 - E11)"] # Updated to CVB5
phase_flags = [true, false] # true = Suspended (Liquid), false = Attached (Solid)
col_labels = ["Suspended Phase (Liquid)", "Attached Phase (Solid)"]

fig = Figure(size = (1400, 1400), font = "Arial", fontsize = 20)

for (i, strain_label) in enumerate(strains)
    for (j, is_liquid) in enumerate(phase_flags)
        ax = Axis(fig[i, j], 
                  xlabel = i == 3 ? "Temperature (°C)" : "", 
                  ylabel = j == 1 ? "TSS (mg/L)" : "",
                  xticks = 5:5:30, yticks = 0:50:300) 
        
        if i < 3
            # --- Rows 1 & 2: Absolute Half-Life (Log Scaled) ---
            is_e11 = (i == 1) # Row 1 is E11, Row 2 is CVB5
            data = [get_thalf(t, s, is_e11, is_liquid) for t in temps, s in tss_vals]
            
            hm = heatmap!(ax, temps, tss_vals, data, 
                          colorscale = log10, 
                          colormap = :glasgow, 
                          colorrange = (7.0, 140.0), 
                          lowclip = :yellow,
                          highclip = :green
                        )

            if j == 2
                Colorbar(fig[i, 3], hm, label = "Half-Life (Days)", 
                         ticks = ([5, 10, 20, 40, 80, 140], ["5", "10", "20", "40", "80", "140+"]))
            end
        else
            # --- Row 3: Survival Deficit (CVB5 - E11) ---
            # Yields POSITIVE days of survival that E11 misses out on
            data_diff = [(get_thalf(t, s, false, is_liquid) - get_thalf(t, s, true, is_liquid)) for t in temps, s in tss_vals]
            
            # 0 is Blue (No difference), High positive is Red (Large deficit)
            hm_diff = heatmap!(ax, temps, tss_vals, data_diff, 
                               colormap = :glasgow,
                               #colorrange = (0.001, 160.0), 
                               #colorscale = log10,
                               lowclip = :yellow,
                               highclip = :green
                    )
            
            if j == 2
                Colorbar(fig[i, 3], hm_diff, label = "Survival Deficit (Days Lost by E11)")
            end
        end
    end
end



# --- 4. Outer Margin Labels ---
for (j, label) in enumerate(col_labels)
    Label(fig[0, j], label, font = :bold, padding = (0, 0, 10, 0), tellwidth = false, tellheight = true)
end

for (i, label) in enumerate(strains)
    Label(fig[i, 0], label, font = :bold, rotation = pi/2, padding = (0, 20, 0, 0), tellwidth = true, tellheight = false)
end

fig

save("virus_half_life_heatmaps.png", fig)