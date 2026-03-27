using CairoMakie
using Colors

# --- 1. Physical Constants & Literature Priors ---
const R = 8.314             
const Ea = 91000.0          
const T_ref_K = 303.15      
const TSS_LIMIT = 300.0     

# --- 2. Physically-Coupled Decay Function ---
function get_k(temp, tss, strain, phase)
    if strain == "E11"
        β0, βT, βTSS = 0.05, 0.02765, 0.02137
        penalty = (phase == 1) ? 0.0375 : 0.0  
    else # CVB15
        β0, βT, βTSS = 0.03, 0.02000, 0.002
        penalty = (phase == 1) ? -0.015 : 0.0 
    end

    tss_eff = min(tss, TSS_LIMIT)
    tss_scaled = tss_eff / 100.0
    k_30 = (β0 + penalty) + (βT * 3.0) + (βTSS * tss_scaled)

    if temp <= 30.0
        t_scaled = temp / 10.0
        k = (β0 + penalty) + (βT * t_scaled) + (βTSS * tss_scaled)
    else
        t_k = temp + 273.15
        arrhenius_factor = exp((Ea / R) * (1/T_ref_K - 1/t_k))
        k = k_30 * arrhenius_factor
    end
    return max(k, 0.001) 
end

# --- 3. Grid Generation & Matrix Setup ---
temps = 0.0:0.5:50.0
tss_vals = 0.0:5.0:600.0
strains = ["E11", "CVB15", "Δ (E11 - CVB15)"]
phases = [0, 1] 
col_labels = ["Suspended Phase", "Attached Phase"]

fig = Figure(size = (1400, 1400), font = "Arial", fontsize = 20)

for (i, strain_label) in enumerate(strains)
    for (j, phase) in enumerate(phases)
        ax = Axis(fig[i, j], 
                  xlabel = i == 3 ? "Temperature (°C)" : "", 
                  ylabel = j == 1 ? "TSS (mg/L)" : "",
                  xticks = 0:10:50, yticks = 0:100:600)
        
        if i < 3
            # --- Rows 1 & 2: Absolute Decay Rates (Log Scaled) ---
            strain_name = (i == 1) ? "E11" : "CVB15"
            data = [get_k(t, s, strain_name, phase) for t in temps, s in tss_vals]
            
            hm = heatmap!(ax, temps, tss_vals, data, 
                          colorscale = log2, colormap = :roma, 
                          colorrange = (0.01, 1.0), highclip = :yellow)
            
            if j == 2
                Colorbar(fig[i, 3], hm, label = "Decay Rate (k)")
            end
        else
            # --- Row 3: Difference (Linear Diverging Scale) ---
            # Subtracting CVB15 from E11 to show excess decay
            data_diff = [(get_k(t, s, "E11", phase) - get_k(t, s, "CVB15", phase)) for t in temps, s in tss_vals]
            
            hm_diff = heatmap!(ax, temps, tss_vals, data_diff, 
                               colormap = :roma, 
                               colorrange = (0.01, 1), # Focus on the delta,
                               colorscale = log2,
                               lowclip = :blue, highclip = :red)
            
            if j == 2
                Colorbar(fig[i, 3], hm_diff, label = "Δk (E11 - CVB15)")
            end
        end

        # Universal Boundary Markers
        vlines!(ax, [30.0], color = :white, linestyle = :dash, linewidth = 2)
        hlines!(ax, [300.0], color = :white, linestyle = :dash, linewidth = 2)
    end
end

# --- 4. Outer Margin Labels ---
for (j, label) in enumerate(col_labels)
    Label(fig[0, j], label, font = :bold, padding = (0, 0, 10, 0),
    tellwidth = false, tellheight = true)
end

for (i, label) in enumerate(strains)
    Label(fig[i, 0], label, font = :bold, rotation = pi/2, padding = (0, 25, 0, 0),
    tellwidth = true, tellheight = false)
end

fig

save("decay_heatmap_matrix_low.png", fig)