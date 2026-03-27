using CairoMakie
using Colors

# --- 1. Empirical Coefficients ---
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

# --- 2. Partitioning Physics ---
# Kp for enteric viruses (e.g., Adenovirus/Enterovirus) ~ 10^4 L/kg
const K_p = 10000.0 

function get_fractions(tss_mg_L)
    # Convert TSS to kg/L (1 mg/L = 1e-6 kg/L)
    tss_kg_L = tss_mg_L * 1e-6
    
    f_solid = (K_p * tss_kg_L) / (1.0 + (K_p * tss_kg_L))
    f_liquid = 1.0 - f_solid
    return f_liquid, f_solid
end

# --- 3. Base Empirical Function ---
function get_k_exact(temp, tss, is_e11, is_liquid)
    Strain = is_e11 ? 1.0 : 0.0  
    Phase = is_liquid ? 1.0 : 0.0 
    T_s, TSS_s = temp / 10.0, tss / 100.0
    
    k = c_int + (c_org * Strain) + (c_phase * Phase) + (c_temp * T_s) + 
        (c_tss * TSS_s) + (c_pxs * Phase * Strain) + (c_pxtss * Phase * TSS_s) + 
        (c_txtss * T_s * TSS_s) + (c_txs * T_s * Strain) + (c_pxt * Phase * T_s) + 
        (c_sxtss * Strain * TSS_s)
    return max(k, 0.005) 
end

# --- 4. Effective Plume Decay (The Hydrodynamic Shift) ---
function get_k_effective(temp, tss, is_e11)
    # Calculate phase-specific k
    k_liq = get_k_exact(temp, tss, is_e11, true)
    k_sol = get_k_exact(temp, tss, is_e11, false)
    
    # Calculate mass fractions based on TSS
    f_liq, f_sol = get_fractions(tss)
    
    # Weighted average decay rate
    return (f_liq * k_liq) + (f_sol * k_sol)
end

get_thalf_eff(temp, tss, is_e11) = log(2) / get_k_effective(temp, tss, is_e11)

# --- 5. Plotting ---
temps = 4.0:0.25:30.0      
tss_vals = 0.0:2.5:300.0   

fig = Figure(size = (1000, 500), font = "Arial", fontsize = 18)

# Panel 1: E11 Plume
ax1 = Axis(fig[1, 1], title = "E11: Effective Plume Half-Life", 
           xlabel = "Temperature (°C)", ylabel = "TSS (mg/L)")
data_e11 = [get_thalf_eff(t, s, true) for t in temps, s in tss_vals]
hm1 = heatmap!(
    ax1,
    temps,
    tss_vals,
    data_e11,
    colorscale = log10, 
    colormap = :magma,
    colorrange = (10, 120.0),
    lowclip=:black,
    highclip=:yellow
)

# Panel 2: CVB5 Plume
ax2 = Axis(fig[1, 2], title = "CVB5: Effective Plume Half-Life", 
           xlabel = "Temperature (°C)")
data_cvb5 = [get_thalf_eff(t, s, false) for t in temps, s in tss_vals]

hm2 = heatmap!(
    ax2,
    temps,
    tss_vals,
    data_cvb5,
    colorscale = log10, 
    colormap = :magma,
    colorrange = (10, 120.0),
    lowclip=:black,
    highclip=:yellow
)

Colorbar(fig[1, 3], hm1, label = "Plume Half-Life (Days)", 
         ticks = ([2, 5, 10, 20, 30], ["2", "5", "10", "20", "30+"]))

fig