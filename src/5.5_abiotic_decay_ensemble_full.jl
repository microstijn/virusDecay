# ==============================================================================
# SCRIPT 7: ABIOTIC DECAY (K_ABIOTIC) MASTER DREDGE
# Purpose: Programmatically test biologically legal combinations of predictors
#          using BOTH Linear and Exponential math on the Abiotic System (N=36).
#          Includes Biophysical Filters, Multimodel Averaging, & CairoMakie Plots.
# ==============================================================================

using DataFrames
using CSV
using GLM
using LsqFit
using StatsModels
using StatsBase
using Printf
using Statistics
using Distributions
using CairoMakie

println("Starting Script 7: Abiotic K Mega-Dredge (Biophysically-Filtered)...")

# ==========================================
# 1. LOAD & PREPARE ABIOTIC DECAY DATA (N=36)
# ==========================================
data_dir = "C:/Users/peete074/Downloads/previr"
input_filepath = joinpath(data_dir, "CLEANED_FINAL_DATASET.csv")
dat_full = CSV.read(input_filepath, DataFrame)

# Isolate Solid Phase (Abiotic K)
df_solid = select(dropmissing(dat_full, [:kSolid_Abiotic, :kSolid_Abiotic_std]),
    :Experiment, :Temperature, :TSS, :Organism, 
    :kSolid_Abiotic => :k_abiotic, :kSolid_Abiotic_std => :SE)
df_solid.Phase .= "Solid"

# Isolate Liquid Phase (Abiotic K)
df_liquid = select(dropmissing(dat_full, [:kLiquid_Abiotic, :kLiquid_Abiotic_std]),
    :Experiment, :Temperature, :TSS, :Organism, 
    :kLiquid_Abiotic => :k_abiotic, :kLiquid_Abiotic_std => :SE)
df_liquid.Phase .= "Liquid"

# Combine into master dataset
dat = vcat(df_solid, df_liquid)
n_obs = nrow(dat)

println("✅ Loaded $(n_obs) combined conditions for Abiotic Decay.")

# ==========================================
# 2. CALCULATE WEIGHTS & SCALED PREDICTORS
# ==========================================
dat.Wt = 1.0 ./ (dat.SE .^ 2)
dat.Wt_Norm = dat.Wt ./ mean(dat.Wt)

dat.Org_Numeric = [o == "E11" ? 1.0 : 0.0 for o in dat.Organism]
dat.Phase_Numeric = [p == "Liquid" ? 1.0 : 0.0 for p in dat.Phase]

dat.Temp_Scaled = dat.Temperature ./ 10.0
dat.TSS_Scaled = dat.TSS ./ 100.0

dat.Temp_x_TSS = dat.Temp_Scaled .* dat.TSS_Scaled
dat.Strain_x_TSS = dat.Org_Numeric .* dat.TSS_Scaled
dat.Temp_x_Strain = dat.Temp_Scaled .* dat.Org_Numeric
dat.Phase_x_Temp = dat.Phase_Numeric .* dat.Temp_Scaled
dat.Phase_x_TSS = dat.Phase_Numeric .* dat.TSS_Scaled
dat.Phase_x_Strain = dat.Phase_Numeric .* dat.Org_Numeric

# ==========================================
# 3. GENERATE FILTERED COMBINATIONS
# ==========================================
predictors = [:Temp_Scaled, :TSS_Scaled, :Org_Numeric, :Phase_Numeric, 
              :Temp_x_TSS, :Strain_x_TSS, :Temp_x_Strain, 
              :Phase_x_Temp, :Phase_x_TSS, :Phase_x_Strain]

function get_combinations(arr)
    n = length(arr)
    combos = Vector{Symbol}[]
    
    # Define interactions for the "Frankensalad" filter
    interactions = [:Temp_x_TSS, :Strain_x_TSS, :Temp_x_Strain, 
                    :Phase_x_Temp, :Phase_x_TSS, :Phase_x_Strain]

    for i in 0:(2^n - 1)
        idx = digits(i, base=2, pad=n)
        combo = arr[findall(==(1), idx)]
        
        valid = true
        
        # --- 1. THE MARGINALITY BOUNCER (Mathematical Filter) ---
        if :Temp_x_TSS in combo && (!(:Temp_Scaled in combo) || !(:TSS_Scaled in combo)); valid = false; end
        if :Strain_x_TSS in combo && (!(:Org_Numeric in combo) || !(:TSS_Scaled in combo)); valid = false; end
        if :Temp_x_Strain in combo && (!(:Temp_Scaled in combo) || !(:Org_Numeric in combo)); valid = false; end
        if :Phase_x_Temp in combo && (!(:Phase_Numeric in combo) || !(:Temp_Scaled in combo)); valid = false; end
        if :Phase_x_TSS in combo && (!(:Phase_Numeric in combo) || !(:TSS_Scaled in combo)); valid = false; end
        if :Phase_x_Strain in combo && (!(:Phase_Numeric in combo) || !(:Org_Numeric in combo)); valid = false; end
        
        # --- 2. THE BIOPHYSICAL BOUNCER (A Priori Knowledge) ---
        # We allow the completely empty combo (Null Model) to pass because it 
        # is mathematically required as a baseline for AICc comparison.
        if !isempty(combo) 
            
            # A. Thermodynamic Mandate: Any mechanistic model must include Temperature
            if !(:Temp_Scaled in combo)
                valid = false
            end
            
            # B. Turbidity Paradox: TSS physically requires a phase context
            if :TSS_Scaled in combo && !(:Phase_Numeric in combo)
                valid = false
            end
            
            # C. Frankensalad Filter: Limit to a max of 3 simultaneous interactions 
            # to prevent biophysically unexplainable "math salads". 
            # (Note: Your winning K=9 model had exactly 2 or 3 interactions, so this keeps it safe!)
            num_interactions = sum([1 for x in combo if x in interactions])
            if num_interactions > 3
                valid = false
            end
        end
        
        if valid
            push!(combos, combo)
        end
    end
    return combos
end

valid_combos = get_combinations(predictors)
println("✅ Filters applied: $(length(valid_combos)) biologically legal structures generated.")

# ==========================================
# 4. FIT ALL MODELS (LINEAR & EXPONENTIAL)
# ==========================================
resp = Term(:k_abiotic)
exp_func(X_matrix, p) = exp.(X_matrix * p)

results = DataFrame(Model = String[], Type = String[], K = Int[], AICc = Float64[], Status = String[], Combo = Vector{Symbol}[])
valid_models_cache = Dict{String, Any}()

println("Fitting models...")

for combo in valid_combos
    if isempty(combo)
        f = FormulaTerm(resp, ConstantTerm(1))
        base_name = "Null"
    else
        f = FormulaTerm(resp, sum(Term.(combo)))
        base_name = join(string.(combo), " + ")
    end
    
    # Linear
    try
        mod_lin = lm(f, dat, wts=dat.Wt_Norm)
        k_lin = length(coef(mod_lin)) + 1
        ll_lin = loglikelihood(mod_lin)
        aicc_lin = (-2 * ll_lin + 2 * k_lin) + (2 * k_lin * (k_lin + 1)) / (n_obs - k_lin - 1)
        push!(results, (base_name, "Linear", k_lin, aicc_lin, "Passed", combo))
        valid_models_cache["Linear_" * base_name] = mod_lin
    catch
    end
    
    # Exponential
    try
        f_schema = apply_schema(f, schema(dat))
        y_data, X_data = modelcols(f_schema, dat)
        y_mean = mean(y_data)
        init_val = y_mean > 0 ? log(y_mean) : -3.0
        p0 = fill(0.01, size(X_data, 2))
        p0[1] = init_val
        
        fit_exp = curve_fit((x,p) -> exp_func(x,p), X_data, y_data, dat.Wt_Norm, p0)
        
        if fit_exp.converged
            se = margin_error(fit_exp, 0.05)
            if !any(se .> 50.0)
                rss = sum(dat.Wt_Norm .* fit_exp.resid.^2)
                ll_exp = -0.5 * n_obs * log(2 * pi) - 0.5 * n_obs * log(rss / n_obs) - 0.5 * n_obs
                k_exp = length(fit_exp.param) + 1
                aicc_exp = (-2 * ll_exp + 2 * k_exp) + (2 * k_exp * (k_exp + 1)) / (n_obs - k_exp - 1)
                push!(results, (base_name, "Exponential", k_exp, aicc_exp, "Passed", combo))
                valid_models_cache["Exponential_" * base_name] = fit_exp
            end
        end
    catch
    end
end

# ==========================================
# 5. CALCULATE AICc WEIGHTS & RVI
# ==========================================
valid_results = filter(row -> row.AICc != Inf, results)
sort!(valid_results, :AICc)

min_aicc = minimum(valid_results.AICc)
valid_results.Delta_AICc = valid_results.AICc .- min_aicc
valid_results.Rel_Likelihood = exp.(-0.5 .* valid_results.Delta_AICc)
valid_results.Weight = valid_results.Rel_Likelihood ./ sum(valid_results.Rel_Likelihood)

# ==========================================
# 8. MULTIMODEL AVERAGING (ZERO-METHOD)
# ==========================================
# Filter to confidence set (Delta AICc <= 4)
conf_set = filter(r -> r.Delta_AICc <= 4.0, valid_results)
conf_set.W_renorm = conf_set.Rel_Likelihood ./ sum(conf_set.Rel_Likelihood)

all_params = vcat([:Intercept], predictors)
avg_results = DataFrame(Parameter=String[], Coef=Float64[], Uncond_SE=Float64[], Z=Float64[], P_val=Float64[])

for p in all_params
    p_str = p == :Intercept ? "(Intercept)" : string(p)
    theta_bar = 0.0
    var_uncond = 0.0
    
    # Pass 1: Calculate averaged coefficient
    for row in eachrow(conf_set)
        mod_obj = valid_models_cache[row.Type * "_" * row.Model]
        cnames = row.Type == "Linear" ? coefnames(mod_obj) : (isempty(row.Combo) ? ["(Intercept)"] : vcat(["(Intercept)"], string.(row.Combo)))
        cvals = row.Type == "Linear" ? coef(mod_obj) : mod_obj.param
        
        idx = findfirst(==(p_str), cnames)
        theta_i = idx !== nothing ? cvals[idx] : 0.0
        theta_bar += row.W_renorm * theta_i
    end
    
    # Pass 2: Calculate unconditional variance
    for row in eachrow(conf_set)
        mod_obj = valid_models_cache[row.Type * "_" * row.Model]
        cnames = row.Type == "Linear" ? coefnames(mod_obj) : (isempty(row.Combo) ? ["(Intercept)"] : vcat(["(Intercept)"], string.(row.Combo)))
        cvals = row.Type == "Linear" ? coef(mod_obj) : mod_obj.param
        ses = row.Type == "Linear" ? stderror(mod_obj) : margin_error(mod_obj, 0.05)
        
        idx = findfirst(==(p_str), cnames)
        theta_i = idx !== nothing ? cvals[idx] : 0.0
        se_i = idx !== nothing ? ses[idx] : 0.0
        
        var_uncond += row.W_renorm * (se_i^2 + (theta_i - theta_bar)^2)
    end
    
    se_final = sqrt(var_uncond)
    z_val = se_final > 0 ? theta_bar / se_final : NaN
    p_val = !isnan(z_val) ? 2 * (1 - cdf(Normal(), abs(z_val))) : NaN
    
    push!(avg_results, (p_str, theta_bar, se_final, z_val, p_val))
end

# Clean up table by dropping parameters that completely zeroed out
avg_results = filter(r -> r.Coef != 0.0, avg_results)

sort!(avg_results, :P_val)

@printf("%-18s | %-10s | %-12s | %-8s | %-10s\n", "Parameter", "Avg Coef", "Uncond. SE", "Z-value", "P-value")
println("-"^95)
for row in eachrow(avg_results)
    p_str = row.P_val < 0.001 ? "<0.001" : @sprintf("%.4f", row.P_val)
    @printf("%-18s | %10.5f | %12.5f | %8.2f | %-10s\n", 
        row.Parameter, row.Coef, row.Uncond_SE, row.Z, p_str)
end
println("="^95)




# ==========================================
# 9. PUBLICATION PLOT (CAIROMAKIE)
# ==========================================
function show_off_model(avg_results, valid_results, valid_models_cache, dat)
    fig = Figure(size = (800, 350), font = "Arial")
    
    # --- PANEL A: FOREST PLOT ---
    ax1 = Axis(
        fig[1, 1],
        title = "A: Standardized effect sizes (Model Average)",
        yticks = (1:(nrow(avg_results)-1), reverse(filter(r -> r.Parameter != "(Intercept)", avg_results).Parameter)),
        xlabel = "Coefficient value (standardized)"
    )
    
    forest_data = filter(r -> r.Parameter != "(Intercept)", avg_results)
    y_points = reverse(1:nrow(forest_data)) # Reverse so highest effect is at top
    
    vlines!(ax1, 0, color = :black, linewidth = 2, linestyle = :dash)
    
    errorbars!(ax1, forest_data.Coef, y_points, forest_data.Uncond_SE, direction = :x, color = :black, whiskerwidth = 8)
    scatter!(ax1, forest_data.Coef, y_points, strokewidth = 1, markersize = 12, color = :black)
    
    # --- PANEL B: OBSERVED VS PREDICTED ---
    ax2 = Axis(fig[1, 2], title = "B: Model accuracy (Top Ranked Model)",
               xlabel = "Predicted abiotic decay (k)",
               ylabel = "Observed abiotic decay (k)")
    
    # Dynamically fetch top model predictions
    top_key = valid_results[1, :Type] * "_" * valid_results[1, :Model]
    top_model = valid_models_cache[top_key]
    preds = predict(top_model)
    obs = dat.k_abiotic
    
    line_range = [minimum(vcat(obs, preds)), maximum(vcat(obs, preds))]
    lines!(ax2, line_range, line_range, color = :black, linestyle = :dot)
    
    colors = [r.Org_Numeric == 1.0 ? :pink : :orangered for r in eachrow(dat)]
    scatter!(ax2, preds, obs, color = colors, markersize = 10, strokewidth = 1, alpha = 0.7)
    
    elem_1 = [MarkerElement(color = :pink, marker = :circle, markersize = 10)]
    elem_2 = [MarkerElement(color = :orangered, marker = :circle, markersize = 10)]
    Legend(fig[1, 2], [elem_1, elem_2], ["CVB5", "E11"], tellheight = false, tellwidth = false, halign = :right, valign = :bottom)

    display(fig)
    save("Abiotic_Model_Showoff.png", fig)
end

show_off_model(avg_results, valid_results, valid_models_cache, dat)
println("\n✅ Plot saved as 'Abiotic_Model_Showoff.png'")