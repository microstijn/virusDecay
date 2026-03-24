# ==============================================================================
# SCRIPT 5: MICROBIAL PHASE MASTER DREDGE (LINEAR + EXPONENTIAL)
# Purpose: Programmatically test all 1024 combinations using BOTH Linear 
#          and Exponential (Arrhenius) math. Total arena = 2048 models.
# ==============================================================================

using DataFrames
using CSV
using GLM
using LsqFit
using StatsModels
using StatsBase
using Printf
using Statistics

println("Starting Script 5: Mega Linear vs Exponential Dredge (2048 Models)...")

# ==========================================
# 1. LOAD & PREPARE DATA (N=36)
# ==========================================
data_dir = "C:/Users/peete074/Downloads/previr"
input_filepath = joinpath(data_dir, "CLEANED_FINAL_DATASET.csv")
dat_full = CSV.read(input_filepath, DataFrame)

df_solid = select(dropmissing(dat_full, [:kSolid_Microbial, :kSolid_Microbial_std]),
    :Experiment, :Temperature, :TSS, :Organism, 
    :kSolid_Microbial => :k_microbial, :kSolid_Microbial_std => :SE)
df_solid.Phase .= "Solid"

df_liquid = select(dropmissing(dat_full, [:kLiquid_Microbial, :kLiquid_Microbial_std]),
    :Experiment, :Temperature, :TSS, :Organism, 
    :kLiquid_Microbial => :k_microbial, :kLiquid_Microbial_std => :SE)
df_liquid.Phase .= "Liquid"

dat = vcat(df_solid, df_liquid)
n_obs = nrow(dat)

# Weights
dat.Wt = 1.0 ./ (dat.SE .^ 2)
dat.Wt_Norm = dat.Wt ./ mean(dat.Wt)

# Numeric Predictors for Dredge
dat.Org_Numeric = [o == "E11" ? 1.0 : 0.0 for o in dat.Organism]
dat.Phase_Numeric = [p == "Liquid" ? 1.0 : 0.0 for p in dat.Phase] # Liquid = 1, Solid = 0

# Scale continuous variables to help LsqFit stability
dat.Temp_Scaled = dat.Temperature ./ 10.0
dat.TSS_Scaled = dat.TSS ./ 100.0

# Interactions
dat.Temp_x_TSS = dat.Temp_Scaled .* dat.TSS_Scaled
dat.Strain_x_TSS = dat.Org_Numeric .* dat.TSS_Scaled
dat.Temp_x_Strain = dat.Temp_Scaled .* dat.Org_Numeric
dat.Phase_x_Temp = dat.Phase_Numeric .* dat.Temp_Scaled
dat.Phase_x_TSS = dat.Phase_Numeric .* dat.TSS_Scaled
dat.Phase_x_Strain = dat.Phase_Numeric .* dat.Org_Numeric

# ==========================================
# 2. GENERATE FORMULAS
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

all_combos = get_combinations(predictors)
resp = Term(:k_microbial)

# LsqFit exponential function helper
exp_func(X_matrix, p) = exp.(X_matrix * p)

results = DataFrame(Model = String[], Type = String[], K = Int[], AICc = Float64[], Status = String[], Combo = Vector{Symbol}[])
valid_models_cache = Dict{String, Any}() # Store winning models to print parameters later

println("Testing 2048 configurations. This will take a few seconds...")

# ==========================================
# 3. FIT ALL MODELS (LINEAR & EXPONENTIAL)
# ==========================================
for combo in all_combos
    # Define Formula
    if isempty(combo)
        f = FormulaTerm(resp, ConstantTerm(1))
        base_name = "Null"
    else
        f = FormulaTerm(resp, sum(Term.(combo)))
        base_name = join(string.(combo), " + ")
    end
    
    # --- A. FIT LINEAR ---
    try
        mod_lin = lm(f, dat, wts=dat.Wt_Norm)
        k_lin = length(coef(mod_lin)) + 1
        ll_lin = loglikelihood(mod_lin)
        aicc_lin = (-2 * ll_lin + 2 * k_lin) + (2 * k_lin * (k_lin + 1)) / (n_obs - k_lin - 1)
        
        push!(results, (base_name, "Linear", k_lin, aicc_lin, "✅ Passed", combo))
        valid_models_cache["Linear_" * base_name] = mod_lin
    catch
        push!(results, (base_name, "Linear", 0, Inf, "❌ Singular Matrix", combo))
    end
    
    # --- B. FIT EXPONENTIAL ---
# --- B. FIT EXPONENTIAL ---
    try
        f_schema = apply_schema(f, schema(dat))
        y_data, X_data = modelcols(f_schema, dat)
        
        y_mean = mean(y_data)
        init_val = y_mean > 0 ? log(y_mean) : -3.0
        p0 = fill(0.01, size(X_data, 2))
        p0[1] = init_val
        
        # FIXED: Removed the transpose (') on X_data
        fit_exp = curve_fit((x,p) -> exp_func(x,p), X_data, y_data, dat.Wt_Norm, p0)
        
        if fit_exp.converged
            se = margin_error(fit_exp, 0.05)
            if any(se .> 50.0)
                push!(results, (base_name, "Exponential", 0, Inf, "SE > 50", combo))
            else
                rss = sum(dat.Wt_Norm .* fit_exp.resid.^2)
                ll_exp = -0.5 * n_obs * log(2 * pi) - 0.5 * n_obs * log(rss / n_obs) - 0.5 * n_obs
                k_exp = length(fit_exp.param) + 1
                aicc_exp = (-2 * ll_exp + 2 * k_exp) + (2 * k_exp * (k_exp + 1)) / (n_obs - k_exp - 1)
                
                push!(results, (base_name, "Exponential", k_exp, aicc_exp, "Passed", combo))
                valid_models_cache["Exponential_" * base_name] = fit_exp
            end
        else
            push!(results, (base_name, "Exponential", 0, Inf, "No Converge", combo))
        end
    catch e
        # Changed to print the actual error if it crashes again, so we aren't blind!
        push!(results, (base_name, "Exponential", 0, Inf, "Error: $(typeof(e))", combo))
    end
end

# ==========================================
# 4. CALCULATE AICc WEIGHTS & RVI
# ==========================================
valid_results = filter(row -> row.AICc != Inf, results)
sort!(valid_results, :AICc)

min_aicc = minimum(valid_results.AICc)
valid_results.Delta_AICc = valid_results.AICc .- min_aicc
valid_results.Rel_Likelihood = exp.(-0.5 .* valid_results.Delta_AICc)
valid_results.Weight = valid_results.Rel_Likelihood ./ sum(valid_results.Rel_Likelihood)

# RVI
rvi_dict = Dict{String, Float64}()
# Let's also track the RVI of Linear vs Exponential Model Types!
rvi_dict["MATH: Linear"] = 0.0
rvi_dict["MATH: Exponential"] = 0.0

for row in eachrow(valid_results)
    rvi_dict["MATH: " * row.Type] += row.Weight
    for var in row.Combo
        str_var = string(var)
        rvi_dict[str_var] = get(rvi_dict, str_var, 0.0) + row.Weight
    end
end

rvi_df = DataFrame(Parameter = String[], RVI = Float64[])
for (k, v) in rvi_dict
    push!(rvi_df, (k, v))
end
sort!(rvi_df, :RVI, rev=true)

# ==========================================
# 5. PRINT PUBLICATION TABLES
# ==========================================
println("\n" * "="^120)
println("TABLE 1: TOP 15 MICROBIAL MEGA-MODELS (OUT OF 2048)")
println("="^120)
@printf("%-65s | %-11s | %-3s | %-8s | %-10s | %-10s\n", "Model Predictors", "Type", "K", "AICc", "Delta AICc", "Weight")
println("-"^120)
for row in eachrow(first(valid_results, 15))
    name_str = length(row.Model) > 62 ? row.Model[1:59] * "..." : row.Model
    @printf("%-65s | %-11s | %-3d | %-8.2f | %-10.2f | %-5.1f%%\n", 
        name_str, row.Type, row.K, row.AICc, row.Delta_AICc, row.Weight * 100)
end
println("="^120)

println("\n" * "="^50)
println("TABLE 2: RELATIVE VARIABLE IMPORTANCE (RVI)")
println("="^50)
@printf("%-25s | %-10s\n", "Parameter / Math Type", "Importance")
println("-"^50)
for row in eachrow(rvi_df)
    @printf("%-25s | %-5.1f%%\n", row.Parameter, row.RVI * 100)
end
println("="^50)

# ==========================================
# 6. EXTRACT WINNING PARAMETERS
# ==========================================
winner = valid_results[1, :]
winner_key = winner.Type * "_" * winner.Model
winning_obj = valid_models_cache[winner_key]

println("\n" * "="^80)
println("TABLE 3: PARAMETERS FOR ABSOLUTE WINNING MODEL")
println("Type: $(winner.Type)")
println("Equation: $(winner.Model)")
println("Note: Temperature is scaled (/10), TSS is scaled (/100)")
println("="^80)

if winner.Type == "Linear"
    return coeftable(winning_obj)
else
    # LsqFit extraction
    param_names = isempty(winner.Combo) ? ["Intercept"] : vcat(["Intercept"], string.(winner.Combo))
    se_exp = margin_error(winning_obj, 0.05)
    @printf("%-20s | %-12s | %-15s\n", "Parameter", "Coef.", "95% Error Bound")
    println("-"^80)
    for i in 1:length(param_names)
        @printf("%-20s | %12.4f | ± %12.4f\n", param_names[i], winning_obj.param[i], se_exp[i])
    end
end
println("="^80)

# ==============================================================================
# 8. FULL MODEL AVERAGING (CONFIDENCE SET)
# Purpose: Blends coefficients across top models (Delta AICc <= 4).
#          Uses the "Zero-Method" (shrinkage) for variables absent from a model.
# ==============================================================================
using Distributions
using LinearAlgebra

println("\n" * "="^95)
println("TABLE 4: FULLY AVERAGED MODEL COEFFICIENTS (CONFIDENCE SET: Delta AICc <= 4)")
println("Note: Shrinkage applied. Calculates Unconditional SE.")
println("="^95)

# 1. Filter to the Confidence Set and re-normalize weights
confidence_set = filter(row -> row.Delta_AICc <= 4.0, valid_results)
total_weight = sum(confidence_set.Weight)
confidence_set.Norm_Weight = confidence_set.Weight ./ total_weight

println("Averaging across the top $(nrow(confidence_set)) ecologically viable models...")

# 2. Define all possible parameters in the ecosystem
all_params = ["(Intercept)", "Temp_Scaled", "TSS_Scaled", "Org_Numeric", "Phase_Numeric", 
              "Temp_x_TSS", "Strain_x_TSS", "Temp_x_Strain", 
              "Phase_x_Temp", "Phase_x_TSS", "Phase_x_Strain"]

avg_coefs = Dict(p => 0.0 for p in all_params)
uncond_var = Dict(p => 0.0 for p in all_params)

# Helper function to extract coefficients safely
function get_model_info(row, mod)
    if row.Type == "Linear"
        names = coefnames(mod)
        # Indestructible parsing for GLM outputs
        c_names = names isa AbstractVector ? names : (names isa Tuple ? collect(names) : [names])
        c_names = string.(c_names)
        return c_names, coef(mod), stderror(mod)
    else 
        c_names = isempty(row.Combo) ? ["(Intercept)"] : vcat(["(Intercept)"], string.(row.Combo))
        covar = estimate_covar(mod)
        return c_names, mod.param, sqrt.(diag(covar))
    end
end

# 3. First Pass: Calculate Weighted Average Coefficients
for row in eachrow(confidence_set)
    weight = row.Norm_Weight
    model_key = row.Type * "_" * row.Model
    mod = valid_models_cache[model_key]
    
    c_names, c_vals, _ = get_model_info(row, mod)
    
    # Safely match lengths in case GLM secretly dropped a collinear variable
    safe_length = min(length(c_names), length(c_vals))
    for idx in 1:safe_length
        name = c_names[idx]
        if haskey(avg_coefs, name) && !isnan(c_vals[idx])
            avg_coefs[name] += weight * c_vals[idx]
        end
    end
end

# 4. Second Pass: Calculate Unconditional Variance
for row in eachrow(confidence_set)
    weight = row.Norm_Weight
    model_key = row.Type * "_" * row.Model
    mod = valid_models_cache[model_key]
    
    c_names, c_vals, se_vals = get_model_info(row, mod)
    
    safe_length = min(length(c_names), length(c_vals))
    curr_mod_coef = Dict(c_names[i] => c_vals[i] for i in 1:safe_length if !isnan(c_vals[i]))
    curr_mod_var = Dict(c_names[i] => se_vals[i]^2 for i in 1:safe_length if !isnan(se_vals[i]))
    
    for p in all_params
        beta_i = get(curr_mod_coef, p, 0.0) 
        var_i = get(curr_mod_var, p, 0.0)   
        
        uncond_var[p] += weight * (var_i + (beta_i - avg_coefs[p])^2)
    end
end

# 5. Compile and Calculate Final p-values
avg_results = DataFrame(
    Parameter = String[], 
    Coef = Float64[], 
    Uncond_SE = Float64[], 
    Z_value = Float64[], 
    P_value = Float64[]
)

for p in all_params
    coef_val = avg_coefs[p]
    se_val = sqrt(uncond_var[p])
    
    if se_val > 0
        z_val = coef_val / se_val
        p_val = 2 * (1 - cdf(Normal(0, 1), abs(z_val)))
        push!(avg_results, (p, coef_val, se_val, z_val, p_val))
    end
end

sort!(avg_results, :P_value)

@printf("%-18s | %-10s | %-12s | %-8s | %-10s\n", "Parameter", "Avg Coef", "Uncond. SE", "Z-value", "P-value")
println("-"^95)
for row in eachrow(avg_results)
    p_str = row.P_value < 0.001 ? "<0.001" : @sprintf("%.4f", row.P_value)
    @printf("%-18s | %10.5f | %12.5f | %8.2f | %-10s\n", 
        row.Parameter, row.Coef, row.Uncond_SE, row.Z_value, p_str)
end
println("="^95)







function show_off_model(avg_results, valid_models_cache, dat)
    fig = Figure(size = (700, 300), font = "Arial")
    
    # --- PANEL A: FOREST PLOT (Effect Sizes) ---
    ax1 = Axis(
        fig[1, 1],
        title = "A: Standardized effect sizes (model average)",
        yticks = (1:nrow(avg_results), avg_results.Parameter),
        xlabel = "Coefficient value (standardized)"
    )
    
    # Hide the intercept for better scaling of effects
    forest_data = filter(r -> r.Parameter != "(Intercept)", avg_results)
    y_points = 1:nrow(forest_data)
    
    vlines!(ax1, 0, color = :black, linewidth = 2, linestyle = :dash) # Zero line
    
    # Plot Error Bars (Unconditional SE)
    errorbars!(
        ax1,
        forest_data.Coef,
        y_points,
        forest_data.Uncond_SE, 
        direction = :x,
        color = :black,
        whiskerwidth = 10
        )
    
    # Plot Points
    scatter!(
        ax1,
        forest_data.Coef,
        y_points,
        #color = :black,
        strokewidth = 1,
        markersize = 12
    )
    
    # --- PANEL B: OBSERVED VS PREDICTED (Model Performance) ---
    ax2 = Axis(fig[1, 2], title = "B: Model accuracy (Observed vs. Predicted)",
               xlabel = "Predicted total decay (k)",
               ylabel = "Observed total decay (k)")
    
    # Get predictions from the Top Model (Rank 1)
    top_model = valid_models_cache["Linear_Temp_Scaled + Org_Numeric"]
    preds = predict(top_model)
    obs = dat.k_microbial
    
    # 1:1 Line
    line_range = [minimum(obs), maximum(obs)]

    lines!(
        ax2,
        line_range,
        line_range,
        color = :black,
        linestyle = :dot
    )
    
    # Scatter points colored by Virus Strain
    colors = [r.Org_Numeric == 1 ? :pink : :orangered for r in eachrow(dat)]
    
    scatter!(
        ax2,
        preds,
        obs,
        color = colors,
        markersize = 10, 
        strokewidth = 1,
        alpha = 0.7
    )
    
    # Add a simple Legend
    elem_1 = [MarkerElement(color = :pink, marker = :circle, markersize = 10)]
    elem_2 = [MarkerElement(color = :orangered, marker = :circle, markersize = 10)]
    Legend(fig[1, 2], [elem_1, elem_2], ["CVB5", "E11"], 
           tellheight = false, tellwidth = false, halign = :right, valign = :bottom)

    display(fig)
    save("Model_Showoff_MICROBIAL_K_Dashboard.png", fig)
end

show_off_model(avg_results, valid_models_cache, dat)