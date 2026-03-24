# ==============================================================================
# SCRIPT 6: TOTAL DECAY (K_TOTAL) MASTER DREDGE
# Purpose: Programmatically test biologically legal combinations of predictors
#          using BOTH Linear and Exponential math on the Total System (N=36).
# ==============================================================================

using DataFrames
using CSV
using GLM
using LsqFit
using StatsModels
using StatsBase
using Printf
using Statistics

println("Starting Script 6: Total K Mega-Dredge (Marginality-Filtered)...")

# ==========================================
# 1. LOAD & PREPARE TOTAL DECAY DATA (N=36)
# ==========================================
data_dir = "C:/Users/peete074/Downloads/previr"
input_filepath = joinpath(data_dir, "CLEANED_FINAL_DATASET.csv")
dat_full = CSV.read(input_filepath, DataFrame)

# Isolate Solid Phase (Total K)
df_solid = select(dropmissing(dat_full, [:kSolid_Total, :kSolid_Total_std]),
    :Experiment, :Temperature, :TSS, :Organism, 
    :kSolid_Total => :k_total, :kSolid_Total_std => :SE)
df_solid.Phase .= "Solid"

# Isolate Liquid Phase (Total K)
df_liquid = select(dropmissing(dat_full, [:kLiquid_Total, :kLiquid_Total_std]),
    :Experiment, :Temperature, :TSS, :Organism, 
    :kLiquid_Total => :k_total, :kLiquid_Total_std => :SE)
df_liquid.Phase .= "Liquid"

# Combine into master dataset
dat = vcat(df_solid, df_liquid)
n_obs = nrow(dat)

println("✅ Loaded $(n_obs) combined conditions for Total Decay.")

# ==========================================
# 2. CALCULATE WEIGHTS & SCALED PREDICTORS
# ==========================================
# Normalized Inverse-Variance Weights
dat.Wt = 1.0 ./ (dat.SE .^ 2)
dat.Wt_Norm = dat.Wt ./ mean(dat.Wt)

# Numeric Predictors
dat.Org_Numeric = [o == "E11" ? 1.0 : 0.0 for o in dat.Organism]
dat.Phase_Numeric = [p == "Liquid" ? 1.0 : 0.0 for p in dat.Phase] # Liquid = 1, Solid = 0

# Scale continuous variables to prevent matrix explosions
dat.Temp_Scaled = dat.Temperature ./ 10.0
dat.TSS_Scaled = dat.TSS ./ 100.0

# Pre-calculate interactions
dat.Temp_x_TSS = dat.Temp_Scaled .* dat.TSS_Scaled
dat.Strain_x_TSS = dat.Org_Numeric .* dat.TSS_Scaled
dat.Temp_x_Strain = dat.Temp_Scaled .* dat.Org_Numeric
dat.Phase_x_Temp = dat.Phase_Numeric .* dat.Temp_Scaled
dat.Phase_x_TSS = dat.Phase_Numeric .* dat.TSS_Scaled
dat.Phase_x_Strain = dat.Phase_Numeric .* dat.Org_Numeric

# ==========================================
# 3. GENERATE MARGINALITY-FILTERED COMBINATIONS
# ==========================================
predictors = [:Temp_Scaled, :TSS_Scaled, :Org_Numeric, :Phase_Numeric, 
              :Temp_x_TSS, :Strain_x_TSS, :Temp_x_Strain, 
              :Phase_x_Temp, :Phase_x_TSS, :Phase_x_Strain]

function get_combinations(arr)
    n = length(arr)
    combos = Vector{Symbol}[]
    for i in 0:(2^n - 1)
        idx = digits(i, base=2, pad=n)
        combo = arr[findall(==(1), idx)]
        
        # --- THE MARGINALITY BOUNCER ---
        valid = true
        if :Temp_x_TSS in combo && (!(:Temp_Scaled in combo) || !(:TSS_Scaled in combo)); valid = false; end
        if :Strain_x_TSS in combo && (!(:Org_Numeric in combo) || !(:TSS_Scaled in combo)); valid = false; end
        if :Temp_x_Strain in combo && (!(:Temp_Scaled in combo) || !(:Org_Numeric in combo)); valid = false; end
        if :Phase_x_Temp in combo && (!(:Phase_Numeric in combo) || !(:Temp_Scaled in combo)); valid = false; end
        if :Phase_x_TSS in combo && (!(:Phase_Numeric in combo) || !(:TSS_Scaled in combo)); valid = false; end
        if :Phase_x_Strain in combo && (!(:Phase_Numeric in combo) || !(:Org_Numeric in combo)); valid = false; end
        
        if valid
            push!(combos, combo)
        end
    end
    return combos
end

valid_combos = get_combinations(predictors)
println("✅ Marginality filter applied: $(length(valid_combos)) biologically legal model structures generated.")

# ==========================================
# 4. FIT ALL MODELS (LINEAR & EXPONENTIAL)
# ==========================================
resp = Term(:k_total)
exp_func(X_matrix, p) = exp.(X_matrix * p)

results = DataFrame(Model = String[], Type = String[], K = Int[], AICc = Float64[], Status = String[], Combo = Vector{Symbol}[])
valid_models_cache = Dict{String, Any}()

println("Fitting models. Let the duel begin...")

for combo in valid_combos
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
        
        push!(results, (base_name, "Linear", k_lin, aicc_lin, "Passed", combo))
        valid_models_cache["Linear_" * base_name] = mod_lin
    catch
        push!(results, (base_name, "Linear", 0, Inf, "Singular Matrix", combo))
    end
    
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
# 5. CALCULATE AICc WEIGHTS & RVI
# ==========================================
valid_results = filter(row -> row.AICc != Inf, results)
sort!(valid_results, :AICc)

min_aicc = minimum(valid_results.AICc)
valid_results.Delta_AICc = valid_results.AICc .- min_aicc
valid_results.Rel_Likelihood = exp.(-0.5 .* valid_results.Delta_AICc)
valid_results.Weight = valid_results.Rel_Likelihood ./ sum(valid_results.Rel_Likelihood)

rvi_dict = Dict{String, Float64}()
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
# 6. PRINT PUBLICATION TABLES
# ==========================================
println("\n" * "="^125)
println("TABLE 1: TOP 15 TOTAL DECAY MODELS (MARGINALITY-FILTERED)")
println("="^125)
@printf("%-70s | %-11s | %-3s | %-8s | %-10s | %-10s\n", "Model Predictors", "Type", "K", "AICc", "Delta AICc", "Weight")
println("-"^125)
for row in eachrow(first(valid_results, 15))
    name_str = length(row.Model) > 67 ? row.Model[1:64] * "..." : row.Model
    @printf("%-70s | %-11s | %-3d | %-8.2f | %-10.2f | %-5.1f%%\n", 
        name_str, row.Type, row.K, row.AICc, row.Delta_AICc, row.Weight * 100)
end
println("="^125)

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
# 7. EXTRACT WINNING PARAMETERS
# ==========================================
winner = valid_results[1, :]
winner_key = winner.Type * "_" * winner.Model
winning_obj = valid_models_cache[winner_key]

println("\n" * "="^85)
println("TABLE 3: PARAMETERS FOR WINNING TOTAL DECAY MODEL")
println("Type: $(winner.Type)")
println("Equation: $(winner.Model)")
println("Note: Temp (/10), TSS (/100), Phase (Liquid=1), Strain (E11=1)")
println("="^85)

if winner.Type == "Linear"
    coeftable(winning_obj)
else
    param_names = isempty(winner.Combo) ? ["Intercept"] : vcat(["Intercept"], string.(winner.Combo))
    se_exp = margin_error(winning_obj, 0.05)
    @printf("%-20s | %-12s | %-15s\n", "Parameter", "Coef.", "95% Error Bound")
    println("-"^85)
    for i in 1:length(param_names)
        @printf("%-20s | %12.4f | ± %12.4f\n", param_names[i], winning_obj.param[i], se_exp[i])
    end
end
println("="^85)