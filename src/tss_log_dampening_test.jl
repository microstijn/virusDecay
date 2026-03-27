# ==============================================================================
# SCRIPT: TSS LOG-DAMPENING TEST
# Purpose: Investigate Logarithmic Dampening via AICc Evaluation
#          Testing if a natural log transformation of TSS (ln(TSS+1)) fits the
#          empirical data better than a strictly linear TSS relationship.
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
using LinearAlgebra

# ==========================================
# CONSTANTS & PARAMETERS
# ==========================================
data_dir = "C:/Users/peete074/Downloads/previr"
input_filepath = joinpath(data_dir, "CLEANED_FINAL_DATASET.csv")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
function get_combinations(predictors, interactions)
    combos = Vector{Symbol}[]
    n = length(predictors)
    for i in 0:(2^n - 1)
        idx = digits(i, base=2, pad=n)
        combo = predictors[findall(==(1), idx)]

        valid = true
        if :Temp_x_TSS in combo && (!(:Temp_Scaled in combo) || !(:TSS_Scaled in combo)); valid = false; end
        if :Strain_x_TSS in combo && (!(:Org_Numeric in combo) || !(:TSS_Scaled in combo)); valid = false; end
        if :Temp_x_Strain in combo && (!(:Temp_Scaled in combo) || !(:Org_Numeric in combo)); valid = false; end
        if :Phase_x_Temp in combo && (!(:Phase_Numeric in combo) || !(:Temp_Scaled in combo)); valid = false; end
        if :Phase_x_TSS in combo && (!(:Phase_Numeric in combo) || !(:TSS_Scaled in combo)); valid = false; end
        if :Phase_x_Strain in combo && (!(:Phase_Numeric in combo) || !(:Org_Numeric in combo)); valid = false; end

        if !isempty(combo)
            if !(:Temp_Scaled in combo); valid = false; end
            if :TSS_Scaled in combo && !(:Phase_Numeric in combo); valid = false; end
            num_interactions = sum([1 for x in combo if x in interactions])
            if num_interactions > 3; valid = false; end
        end

        if valid; push!(combos, combo); end
    end
    return combos
end

exp_func(X_matrix, p) = exp.(X_matrix * p)

function get_model_info(mod_type, combo, mod)
    if mod_type == "Linear"
        names = coefnames(mod)
        c_names = names isa AbstractVector ? names : (names isa Tuple ? collect(names) : [names])
        return string.(c_names), coef(mod), stderror(mod)
    else
        c_names = isempty(combo) ? ["(Intercept)"] : vcat(["(Intercept)"], string.(combo))
        covar = estimate_covar(mod)
        return c_names, mod.param, sqrt.(diag(covar))
    end
end

# ==========================================
# MAIN EXECUTION
# ==========================================
function main()
    println("Loading dataset from: $input_filepath")
    dat_full = CSV.read(input_filepath, DataFrame)

    # 1. PREPARE DATA (Total K)
    df_solid = select(dropmissing(dat_full, [:kSolid_Total, :kSolid_Total_std]),
        :Experiment, :Temperature, :TSS, :Organism,
        :kSolid_Total => :k_val, :kSolid_Total_std => :SE)
    df_solid.Phase .= "Solid"

    df_liquid = select(dropmissing(dat_full, [:kLiquid_Total, :kLiquid_Total_std]),
        :Experiment, :Temperature, :TSS, :Organism,
        :kLiquid_Total => :k_val, :kLiquid_Total_std => :SE)
    df_liquid.Phase .= "Liquid"

    dat = vcat(df_solid, df_liquid)
    n_obs = nrow(dat)
    println("Loaded $(n_obs) conditions for Total Decay.")

    dat.Wt = 1.0 ./ (dat.SE .^ 2)
    dat.Wt_Norm = dat.Wt ./ mean(dat.Wt)
    dat.Org_Numeric = [o == "E11" ? 1.0 : 0.0 for o in dat.Organism]
    dat.Phase_Numeric = [p == "Liquid" ? 1.0 : 0.0 for p in dat.Phase]
    dat.Temp_Scaled = dat.Temperature ./ 10.0

    predictors_list = [:Temp_Scaled, :TSS_Scaled, :Org_Numeric, :Phase_Numeric,
                       :Temp_x_TSS, :Strain_x_TSS, :Temp_x_Strain,
                       :Phase_x_Temp, :Phase_x_TSS, :Phase_x_Strain]
    interactions_list = [:Temp_x_TSS, :Strain_x_TSS, :Temp_x_Strain,
                         :Phase_x_Temp, :Phase_x_TSS, :Phase_x_Strain]
    combos = get_combinations(predictors_list, interactions_list)
    resp = Term(:k_val)

    # Reusable fitting function
    function fit_all(current_dat)
        results = DataFrame(Model = String[], Type = String[], Num_Params = Int[], AICc = Float64[], Combo = Vector{Symbol}[])
        cache = Dict{String, Any}()

        for combo in combos
            if isempty(combo)
                f = FormulaTerm(resp, ConstantTerm(1))
                base_name = "Null"
            else
                f = FormulaTerm(resp, sum([ConstantTerm(1); Term.(combo)]))
                base_name = join(string.(combo), " + ")
            end

            # Linear
            try
                mod_lin = lm(f, current_dat, wts=current_dat.Wt_Norm)
                k_lin = length(coef(mod_lin)) + 1
                ll_lin = loglikelihood(mod_lin)
                aicc_lin = (-2 * ll_lin + 2 * k_lin) + (2 * k_lin * (k_lin + 1)) / (n_obs - k_lin - 1)
                push!(results, (base_name, "Linear", k_lin, aicc_lin, combo))
                cache["Linear_" * base_name] = mod_lin
            catch
            end

            # Exponential
            try
                f_schema = apply_schema(f, schema(current_dat))
                y_data, X_data = modelcols(f_schema, current_dat)
                init_val = mean(y_data) > 0 ? log(mean(y_data)) : -3.0
                p0 = fill(0.01, size(X_data, 2))
                p0[1] = init_val

                fit_exp = curve_fit((x,p) -> exp_func(x,p), X_data, y_data, current_dat.Wt_Norm, p0)
                if fit_exp.converged && !any(margin_error(fit_exp, 0.05) .> 50.0)
                    rss = sum(current_dat.Wt_Norm .* fit_exp.resid.^2)
                    ll_exp = -0.5 * n_obs * log(2 * pi) - 0.5 * n_obs * log(rss / n_obs) - 0.5 * n_obs
                    k_exp = length(fit_exp.param) + 1
                    aicc_exp = (-2 * ll_exp + 2 * k_exp) + (2 * k_exp * (k_exp + 1)) / (n_obs - k_exp - 1)
                    push!(results, (base_name, "Exponential", k_exp, aicc_exp, combo))
                    cache["Exponential_" * base_name] = fit_exp
                end
            catch
            end
        end
        return results, cache
    end

    # 2. RUN BASELINE
    println("Evaluating Baseline (TSS / 100)...")
    dat_base = copy(dat)
    dat_base.TSS_Scaled = dat_base.TSS ./ 100.0
    dat_base.Temp_x_TSS = dat_base.Temp_Scaled .* dat_base.TSS_Scaled
    dat_base.Strain_x_TSS = dat_base.Org_Numeric .* dat_base.TSS_Scaled
    dat_base.Temp_x_Strain = dat_base.Temp_Scaled .* dat_base.Org_Numeric
    dat_base.Phase_x_Temp = dat_base.Phase_Numeric .* dat_base.Temp_Scaled
    dat_base.Phase_x_TSS = dat_base.Phase_Numeric .* dat_base.TSS_Scaled
    dat_base.Phase_x_Strain = dat_base.Phase_Numeric .* dat_base.Org_Numeric

    res_base, _ = fit_all(dat_base)
    best_base_aicc = minimum(res_base.AICc)

    # 3. RUN LOG-DAMPENING
    println("Evaluating Log-Dampened (ln(TSS + 1))...")
    dat_log = copy(dat)
    dat_log.TSS_Scaled = log.(dat_log.TSS .+ 1.0)
    dat_log.Temp_x_TSS = dat_log.Temp_Scaled .* dat_log.TSS_Scaled
    dat_log.Strain_x_TSS = dat_log.Org_Numeric .* dat_log.TSS_Scaled
    dat_log.Temp_x_Strain = dat_log.Temp_Scaled .* dat_log.Org_Numeric
    dat_log.Phase_x_Temp = dat_log.Phase_Numeric .* dat_log.Temp_Scaled
    dat_log.Phase_x_TSS = dat_log.Phase_Numeric .* dat_log.TSS_Scaled
    dat_log.Phase_x_Strain = dat_log.Phase_Numeric .* dat_log.Org_Numeric

    res_log, cache_log = fit_all(dat_log)
    best_log_aicc = minimum(res_log.AICc)
    delta_aicc_global = best_base_aicc - best_log_aicc

    # 4. PRINT COMPARISON SUMMARY
    println("\n" * "="^80)
    println("LOG-DAMPENING SUMMARY RESULTS (Total Decay)")
    println("="^80)
    @printf("Best Baseline AICc:     %.2f\n", best_base_aicc)
    @printf("Best Log-Dampened AICc: %.2f\n", best_log_aicc)
    @printf("Delta AICc Improvement: %.2f  (> 2 is meaningful)\n", delta_aicc_global)

    # 5. MULTIMODEL AVERAGING FOR LOG-DAMPENED MODELS
    min_log_aicc = minimum(res_log.AICc)
    res_log.Delta_AICc = res_log.AICc .- min_log_aicc
    res_log.Rel_Likelihood = exp.(-0.5 .* res_log.Delta_AICc)
    res_log.Weight = res_log.Rel_Likelihood ./ sum(res_log.Rel_Likelihood)

    conf_set = filter(r -> r.Delta_AICc <= 4.0, res_log)
    conf_set.W_renorm = conf_set.Rel_Likelihood ./ sum(conf_set.Rel_Likelihood)

    all_params = ["(Intercept)", "Temp_Scaled", "TSS_Scaled", "Org_Numeric", "Phase_Numeric",
                  "Temp_x_TSS", "Strain_x_TSS", "Temp_x_Strain",
                  "Phase_x_Temp", "Phase_x_TSS", "Phase_x_Strain"]
    avg_coefs = Dict(p => 0.0 for p in all_params)
    uncond_var = Dict(p => 0.0 for p in all_params)

    # Pass 1: Weighted average coefficients
    for row in eachrow(conf_set)
        mod = cache_log[row.Type * "_" * row.Model]
        c_names, c_vals, _ = get_model_info(row.Type, row.Combo, mod)

        safe_length = min(length(c_names), length(c_vals))
        for idx in 1:safe_length
            name = c_names[idx]
            if haskey(avg_coefs, name) && !isnan(c_vals[idx])
                avg_coefs[name] += row.W_renorm * c_vals[idx]
            end
        end
    end

    # Pass 2: Unconditional variance
    for row in eachrow(conf_set)
        mod = cache_log[row.Type * "_" * row.Model]
        c_names, c_vals, se_vals = get_model_info(row.Type, row.Combo, mod)

        safe_length = min(length(c_names), length(c_vals))
        curr_mod_coef = Dict(c_names[i] => c_vals[i] for i in 1:safe_length if !isnan(c_vals[i]))
        curr_mod_var = Dict(c_names[i] => se_vals[i]^2 for i in 1:safe_length if !isnan(se_vals[i]))

        for p in all_params
            beta_i = get(curr_mod_coef, p, 0.0)
            var_i = get(curr_mod_var, p, 0.0)
            uncond_var[p] += row.W_renorm * (var_i + (beta_i - avg_coefs[p])^2)
        end
    end

    avg_results = DataFrame(Parameter=String[], Coef=Float64[], Uncond_SE=Float64[], Z=Float64[], P_val=Float64[])
    for p in all_params
        theta_bar = avg_coefs[p]
        var_uncond = uncond_var[p]

        if var_uncond > 0 || theta_bar != 0.0
            se_final = sqrt(var_uncond)
            z_val = se_final > 0 ? theta_bar / se_final : NaN
            p_val = !isnan(z_val) ? 2 * (1 - cdf(Normal(), abs(z_val))) : NaN

            push!(avg_results, (p, theta_bar, se_final, z_val, p_val))
        end
    end
    sort!(avg_results, :P_val)

    # 6. PRINT AVERAGED COEFFICIENTS
    println("\n" * "="^95)
    println("TABLE 4: AVERAGED MODEL PARAMETERS (Log-Dampened ln(TSS+1), Delta AICc <= 4)")
    println("="^95)
    @printf("%-18s | %-10s | %-12s | %-8s | %-10s\n", "Parameter", "Avg Coef", "Uncond. SE", "Z-value", "P-value")
    println("-"^95)
    for row in eachrow(avg_results)
        p_str = row.P_val < 0.001 ? "<0.001" : @sprintf("%.4f", row.P_val)
        @printf("%-18s | %10.5f | %12.5f | %8.2f | %-10s\n",
            row.Parameter, row.Coef, row.Uncond_SE, row.Z, p_str)
    end
    println("="^95)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
