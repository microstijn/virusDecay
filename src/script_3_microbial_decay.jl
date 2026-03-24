# ==============================================================================
# SCRIPT 3: MICROBIAL DECAY (K_MICROBIAL) MASTER DREDGE
# Purpose: Programmatically test biologically legal combinations of predictors
#          using BOTH Linear and Exponential math on the Microbial System (N=36).
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
using LinearAlgebra

function get_combinations(predictors, interactions)
    combos = Vector{Symbol}[]
    n = length(predictors)
    for i in 0:(2^n - 1)
        idx = digits(i, base=2, pad=n)
        combo = predictors[findall(==(1), idx)]

        valid = true

        # --- 1. THE MARGINALITY BOUNCER (Mathematical Filter) ---
        if :Temp_x_TSS in combo && (!(:Temp_Scaled in combo) || !(:TSS_Scaled in combo)); valid = false; end
        if :Strain_x_TSS in combo && (!(:Org_Numeric in combo) || !(:TSS_Scaled in combo)); valid = false; end
        if :Temp_x_Strain in combo && (!(:Temp_Scaled in combo) || !(:Org_Numeric in combo)); valid = false; end
        if :Phase_x_Temp in combo && (!(:Phase_Numeric in combo) || !(:Temp_Scaled in combo)); valid = false; end
        if :Phase_x_TSS in combo && (!(:Phase_Numeric in combo) || !(:TSS_Scaled in combo)); valid = false; end
        if :Phase_x_Strain in combo && (!(:Phase_Numeric in combo) || !(:Org_Numeric in combo)); valid = false; end

        # --- 2. THE BIOPHYSICAL BOUNCER (A Priori Knowledge) ---
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

function get_model_info(row, mod)
    if row.Type == "Linear"
        names = coefnames(mod)
        c_names = names isa AbstractVector ? names : (names isa Tuple ? collect(names) : [names])
        return string.(c_names), coef(mod), stderror(mod)
    else
        c_names = isempty(row.Combo) ? ["(Intercept)"] : vcat(["(Intercept)"], string.(row.Combo))
        covar = estimate_covar(mod)
        return c_names, mod.param, sqrt.(diag(covar))
    end
end

function print_tables(valid_results, rvi_df, valid_models_cache, avg_results)
    println("\n" * "="^125)
    println("TABLE 1: TOP 15 MODELS")
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

    winner = valid_results[1, :]
    winner_key = winner.Type * "_" * winner.Model
    winning_obj = valid_models_cache[winner_key]

    println("\n" * "="^85)
    println("TABLE 3: PARAMETERS FOR WINNING MODEL")
    println("Type: $(winner.Type)")
    println("Equation: $(winner.Model)")
    println("="^85)

    if winner.Type == "Linear"
        println(coeftable(winning_obj))
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

    println("\n" * "="^95)
    println("TABLE 4: AVERAGED MODEL PARAMETERS (Confidence Set Delta AICc <= 4)")
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

function plot_results(avg_results, avg_preds, dat, plot_filename)
    fig = Figure(size = (800, 350), font = "Arial")

    # --- PANEL A: FOREST PLOT ---
    forest_data = filter(r -> r.Parameter != "(Intercept)", avg_results)
    y_points = reverse(1:nrow(forest_data))

    ax1 = Axis(
        fig[1, 1],
        title = "A: Standardized effect sizes (Model Average)",
        yticks = (y_points, forest_data.Parameter),
        xlabel = "Coefficient value (standardized)"
    )

    vlines!(ax1, 0, color = :black, linewidth = 2, linestyle = :dash)
    errorbars!(ax1, forest_data.Coef, y_points, forest_data.Uncond_SE, direction = :x, color = :black, whiskerwidth = 8)
    scatter!(ax1, forest_data.Coef, y_points, strokewidth = 1, markersize = 12, color = :black)

    # --- PANEL B: OBSERVED VS PREDICTED ---
    ax2 = Axis(fig[1, 2], title = "B: Model accuracy (Averaged Model)",
               xlabel = "Predicted microbial decay (k)",
               ylabel = "Observed microbial decay (k)")

    preds = avg_preds
    obs = dat.k_val

    line_range = [minimum(vcat(obs, preds)), maximum(vcat(obs, preds))]
    lines!(ax2, line_range, line_range, color = :black, linestyle = :dot)

    colors = [r.Org_Numeric == 1.0 ? :pink : :orangered for r in eachrow(dat)]
    scatter!(ax2, preds, obs, color = colors, markersize = 10, strokewidth = 1, alpha = 0.7)

    elem_1 = [MarkerElement(color = :pink, marker = :circle, markersize = 10)]
    elem_2 = [MarkerElement(color = :orangered, marker = :circle, markersize = 10)]
    Legend(fig[1, 2], [elem_1, elem_2], ["CVB5", "E11"], tellheight = false, tellwidth = false, halign = :right, valign = :bottom)

    save(plot_filename, fig)
    println("\n✅ Plot saved as '$(plot_filename)'")
end

function main()
    println("Starting Script 3: Microbial K Mega-Dredge (Biophysically-Filtered)...")

    # ==========================================
    # 1. LOAD & PREPARE DATA
    # ==========================================
    data_dir = "C:/Users/peete074/Downloads/previr"
    input_filepath = joinpath(data_dir, "CLEANED_FINAL_DATASET.csv")
    dat_full = CSV.read(input_filepath, DataFrame)

    # Isolate Solid Phase
    df_solid = select(dropmissing(dat_full, [:kSolid_Microbial, :kSolid_Microbial_std]),
        :Experiment, :Temperature, :TSS, :Organism,
        :kSolid_Microbial => :k_val, :kSolid_Microbial_std => :SE)
    df_solid.Phase .= "Solid"

    # Isolate Liquid Phase
    df_liquid = select(dropmissing(dat_full, [:kLiquid_Microbial, :kLiquid_Microbial_std]),
        :Experiment, :Temperature, :TSS, :Organism,
        :kLiquid_Microbial => :k_val, :kLiquid_Microbial_std => :SE)
    df_liquid.Phase .= "Liquid"

    # Combine into master dataset
    dat = vcat(df_solid, df_liquid)
    n_obs = nrow(dat)

    println("✅ Loaded $(n_obs) combined conditions for Microbial Decay.")

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

    interactions = [:Temp_x_TSS, :Strain_x_TSS, :Temp_x_Strain,
                    :Phase_x_Temp, :Phase_x_TSS, :Phase_x_Strain]

    combos = get_combinations(predictors, interactions)
    println("✅ Filters applied: $(length(combos)) biologically legal structures generated.")

    # ==========================================
    # 4. FIT ALL MODELS (LINEAR & EXPONENTIAL)
    # ==========================================
    resp = Term(:k_val)
    exp_func(X_matrix, p) = exp.(X_matrix * p)

    results = DataFrame(Model = String[], Type = String[], K = Int[], AICc = Float64[], Status = String[], Combo = Vector{Symbol}[])
    valid_models_cache = Dict{String, Any}()

    println("Fitting models...")

    for combo in combos
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
            push!(results, (base_name, "Linear", 0, Inf, "Singular Matrix", combo))
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
    # 6. MULTIMODEL AVERAGING (ZERO-METHOD)
    # ==========================================
    conf_set = filter(r -> r.Delta_AICc <= 4.0, valid_results)
    conf_set.W_renorm = conf_set.Rel_Likelihood ./ sum(conf_set.Rel_Likelihood)

    all_params = ["(Intercept)", "Temp_Scaled", "TSS_Scaled", "Org_Numeric", "Phase_Numeric",
                  "Temp_x_TSS", "Strain_x_TSS", "Temp_x_Strain",
                  "Phase_x_Temp", "Phase_x_TSS", "Phase_x_Strain"]

    avg_coefs = Dict(p => 0.0 for p in all_params)
    uncond_var = Dict(p => 0.0 for p in all_params)

    # Pass 1: Weighted average coefficients
    for row in eachrow(conf_set)
        mod = valid_models_cache[row.Type * "_" * row.Model]
        c_names, c_vals, _ = get_model_info(row, mod)

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
        mod = valid_models_cache[row.Type * "_" * row.Model]
        c_names, c_vals, se_vals = get_model_info(row, mod)

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

    # ==========================================
    # 7. MODEL AVERAGED PREDICTIONS
    # ==========================================
    avg_preds = zeros(Float64, n_obs)
    for row in eachrow(conf_set)
        mod = valid_models_cache[row.Type * "_" * row.Model]
        if row.Type == "Linear"
            pred_i = predict(mod)
        else
            f_temp = isempty(row.Combo) ? FormulaTerm(resp, ConstantTerm(1)) : FormulaTerm(resp, sum(Term.(row.Combo)))
            f_schema_temp = apply_schema(f_temp, schema(dat))
            _, X_temp = modelcols(f_schema_temp, dat)
            pred_i = exp_func(X_temp, mod.param)
        end
        avg_preds .+= row.W_renorm .* pred_i
    end

    # ==========================================
    # 8. PRINTING TABLES
    # ==========================================
    print_tables(valid_results, rvi_df, valid_models_cache, avg_results)

    # ==========================================
    # 9. PLOTTING
    # ==========================================
    plot_results(avg_results, avg_preds, dat, "dashboard_microbial_decay.png")
end

main()
