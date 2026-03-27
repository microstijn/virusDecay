# ==============================================================================
# SCRIPT: TSS SATURATION TEST
# Purpose: Investigate Asymptotic TSS Saturation (Langmuir) via AICc Grid Search
#          Testing if a physical saturation curve fits the empirical data better
#          than a strictly linear TSS relationship.
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

# ==========================================
# CONSTANTS & PARAMETERS
# ==========================================
# Data Directory
data_dir = "C:/Users/peete074/Downloads/previr"
input_filepath = joinpath(data_dir, "CLEANED_FINAL_DATASET.csv")

# Grid of potential half-saturation constants (K)
K_GRID = [10.0, 25.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0]

# Response variables to test
RESPONSES = [
    (solid=:kSolid_Total, solid_std=:kSolid_Total_std, liquid=:kLiquid_Total, liquid_std=:kLiquid_Total_std, name="Total"),
    (solid=:kSolid_Abiotic, solid_std=:kSolid_Abiotic_std, liquid=:kLiquid_Abiotic, liquid_std=:kLiquid_Abiotic_std, name="Abiotic"),
    (solid=:kSolid_Microbial, solid_std=:kSolid_Microbial_std, liquid=:kLiquid_Microbial, liquid_std=:kLiquid_Microbial_std, name="Microbial")
]

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

exp_func(X_matrix, p) = exp.(X_matrix * p)


function run_pipeline(dat_full, response_info)
    println("\n" * "="^80)
    println("Running Analysis for: $(response_info.name) Decay")
    println("="^80)

    # --- 1. PREPARE BASELINE DATA ---
    df_solid = select(dropmissing(dat_full, [response_info.solid, response_info.solid_std]),
        :Experiment, :Temperature, :TSS, :Organism,
        response_info.solid => :k_val, response_info.solid_std => :SE)
    df_solid.Phase .= "Solid"

    df_liquid = select(dropmissing(dat_full, [response_info.liquid, response_info.liquid_std]),
        :Experiment, :Temperature, :TSS, :Organism,
        response_info.liquid => :k_val, response_info.liquid_std => :SE)
    df_liquid.Phase .= "Liquid"

    dat = vcat(df_solid, df_liquid)
    n_obs = nrow(dat)
    println("Loaded $(n_obs) conditions.")

    dat.Wt = 1.0 ./ (dat.SE .^ 2)
    dat.Wt_Norm = dat.Wt ./ mean(dat.Wt)
    dat.Org_Numeric = [o == "E11" ? 1.0 : 0.0 for o in dat.Organism]
    dat.Phase_Numeric = [p == "Liquid" ? 1.0 : 0.0 for p in dat.Phase]
    dat.Temp_Scaled = dat.Temperature ./ 10.0

    # Global definitions for predictors
    predictors_list = [:Temp_Scaled, :TSS_Scaled, :Org_Numeric, :Phase_Numeric,
                       :Temp_x_TSS, :Strain_x_TSS, :Temp_x_Strain,
                       :Phase_x_Temp, :Phase_x_TSS, :Phase_x_Strain]
    interactions_list = [:Temp_x_TSS, :Strain_x_TSS, :Temp_x_Strain,
                         :Phase_x_Temp, :Phase_x_TSS, :Phase_x_Strain]

    combos = get_combinations(predictors_list, interactions_list)
    resp = Term(:k_val)

    # DataFrame to store all results
    all_results = DataFrame(
        Response = String[],
        Transform = String[],
        K_val = Union{Missing, Float64}[],
        MathType = String[],
        ModelName = String[],
        K = Int[],
        AICc = Float64[]
    )

    function fit_all_combos(current_dat, transform_name, k_val_val)
        for combo in combos
            if isempty(combo)
                f = FormulaTerm(resp, ConstantTerm(1))
                base_name = "Null"
            else
                f = FormulaTerm(resp, sum([ConstantTerm(1); Term.(combo)]))
                base_name = join(string.(combo), " + ")
            end

            # Linear Fit
            try
                mod_lin = lm(f, current_dat, wts=current_dat.Wt_Norm)
                k_lin = length(coef(mod_lin)) + 1
                ll_lin = loglikelihood(mod_lin)
                aicc_lin = (-2 * ll_lin + 2 * k_lin) + (2 * k_lin * (k_lin + 1)) / (n_obs - k_lin - 1)
                push!(all_results, (response_info.name, transform_name, k_val_val, "Linear", base_name, k_lin, aicc_lin))
            catch
                push!(all_results, (response_info.name, transform_name, k_val_val, "Linear", base_name, 0, Inf))
            end

            # Exponential Fit
            try
                f_schema = apply_schema(f, schema(current_dat))
                y_data, X_data = modelcols(f_schema, current_dat)
                y_mean = mean(y_data)
                init_val = y_mean > 0 ? log(y_mean) : -3.0
                p0 = fill(0.01, size(X_data, 2))
                p0[1] = init_val

                fit_exp = curve_fit((x,p) -> exp_func(x,p), X_data, y_data, current_dat.Wt_Norm, p0)

                if fit_exp.converged
                    se = margin_error(fit_exp, 0.05)
                    if any(se .> 50.0)
                        push!(all_results, (response_info.name, transform_name, k_val_val, "Exponential", base_name, 0, Inf))
                    else
                        rss = sum(current_dat.Wt_Norm .* fit_exp.resid.^2)
                        ll_exp = -0.5 * n_obs * log(2 * pi) - 0.5 * n_obs * log(rss / n_obs) - 0.5 * n_obs
                        k_exp = length(fit_exp.param) + 1
                        aicc_exp = (-2 * ll_exp + 2 * k_exp) + (2 * k_exp * (k_exp + 1)) / (n_obs - k_exp - 1)
                        push!(all_results, (response_info.name, transform_name, k_val_val, "Exponential", base_name, k_exp, aicc_exp))
                    end
                else
                    push!(all_results, (response_info.name, transform_name, k_val_val, "Exponential", base_name, 0, Inf))
                end
            catch e
                push!(all_results, (response_info.name, transform_name, k_val_val, "Exponential", base_name, 0, Inf))
            end
        end
    end

    # --- 2. FIT BASELINE MODELS ---
    println("Fitting Baseline Models...")
    dat_base = copy(dat)
    dat_base.TSS_Scaled = dat_base.TSS ./ 100.0
    # Baseline interaction terms
    dat_base.Temp_x_TSS = dat_base.Temp_Scaled .* dat_base.TSS_Scaled
    dat_base.Strain_x_TSS = dat_base.Org_Numeric .* dat_base.TSS_Scaled
    dat_base.Temp_x_Strain = dat_base.Temp_Scaled .* dat_base.Org_Numeric
    dat_base.Phase_x_Temp = dat_base.Phase_Numeric .* dat_base.Temp_Scaled
    dat_base.Phase_x_TSS = dat_base.Phase_Numeric .* dat_base.TSS_Scaled
    dat_base.Phase_x_Strain = dat_base.Phase_Numeric .* dat_base.Org_Numeric

    fit_all_combos(dat_base, "Baseline", missing)

    # --- 3. FIT TRANSFORMED MODELS (GRID SEARCH) ---
    for k_val in K_GRID
        println("Fitting Transformed Models (K = $k_val)...")
        dat_trans = copy(dat)

        # Calculate transformed TSS
        dat_trans.TSS_Scaled = dat_trans.TSS ./ (k_val .+ dat_trans.TSS)

        # Transformed interaction terms
        dat_trans.Temp_x_TSS = dat_trans.Temp_Scaled .* dat_trans.TSS_Scaled
        dat_trans.Strain_x_TSS = dat_trans.Org_Numeric .* dat_trans.TSS_Scaled
        dat_trans.Temp_x_Strain = dat_trans.Temp_Scaled .* dat_trans.Org_Numeric
        dat_trans.Phase_x_Temp = dat_trans.Phase_Numeric .* dat_trans.Temp_Scaled
        dat_trans.Phase_x_TSS = dat_trans.Phase_Numeric .* dat_trans.TSS_Scaled
        dat_trans.Phase_x_Strain = dat_trans.Phase_Numeric .* dat_trans.Org_Numeric

        fit_all_combos(dat_trans, "Transformed", k_val)
    end

    return all_results
end

# MAIN EXECUTION
function main()
    println("Loading dataset from: $input_filepath")
    dat_full = CSV.read(input_filepath, DataFrame)

    master_results = DataFrame()

    for resp in RESPONSES
        res_df = run_pipeline(dat_full, resp)
        append!(master_results, res_df)
    end

    # Process results: calculate Delta AICc from the BEST baseline model
    master_summary = DataFrame()
    for resp_name in ["Total", "Abiotic", "Microbial"]
        df_resp = filter(r -> r.Response == resp_name && r.AICc != Inf, master_results)

        if nrow(df_resp) > 0
            # Find best baseline model
            df_base = filter(r -> r.Transform == "Baseline", df_resp)
            best_base_aicc = minimum(df_base.AICc)

            # Calculate delta AICc against best baseline (Positive = Better than baseline)
            df_resp.Delta_AICc = best_base_aicc .- df_resp.AICc

            # Summarize by grouping
            for group in eachrow(unique(df_resp[!, [:Transform, :K_val]]))
                df_group = filter(r -> r.Transform == group.Transform && isequal(r.K_val, group.K_val), df_resp)

                # Get the absolute best model in this group
                best_idx = argmin(df_group.AICc)
                best_model_row = df_group[best_idx, :]

                push!(master_summary, (
                    Response = resp_name,
                    Transform = group.Transform,
                    K_val = group.K_val,
                    MathType = best_model_row.MathType,
                    AICc = best_model_row.AICc,
                    Delta_AICc = best_model_row.Delta_AICc,
                    Best_Predictors = best_model_row.ModelName
                ))
            end
        end
    end

    println("\n" * "="^120)
    println("SUMMARY RESULTS")
    println("="^120)

    for resp_name in ["Total", "Abiotic", "Microbial"]
        println("\n--- $(uppercase(resp_name)) DECAY ---")
        df_print = filter(r -> r.Response == resp_name, master_summary)
        sort!(df_print, :AICc)

        @printf("%-12s | %-7s | %-12s | %-8s | %-10s | %-50s\n", "Transform", "K_val", "MathType", "AICc", "Delta AICc", "Best Predictors")
        println("-"^120)
        for row in eachrow(df_print)
            k_str = ismissing(row.K_val) ? "N/A" : @sprintf("%.1f", row.K_val)
            pred_str = length(row.Best_Predictors) > 47 ? row.Best_Predictors[1:44] * "..." : row.Best_Predictors
            @printf("%-12s | %-7s | %-12s | %-8.2f | %-10.2f | %-50s\n",
                row.Transform, k_str, row.MathType, row.AICc, row.Delta_AICc, pred_str)
        end
    end
end

# Run the main execution
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
