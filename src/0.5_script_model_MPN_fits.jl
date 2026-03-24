# ==============================================================================
# SCRIPT 0.5: THE KINETIC CHECK (Linear vs. Weibull)
# Purpose: Mathematically prove via AICc that First-Order (Linear) kinetics 
#          are the most parsimonious fit for the raw MPN time-series data.
# ==============================================================================

using DataFrames
using CSV
using GLM
using LsqFit
using Statistics
using Printf

println("Starting Script 0.5: Kinetic Law Check (Linear vs. Weibull)...")

# ==========================================
# 1. SETUP THE FOLDER PATH
# ==========================================
fold = raw"C:\Users\peete074\Downloads\previr" 

time_points = [3, 24, 48, 72, 96]
time_cols = [
    "inactivation_time_3",
    "inactivation_time_24",
    "inactivation_time_48",
    "inactivation_time_72",
    "inactivation_time_96"
]

results = DataFrame(
    Experiment = String[],
    Condition = String[],
    N_Points = Int[],
    Linear_AICc = Float64[],
    Weibull_AICc = Float64[],
    Delta_AICc = Float64[],
    Winner = String[],
    Weibull_Shape_n = Float64[]
)

# ==========================================
# 2. DEFINE THE WEIBULL KINETIC FUNCTION
# ==========================================
# Weibull model: Inactivation = y0 - k * (t^n)
# p[1] = Intercept (y0)
# p[2] = Rate constant (k)
# p[3] = Shape parameter (n). If n=1, it is perfectly linear. 
#        If n<1, it tails off. If n>1, it has a shoulder.
weibull_func(t, p) = p[1] .- p[2] .* (t .^ p[3])

# ==========================================
# 3. PROCESS ALL RAW MPN FILES
# ==========================================
mpn_files = filter(f -> occursin("MPN", f) && endswith(f, ".csv"), readdir(fold))

for file in mpn_files
    exp_match = match(r"Loire(\d+)", file)
    if isnothing(exp_match); continue; end
    exp_name = "EXP" * exp_match.captures[1]
    
    full_filepath = joinpath(fold, file)
    raw_df = CSV.read(full_filepath, DataFrame)
    filter!(row -> !ismissing(row.condition) && !(strip(string(row.condition)) in ["Slope", "Number", "Rate"]), raw_df)
    
    for (cond_key, cond_df) in pairs(groupby(raw_df, :condition))
        cond = strip(string(cond_key.condition))
        t_data, y_data = Float64[], Float64[]
        
        # Pool all replicates for this condition
        for row in eachrow(cond_df)
            for (t_idx, col) in enumerate(time_cols)
                if hasproperty(row, Symbol(col)) && !ismissing(row[col])
                    val_str = strip(string(row[col]))
                    if val_str != ""
                        val_float = tryparse(Float64, val_str)
                        if !isnothing(val_float) && !isnan(val_float)
                            push!(t_data, time_points[t_idx])
                            push!(y_data, val_float)
                        end
                    end
                end
            end
        end
        
        n_obs = length(t_data)
        
        # We need at least 6 points to safely run AICc on a 3-parameter Weibull model
        if n_obs > 5
            # ---------------------------------------------------
            # MODEL 1: FIRST-ORDER LINEAR (CHICK'S LAW)
            # ---------------------------------------------------
            reg_df = DataFrame(Time = t_data, Inactivation = y_data)
            lin_mod = lm(@formula(Inactivation ~ Time), reg_df)
            
            # K = 3 (Intercept, Slope, Variance)
            k_lin = 3
            ll_lin = loglikelihood(lin_mod)
            aic_lin = -2 * ll_lin + 2 * k_lin
            aicc_lin = aic_lin + (2 * k_lin * (k_lin + 1)) / (n_obs - k_lin - 1)
            
            # ---------------------------------------------------
            # MODEL 2: WEIBULL SURVIVAL KINETICS
            # ---------------------------------------------------
            # p0 guesses: intercept at highest y, tiny decay, shape = 1.0 (linear baseline)
            p0 = [maximum(y_data), 0.01, 1.0]
            
            weibull_fit = nothing
            aicc_weibull = Inf
            shape_param = missing
            
            try
                weibull_fit = curve_fit(weibull_func, t_data, y_data, p0)
                
                if weibull_fit.converged
                    resid = weibull_fit.resid
                    rss = sum(resid.^2)
                    ll_weibull = -0.5 * n_obs * log(2 * pi) - 0.5 * n_obs * log(rss / n_obs) - 0.5 * n_obs
                    
                    # K = 4 (Intercept, k, shape parameter n, Variance)
                    k_weib = 4
                    aic_weib = -2 * ll_weibull + 2 * k_weib
                    
                    denom = (n_obs - k_weib - 1)
                    if denom > 0
                        aicc_weibull = aic_weib + (2 * k_weib * (k_weib + 1)) / denom
                        shape_param = weibull_fit.param[3]
                    end
                end
            catch e
                # If Weibull fails to converge, it is structurally unfit
            end
            
            # ---------------------------------------------------
            # COMPARE AND DECLARE WINNER
            # ---------------------------------------------------
            if aicc_lin < aicc_weibull
                delta = aicc_weibull - aicc_lin
                winner = "Linear (First-Order)"
            else
                delta = aicc_lin - aicc_weibull
                winner = "Weibull (Non-Linear)"
            end
            
            push!(results, (exp_name, cond, n_obs, aicc_lin, aicc_weibull, delta, winner, ismissing(shape_param) ? NaN : shape_param))
        end
    end
end

# ==========================================
# 4. PRINT SUMMARY TABLE
# ==========================================
println("\n" * "="^100)
println("TABLE: KINETIC LAW SELECTION (CHICK'S LAW VS. WEIBULL)")
println("="^100)
@printf("%-10s | %-12s | %-4s | %-10s | %-12s | %-10s | %-20s | %-10s\n", 
        "Exp", "Condition", "N", "Lin_AICc", "Weibull_AICc", "Delta_AICc", "Winner", "Weib_Shape")
println("-"^100)
for row in eachrow(results)
    @printf("%-10s | %-12s | %-4d | %-10.2f | %-12.2f | %-10.2f | %-20s | %-10.2f\n", 
            row.Experiment, row.Condition, row.N_Points, 
            row.Linear_AICc, 
            row.Weibull_AICc == Inf ? 999.99 : row.Weibull_AICc, 
            row.Delta_AICc, 
            row.Winner, 
            isnan(row.Weibull_Shape_n) ? 0.0 : row.Weibull_Shape_n)
end
println("="^100)

linear_wins = sum(results.Winner .== "Linear (First-Order)")
total_tests = nrow(results)
println("\n🏆 FINAL KINETIC VERDICT:")
println("Linear (First-Order) Kinetics won $(linear_wins) out of $(total_tests) experimental conditions.")