using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# ==============================================================================
# SCRIPT 0: RAW MPN TO MODEL DATA
# Purpose: Calculate k rates and Std Error directly from time-series MPN files.
#          Safely handles messy Excel cells (text, blanks, NDs).
# ==============================================================================

using DataFrames
using CSV
using GLM
using Statistics

# ==========================================
# 1. SETUP THE FOLDER PATH
# ==========================================

fold = raw"C:\Users\peete074\Downloads\previr" 

# ==========================================
# 2. SETUP METADATA
# ==========================================
time_points = [3, 24, 48, 72, 96]
time_cols = [
    "inactivation_time_3",
    "inactivation_time_24",
    "inactivation_time_48",
    "inactivation_time_72",
    "inactivation_time_96"
]


env_params = Dict(
    "EXP1" => (TSS = 88.0, Salinity = 19.3),
    "EXP2" => (TSS = 194.0, Salinity = 26.0),
    "EXP3" => (TSS = 72.0, Salinity = 12.6)
)

derived_rates = DataFrame(
    Experiment = String[],
    Temperature = Int[],
    Organism = String[],
    Condition = String[],
    k_rate = Float64[],
    k_std = Float64[]
)

# ==========================================
# 3. PROCESS ALL RAW MPN FILES
# ==========================================
mpn_files = filter(
    f -> occursin("MPN", f) && endswith(f, ".csv"),
    readdir(fold)
)

# processing Excel files so much fun... 
for file in mpn_files
    exp_match = match(r"Loire(\d+)", file)
    temp_match = match(r"T(\d+)", file)
    if isnothing(exp_match) || isnothing(temp_match); continue; end
    
    exp_name = "EXP" * exp_match.captures[1]
    temp_val = parse(Int, temp_match.captures[1])
    
    full_filepath = joinpath(fold, file)
    raw_df = CSV.read(full_filepath, DataFrame)
    
    # Drop Excel summary rows
    filter!(row -> !ismissing(row.condition) && !(strip(string(row.condition)) in ["Slope", "Number", "Rate"]), raw_df)
    
    for (cond_key, cond_df) in pairs(groupby(raw_df, :condition))
        cond = strip(string(cond_key.condition))
        x_pooled, y_pooled = Float64[], Float64[]
        
        for row in eachrow(cond_df)
            for (t_idx, col) in enumerate(time_cols)
                if hasproperty(row, Symbol(col)) && !ismissing(row[col])
                    # BUGFIX: Try to parse the cell as a float. If it's text/blank, ignore it.
                    val_str = strip(string(row[col]))
                    if val_str != ""
                        val_float = tryparse(Float64, val_str)
                        if !isnothing(val_float) && !isnan(val_float)
                            push!(x_pooled, time_points[t_idx])
                            push!(y_pooled, val_float)
                        end
                    end
                end
            end
        end
        
        # Only run regression if we successfully extracted more than 2 valid time points
        if length(x_pooled) > 2
            reg_df = DataFrame(Time = x_pooled, Inactivation = y_pooled)
            model = lm(@formula(Inactivation ~ Time), reg_df)
            
            slope_log10 = coef(model)[2]
            se_slope_log10 = stderror(model)[2]
            
            # Scale both the rate and the standard error by 2.303
            k_rate = slope_log10 * -2.303
            k_std = se_slope_log10 * 2.303
            
            org = occursin("E11", cond) ? "E11" : "CVB5"
            push!(derived_rates, (exp_name, temp_val, org, cond, k_rate, k_std))
        end
    end
end

# ==========================================
# 4. PIVOT DATA & ISOLATE ECOLOGICAL MECHANISMS
# ==========================================
final_df = DataFrame()

for (group_key, group_df) in pairs(groupby(derived_rates, [:Experiment, :Temperature, :Organism]))
    exp, temp, org = group_key.Experiment, group_key.Temperature, group_key.Organism
    
    get_val(c) = begin
        row = filter(:Condition => ==(c), group_df)
        nrow(row) > 0 ? (row.k_rate[1], row.k_std[1]) : (missing, missing)
    end
    
    L_rate, L_std       = get_val("$(org)L")
    S_rate, S_std       = get_val("$(org)S")
    NegL_rate, NegL_std = get_val("$(org)NegL")
    NegS_rate, NegS_std = get_val("$(org)NegS")
    
    # Solid Microbial Decay (Total - Abiotic)
    if !ismissing(S_rate) && !ismissing(NegS_rate)
        kSolid_Microbial = S_rate - NegS_rate
        kSolid_Microbial_std = sqrt(S_std^2 + NegS_std^2)
        Wt_Solid_Mic = 1.0 / (kSolid_Microbial_std^2)
    else
        kSolid_Microbial, kSolid_Microbial_std, Wt_Solid_Mic = missing, missing, missing
    end
    
    # 2. Liquid Microbial Decay (Total - Abiotic)
    if !ismissing(L_rate) && !ismissing(NegL_rate)
        kLiquid_Microbial = L_rate - NegL_rate
        kLiquid_Microbial_std = sqrt(L_std^2 + NegL_std^2)
        Wt_Liquid_Mic = 1.0 / (kLiquid_Microbial_std^2)
    else
        kLiquid_Microbial, kLiquid_Microbial_std, Wt_Liquid_Mic = missing, missing, missing
    end
    
    push!(final_df, (
        Experiment = exp, Temperature = temp, TSS = env_params[exp].TSS, Organism = org,
        Org_Numeric = (org == "E11" ? 1.0 : 0.0),
        
        kSolid_Microbial = kSolid_Microbial,
        kSolid_Microbial_std = kSolid_Microbial_std,
        Wt_Solid_Mic = Wt_Solid_Mic,
        
        kLiquid_Microbial = kLiquid_Microbial,
        kLiquid_Microbial_std = kLiquid_Microbial_std,
        Wt_Liquid_Mic = Wt_Liquid_Mic
    ))
end

final_df.Temp_x_TSS = final_df.Temperature .* final_df.TSS
final_df.Strain_x_TSS = final_df.Org_Numeric .* final_df.TSS
final_df.Temp_x_Strain = final_df.Temperature .* final_df.Org_Numeric

# ==========================================
# 5. EXPORT
# ==========================================
output_filename = joinpath(fold, "CLEANED_FINAL_DATASET.csv")
CSV.write(output_filename, final_df)
