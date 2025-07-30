
Lbool(b) = b ? 1 : -1
@with_kw mutable struct Param{TI,TF,TB}
    Ns::TI = 0
    Nc::TI = 0
    K::TI = 0
    P::TI = 0
    ΔS::TF = 0
    f::TF = 0
    r::TF = 0
    γh::TF = 0
    σh::TF = 0
    γJ::TF = 0
    aJ::TF = 0
    bJ::TF = 0
    σJ0::TF = 0
    σlogJ::TF = 0
    σJ::TF = 0
    γW::TF = 0
    αW::TF = 0
    λ::TF = 0
    λu::TF = 0
    dt::TF = 0
    t_end::TF = 0
    model::Symbol = :model
    _corr_fit::Any = []
end


#############################################
#Return T̄ according to f̄ when using parameter
#############################################
Tf(f) = sqrt(2) * erfcinv(2f)
function Base.getproperty(the_type::Param,prop::Symbol)
    if prop == :N_t
        return round(Int,(the_type.t_end/the_type.dt))
    end
    if prop == :μJ
        return -log(the_type.K/2)/2 - the_type.σJtotal^2 #var(J)でなくmean(J^2)を正規化するため2で割らない
    end
    if prop == :corr_fit
        return getfield(the_type,:_corr_fit)
    end
    return getfield(the_type,prop)
end

function Base.setproperty!(the_type::Param, prop::Symbol, val)
    if prop == :f 
        setfield!(the_type, :f, val)
        setfield!(the_type, :_corr_fit, corr_fitting(val))
    else
        setfield!(the_type, prop, val)
    end
end

corr_trans(x,p) = 1 .- (1 .-x) .^(1/(1+exp(-p[1])))# + 1/(1+exp(p[1]))*(1 .-x)  .^(1/(1+exp(-p[3])))
function corr_fitting(f)
    tmp_correlation_ϕ(y,ρ,f) = (1/2π) * quadgk(x -> (ϕ(x - Tf(f)) - f)*(ϕ(ρ*x + sqrt(1-ρ^2)*y-Tf(f))-f) * exp(-(x^2+y^2)/2), -5,5)[1]
    correlation_ϕ(ρ,f) =  quadgk(y -> tmp_correlation_ϕ(y,ρ,f), -5,5)[1]/(f*(1-f))
    fit = curve_fit(corr_trans,0:0.01:1,[correlation_ϕ(ρ,f) for ρ in 0:0.01:1],[0.])
    return fit.param
end
p = Param{Int64,Float64,Bool}();
