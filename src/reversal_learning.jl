function reversal_learning(p,N_simu)
    t_event_list = Vector{Any}(undef,N_simu)
    gL = Vector{Any}(undef,N_simu);
    X = Vector{Any}(undef,N_simu);
    W = Vector{Any}(undef,N_simu);
    Lt = Vector{Any}(undef,N_simu);
    #println("γh: ", p.γh, " γJ: ", p.γJ, " σh: ", p.σh, " σJ", p.σJ)
    for i in 1:N_simu
        #if i%100 == 0 println(i," : ",now()) end
        t_event_list[i],h̃,J̃, X[i],gL[i] = trajectory(p,1/p.γW,1,S,L,ϕ)
    end

    t_event = [[[] for i in 1:p.P+2] for j in 1:N_simu]
    i_event = [[[] for i in 1:p.P+2] for j in 1:N_simu]
    t_all = Vector{Any}(undef,N_simu)
    i_simu = 1
    for i_simu in 1:N_simu
        for i in 1:length(t_event_list[i_simu])
            push!(i_event[i_simu][t_event_list[i_simu][i][2]+1], i)
            push!(t_event[i_simu][t_event_list[i_simu][i][2]+1], t_event_list[i_simu][i][1])
        end
        t_all[i_simu] = [t_event_list[i_simu][i][1] for i in 1:length(t_event_list[i_simu])];
    end
    
    gL_array = stack([gL[i_simu][:,i_event[i_simu][1][2:end]] for i_simu in 1:N_simu])
    #return gL_array
    return gL_list = reshape(gL_array,size(gL_array)[1], size(gL_array)[2]*size(gL_array)[3])
end

PI(gL_list) = 1 - 2mean(gL_list .< 0)
function PI_boot_gL_list(gL_list)
    N_smp = 1000
    gL_boot = [sample(gL_list,length(gL_list)) for i_smp in 1:N_smp]
    PI_boot = PI.(gL_boot)
end

function PI_boot_gL_list(gL_list,m)
    N_smp = 1000
    gL_boot = [sample(gL_list[m,:],length(gL_list[m,:])) for i_smp in 1:N_smp]
    PI_boot = PI.(gL_boot)
end


#QuadGKを使う場合
rtol = 1e0
ρL(t,s,p,ρmn) = ρmn * exp(-2p.r*abs(t-s))
ρL_bias(t,s,p,ρmn,bias) = ρmn * (bias + (1-bias)* exp(-2p.r*abs(t-s)))
ρh(t,s,p,ρS) = (ρS*(1+p.σJ^2*exp(-p.γJ*abs(t-s))) + p.σh^2*exp(-p.γh*abs(t-s)))/(1+p.σJ^2+p.σh^2)
ρC(t,s,p,ρS) = corr_trans(ρh(t,s,p,ρS),p.corr_fit)
function CovC(p,t,s,u,m,n,k,ρSmn)
      T = -quantile(Normal(), p.f)
      Σ = ones(3,3)
      tlist = [t,s,u]
      mlist = [m,n,k]
      for i in 1:3, j in 1:3
            if i == j continue end
            Σ[i,j] = ρh(tlist[i],tlist[j],p,ρSmn[mlist[i],mlist[j]])
      end
      #
      return trivariate_normal_cdf_monte_carlo(T,Σ) - p.f^3 - p.f^2*(1-p.f) * (ρC(t,s,p,ρSmn[m,n]) + ρC(t,u,p,ρSmn[m,k])) - p.f^2 * (1 - p.f)^2 * ρC(t,s,p,ρSmn[m,n]) * ρC(t,u,p,ρSmn[m,k])
end


Esq(p,m,ρLmn,ρSmn) = (p.λ * quadgk(s->
      sum(ρL(p.t_end,s,p,ρLmn[m,n]) * ρC(p.t_end,s,p,ρSmn[m,n]) * exp(-p.γW*(p.t_end-s)) for n in 1:size(ρLmn,2))
      ,0,p.t_end,rtol=rtol)[1])^2 * (p.f*(1-p.f))^2

Esq_bias(p,m,ρLmn,ρSmn,bias) = (p.λ * quadgk(s->
      sum(ρL_bias(p.t_end,s,p,ρLmn[m,n],bias) * ρC(p.t_end,s,p,ρSmn[m,n]) * exp(-p.γW*(p.t_end-s)) for n in 1:size(ρLmn,2))
      ,0,p.t_end,rtol=rtol)[1])^2 * (p.f*(1-p.f))^2

V1(p,m,ρLmn,ρSmn) = p.λ*quadgk(s->
      sum(ρC(p.t_end,s,p,ρSmn[m,n])^2 * exp(-2p.γW*(p.t_end-s)) for n in 1:size(ρLmn,2))
      ,0,p.t_end,rtol=rtol)[1] * (p.f*(1-p.f))^2

VL(p,m,ρLmn,ρSmn) = p.λ*quadgk(s->
      sum(ρL(p.t_end,s,p,ρLmn[m,n])^2 for n in 1:size(ρLmn,2))
      ,0,p.t_end,rtol=rtol)[1]
      
V2(p,m,ρLmn,ρSmn) = p.λ^2 * quadgk(s->quadgk(u->
      sum(exp(-p.γW * (2p.t_end-s-u))*
      (ρL(s,u,p,ρLmn[n,k]) - ρL(p.t_end,s,p,ρLmn[m,n])*ρL(p.t_end,u,p,ρLmn[m,k])) * 
      ρC(p.t_end,s,p,ρSmn[m,n]) * ρC(p.t_end,u,p,ρSmn[m,k])
      for n in 1:size(ρLmn,2), k in 1:size(ρLmn,2))
      ,0,p.t_end,rtol=rtol)[1], 0,p.t_end,rtol=rtol)[1] * (p.f*(1-p.f))^2

V2_approx(p,m,ρLmn,ρSmn) = (p.λ^2/p.r)*quadgk(s->
    sum(ρLmn[n,k] * (1-exp(-4p.r*(p.t_end-s))) * ρC(p.t_end,s,p,ρSmn[m,n]) * ρC(p.t_end,s,p,ρSmn[m,k]) * exp(-2p.γW*(p.t_end-s))
    for n in 1:size(ρLmn,2), k in 1:size(ρLmn,2))
    ,0,p.t_end,rtol=rtol)[1] * (p.f*(1-p.f))^2 - Esq(p,m,ρLmn,ρSmn)

V2_bias(p,m,ρLmn,ρSmn,bias) = p.λ^2 * quadgk(s->quadgk(u->
      sum(exp(-p.γW * (2p.t_end-s-u))*
      (ρL_bias(s,u,p,ρLmn[n,k],bias) - ρL_bias(p.t_end,s,p,ρLmn[m,n],bias)*ρL_bias(p.t_end,u,p,ρLmn[m,k],bias)) * 
      ρC(p.t_end,s,p,ρSmn[m,n]) * ρC(p.t_end,u,p,ρSmn[m,k])
      for n in 1:size(ρLmn,2), k in 1:size(ρLmn,2))
      ,0,p.t_end,rtol=rtol)[1], 0,p.t_end,rtol=rtol)[1] * (p.f*(1-p.f))^2


#Lの相関行列が単位行列の場合
V2_uncorr(p,m) = p.λ^2 * quadgk(s->quadgk(u->
      sum(exp(-p.γW * (2p.t_end-s-u))*
      (ρL(s,u,p,1) - ρL(p.t_end,s,p,ρLmn[m,n])*ρL(p.t_end,u,p,ρLmn[m,n])) * 
      ρC(p.t_end,s,p,ρSmn[m,n]) * ρC(p.t_end,u,p,ρSmn[m,n])
      for n in 1:p.P)
      ,0,p.t_end,rtol=rtol)[1], 0,p.t_end,rtol=rtol)[1] * (p.f*(1-p.f))^2


#V2_approx(p,m,ρLmn,ρSmn) = V1(p,m,ρLmn,ρSmn)*(p.λ/(p.r)) - Esq(p,m,ρLmn,ρSmn)

V3(p,m,ρLmn,ρSmn) = p.λ * quadgk(s->
      sum(exp(-2p.γW*(p.t_end-s)) * (p.f + ρC(p.t_end,s,p,ρSmn[m,n]) * (1 - p.f * (2 + ρC(p.t_end,s,p,ρSmn[m,n])))) for n in 1:size(ρLmn,2))
      ,0,p.t_end,rtol=rtol)[1] * p.f*(1-p.f) / p.Nc
V4(p,m,ρLmn,ρSmn) = p.λ^2 * quadgk(s->quadgk(u->
      sum(exp(-p.γW * (2p.t_end-s-u))*
      ρL(s,u,p,ρLmn[n,k]) * 
      CovC(p,p.t_end,s,u,m,n,k,ρSmn)/p.Nc
      for n in 1:size(ρLmn,2), k in 1:size(ρLmn,2))
      ,0,p.t_end,rtol=rtol)[1], 0,p.t_end,rtol=rtol)[1]

V4_bias(p,m,ρLmn,ρSmn,bias) = p.λ^2 * quadgk(s->quadgk(u->
      sum(exp(-p.γW * (2p.t_end-s-u))*
      ρL_bias(s,u,p,ρLmn[n,k],bias) * 
      CovC(p,p.t_end,s,u,m,n,k,ρSmn)/p.Nc
      for n in 1:size(ρLmn,2), k in 1:size(ρLmn,2))
      ,0,p.t_end,rtol=rtol)[1], 0,p.t_end,rtol=rtol)[1]

#Lの相関行列が単位行列の場合
V4_uncorr(p,m,ρSmn) = p.λ^2 * quadgk(s->quadgk(u->
      sum(exp(-p.γW * (2p.t_end-s-u))*
      ρL(s,u,p,1) * 
      CovC(p,p.t_end,s,u,m,n,n,ρSmn)/p.Nc
      for n in 1:p.P)
      ,0,p.t_end,rtol=rtol)[1], 0,p.t_end,rtol=rtol)[1]
V4_approx(p,m,ρLmn,ρSmn) = 0*V3(p,m,ρLmn,ρSmn) * (p.λ/(p.r))
PI(p,m,ρLmn,ρSmn) = 1 - 2H(sqrt(Esq(p,m,ρLmn,ρSmn)/(V1(p,m,ρLmn,ρSmn) + V2(p,m,ρLmn,ρSmn) + V3(p,m,ρLmn,ρSmn) + V4(p,m,ρLmn,ρSmn))))

PI_bias(p,m,ρLmn,ρSmn,bias) = 1 - 2H(sqrt(Esq_bias(p,m,ρLmn,ρSmn,bias)/(V1(p,m,ρLmn,ρSmn) + V2_bias(p,m,ρLmn,ρSmn,bias) + V3(p,m,ρLmn,ρSmn) + V4_bias(p,m,ρLmn,ρSmn,bias))))
PI_uncorr(p,m,ρLmn,ρSmn) = 1 - 2H(sqrt(Esq(p,m,ρLmn,ρSmn)/(V1(p,m,ρLmn,ρSmn) + V2_uncorr(p,m) + V3(p,m,ρLmn,ρSmn) + V4_uncorr(p,m,ρSmn))))

PI_approx(p,m,ρLmn,ρSmn) = 1 - 2H(sqrt(Esq(p,m,ρLmn,ρSmn)/(V1(p,m,ρLmn,ρSmn) + V2_approx(p,m,ρLmn,ρSmn) + V3(p,m,ρLmn,ρSmn) + V4_approx(p,m,ρLmn,ρSmn))))
PI_inf(p,m,ρLmn,ρSmn) = 1 - 2H(sqrt(Esq(p,m,ρLmn,ρSmn)/(V1(p,m,ρLmn,ρSmn) + V2(p,m,ρLmn,ρSmn))))
PI_inf_approx(p,m,ρLmn,ρSmn) = 1 - 2H(sqrt(Esq(p,m,ρLmn,ρSmn)/(V1(p,m,ρLmn,ρSmn) + V2_approx(p,m,ρLmn,ρSmn))))

using Distributions
using LinearAlgebra

function bivariate_normal_cdf_monte_carlo(T, ρ, n_samples=10_000)
    if abs(ρ - 1.0) < 1e-10
        #println("uni")
        return fT(T)
    else
        #println("bi")   
        d = MvNormal([0.0, 0.0], [1.0 ρ; ρ 1.0])
        samples = rand(d, n_samples)
        #error("bi ", ρ," ", sum(all(samples .> T, dims=1)) / n_samples)
        return sum(all(samples .> T, dims=1)) / n_samples
    end
end

function trivariate_normal_cdf_monte_carlo(T, Σ, n_samples=10_000)
    # Check for duplicated variables by comparing correlation coefficients
    if abs(Σ[2,3] - 1.0) < 1e-10  # C2 = C3
        return bivariate_normal_cdf_monte_carlo(T, Σ[1,2], n_samples)
    elseif abs(Σ[1,2] - 1.0) < 1e-10  # C1 = C2
        return bivariate_normal_cdf_monte_carlo(T, Σ[1,3], n_samples)
    elseif abs(Σ[1,3] - 1.0) < 1e-10  # C1 = C3
        return bivariate_normal_cdf_monte_carlo(T, Σ[1,2], n_samples)
    else
        # No duplicates, proceed with trivariate case
        #println("tri")
        d = MvNormal(zeros(3), Σ)
        samples = rand(d, n_samples)
        return sum(all(samples .> T, dims=1)) / n_samples
    end
end