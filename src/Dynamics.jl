


function simulate_poisson(p)
    λvec = [p.λ*ones(p.P); p.r]
    λtotal = sum(λvec)
    t = 0
    t_event_list = [(0.,0)]
    while true
        t = t + rand(Exponential(1/λtotal))
        if t < p.t_end
            event = rand(Categorical(λvec/λtotal))
            push!(t_event_list, (t,event))
        else
            break
        end
    end
    push!(t_event_list,(p.t_end,0))
    return t_event_list
end

function t_fixed_step(p)
    t = 0
    t_event_list = [(0.,0)]
    while true
        t = round(t + p.dt,digits=1)
        if t < p.t_end
            event = 0
            push!(t_event_list, (t,event))
        else
            break
        end
    end
    push!(t_event_list,(p.t_end,0))
    return t_event_list
end

function t_fixed_step(t_init,t_step,t_end)
    t = t_init
    t_event_list = [(t_init,0)]
    while true
        t = round(t + t_step,digits=1)
        if t < t_end
            event = 0
            push!(t_event_list, (t,event))
        else
            break
        end
    end
    push!(t_event_list,(t_end,0))
    return t_event_list
end

function merge_t_event(t_event_list1,t_event_list2)
    # merge two event list according to the first argument of each tuple by using merge sort.
    n1 = length(t_event_list1)
    n2 = length(t_event_list2)
    t_event_list = []
    i1 = 1
    i2 = 1
    while i1 <= n1 && i2 <= n2
        if t_event_list1[i1][1] < t_event_list2[i2][1]
            push!(t_event_list,t_event_list1[i1])
            i1 += 1
        else
            push!(t_event_list,t_event_list2[i2])
            i2 += 1
        end
    end
    while i1 <= n1
        push!(t_event_list,t_event_list1[i1])
        i1 += 1
    end
    while i2 <= n2
        push!(t_event_list,t_event_list2[i2])
        i2 += 1
    end
    # remove duplicated events
    n = length(t_event_list)
    i = 1
    while i < n
        if t_event_list[i][1] == t_event_list[i+1][1]
            deleteat!(t_event_list, i+1)
            n -= 1
        else
            i += 1
        end
    end
    return t_event_list
end


is_sparse(model::Symbol) = model in (:sparse_normal,:sparse_lognormal)
is_normal(model::Symbol) = model in (:dense_normal,:sparse_normal)
ele(J) = issparse(J) ? J.nzval : J
ele(J,i) = issparse(J[i]) ? J[i].nzval : J[i]

function simulate_noise(p,t_event_list)
    n_event = length(t_event_list) #t=0 and t=t_end
    #to save random computation, we make initial state and repeat it n_event times.
    h̃ = repeat(randn(p.Nc),1,n_event)
    J0 = 0
    if p.model === :sparse_lognormal
        J0 = sprand(p.Nc,p.Ns,p.K/p.Ns)*2
        #ele(J0)[:] = ones(size(ele(J0)))
    elseif p.model === :sparse_sigmoid_normal
        J0 = sprand(p.Nc,p.Ns,p.K/p.Ns)*2
    elseif p.model === :sparse_normal
        J0 = sprandn(p.Nc,p.Ns,p.K/p.Ns)
    elseif p.model === :dense_lognormal
        #J0 = p.σJ0 * randn(p.Nc,p.Ns) .- p.σJ0^2
        J0 = rand(p.Nc,p.Ns)*4
    elseif p.model === :dense_normal
        J0 = randn(p.Nc,p.Ns)
    end
    J = [similar(J0) for _ in 1:n_event]
    ele(J,1)[:] = randn(size(ele(J,1)))
    
    #Ornstein-Uhlenbeck process (Fix the variance to sigma = 1 and scale it later)
    for i_event in 1:n_event-1
        dt = t_event_list[i_event+1][1] - t_event_list[i_event][1]
        #println(t_event_list[i_event+1][1]," ",t_event_list[i_event][1])
        h̃[:,i_event+1] = h̃[:,i_event]*exp(-p.γh * dt) + sqrt(1 - exp(-2p.γh * dt)) * randn(p.Nc)
        ele(J,i_event+1)[:] = ele(J,i_event)*exp(-p.γJ * dt) + sqrt(1 - exp(-2p.γJ * dt)) * randn(size(ele(J,i_event)))
    end
    
    h̃ = h̃ * p.σh
    #Exponential transform for lognormal model.
    for i_event in 1:n_event
        if p.model === :sparse_lognormal
            ele(J,i_event)[:] = (ele(J0) .+ p.σJ * exp.(p.σlogJ * ele(J,i_event) .- p.σlogJ^2)) / sqrt(p.K)
        elseif p.model === :sparse_sigmoid_normal
            #↓OU processをexpでなくsigmoidに代入した場合
            ele(J,i_event)[:] = (ele(J0) .+ p.σJ * exp.(p.σlogJ * ele(J,i_event) .- p.σlogJ^2) ./ (1 .+ exp.(p.σlogJ * ele(J,i_event) .- p.σlogJ^2))) / sqrt(p.K)
        elseif p.model === :sparse_normal
            ele(J,i_event)[:] = (ele(J0) + p.σJ * ele(J,i_event)) / sqrt(p.K)
        elseif p.model === :dense_lognormal
            ele(J,i_event)[:] = (ele(J0) .+ p.σJ * exp.(p.σlogJ * ele(J,i_event) .- p.σlogJ^2)) / sqrt(p.Ns)
        elseif p.model === :dense_normal
            ele(J,i_event)[:] = (ele(J0) + p.σJ * ele(J,i_event)) / sqrt(p.Ns)
        end
    end
    return h̃, J
end


function simulate_C(p,t_event_list,S,h̃,J,ϕ)
    n_event = length(t_event_list)
    h = zeros(p.Nc,p.P,n_event)
    C = zeros(p.Nc,p.P,n_event)
    T = 0
    for i_event in 1:n_event
        hS = J[i_event] * S
        #subtract the mean only for lognormal distribution model
        h[:,:,i_event] = hS .- mean(hS,dims=1) * !is_normal(p.model) .+ h̃[:,i_event]
        if i_event == 1
            #Adjust T depending on the initial variance of membrane potential in response to random stimuli #1 and fix it.
            T = std(h[:,1,i_event]) * Tf(p.f)
        end
        C[:,:,i_event] = ϕ.(h[:,:,i_event] .-  T);
    end
    return h,C
end

function get_X(t_event_list)
    n_event = length(t_event_list)
    X = ones(Bool,n_event)
    X[1] = (rand() > 0.5) ? 1 : 0
    
    for i_event in 1:n_event -1
        event_t = t_event_list[i_event+1][2]
        if event_t == p.P+1
            X[i_event+1] = 1-X[i_event]
        else
            X[i_event+1] = X[i_event]
        end
    end
    return X
end

function simulate_W_error(p,t_event_list,h̃,J,S,L,ϕ)
    n_event = length(t_event_list)
    W = zeros(p.Nc,n_event)
    X = ones(Bool,n_event)
    gL = zeros(p.P,n_event)
    h = zeros(p.Nc,p.P,n_event)
    C = zeros(p.Nc,p.P,n_event)
    T = 0
    T_check = 0

    X[1] = (rand() > 0.5) ? 1 : 0

    J0 = @view(J[1]); h̃0 = @view(h̃[:,1]);
    S0 = S
    h0 = (J0 .- mean(J0,dims=1) * !is_normal(p.model)) * S0  .+ h̃0
    T = std(h0) * Tf(p.f)

    for i_event in 1:n_event-1
        dt = t_event_list[i_event+1][1] - t_event_list[i_event][1]


        event_t = t_event_list[i_event+1][2]
        if event_t == p.P+1
            X[i_event+1] = 1-X[i_event]
        else
            X[i_event+1] = X[i_event]
        end
        
        Lt = X[i_event+1] ? L : -L

        Jt = @view(J[i_event+1]); h̃t = @view(h̃[:,i_event+1]);
        #noise = rand(Bernoulli(p.ΔS/2),p.Ns)
        #St = S .⊻ noise;
        St = S

        h[:,:,i_event+1] = (Jt .- mean(Jt,dims=1) * !is_normal(p.model)) * St  .+ h̃t
        if i_event == 1
            #Adjust T depending on the initial variance of membrane potential in response to random stimuli #1 and fix it.
            #T = std(h[:,1,i_event+1]) * Tf(p.f)
            T_check = 1
        end
        C[:,:,i_event+1] = ϕ.(h[:,:,i_event+1] .-  T);

        W[:,i_event+1] = @view(W[:,i_event])*exp(-p.γW * dt)

        ################################
        #Evaluation before update of W #
        ################################
        y = C[:,:,i_event+1]' * W[:,i_event+1]/p.Nc
        gL[:,i_event+1] = y .* Lt
        
        
        ############
        #Jump term #
        ############
        if 1 <= event_t <= p.P
            W[:,i_event+1] += p.αW * Lt[event_t] * (C[:,event_t,i_event+1] .- p.f)
        end
    end
    if T_check == 0
        error("T not initialized")
    end
    return W,X,gL
end


function mean_error(p)
    t_event_list = simulate_xy(p);
    h̃, J̃ = simulate_noise(p,t_event_list);
    _,_,gL = simulate_W_error(p,t_event_list,h̃,J̃);
    i_t_half = findfirst(x->x[1]>0.5*p.t_end,t_event_list)
    h̃ = nothing; J̃ = nothing
    return mean(gL[:,end] .> 0), gL[1,end]
end

function trajectory(p,t_init,t_step,S,L,ϕ)
    t_event_list_poisson = simulate_poisson(p);
    t_event_list_fixed = t_fixed_step(t_init,t_step,p.t_end);
    t_event_list = merge_t_event(t_event_list_poisson,t_event_list_fixed);
    h̃, J = simulate_noise(p,t_event_list);
    _,X,gL = simulate_W_error(p,t_event_list,h̃,J,S,L,ϕ);
    return t_event_list,h̃,J,X,gL
end

function simulation_sample_trajectory(p,ϕ,S,do_calc; N_ensemble = 100,only_responsive = false)
        #ϕ(x) = sigmoid(2(x-1))
        #S = randn(p.Ns,p.P)

        #simulation_example
        t_event_list = t_fixed_step(p)
        h̃, J = simulate_noise(p,t_event_list)
        h,C = simulate_C(p,t_event_list,S,h̃,J,ϕ)
        if only_responsive
                C = C[sum(C[:,1,:],dims=[2])[:,1].>0,:,:]
                Nc = size(C,1)
        else
                Nc = p.Nc
        end
        corr_time = corr(cent(C[:,1,:]),Nc)[1,:]
        active_fraction = [sum(C[:,1,i_t])/Nc for i_t in 1:size(C)[3]]
        #Cit = representation_simulation(p,binary);

        #simulation_ensemble
        corr_time_ensemble = zeros(N_ensemble,length(corr_time))
        active_fraction_ensemble = zeros(N_ensemble,length(corr_time))
        corr_time_theory = []
        if N_ensemble > 0
                for i_simu = 1:N_ensemble
                        t_event_list = t_fixed_step(p)
                        h̃, J = simulate_noise(p,t_event_list)
                        h,C = simulate_C(p,t_event_list,S,h̃,J,ϕ)
                        if only_responsive
                                C = C[sum(C[:,1,:],dims=[2])[:,1].>0,:,:]
                                Nc = size(C,1)
                        else
                                Nc = p.Nc
                        end
                        corr_time_ensemble[i_simu,:] = corr(cent(C[:,1,:]),Nc)[1,:]
                        active_fraction_ensemble[i_simu,:] = [sum(C[:,1,i_t])/Nc for i_t in 1:size(C)[3]]
                end
                tlist = 0:p.dt:p.t_end
                if do_calc
                        corr_time_theory = [correlation_ϕ.((1+p.σJ^2*exp(-p.γJ * t) + p.σh^2*exp(-p.γh * t))/(1+p.σJ^2 + p.σh^2),p.f) for t in tlist]
                end
        end
        return (h,C,corr_time,active_fraction,corr_time_ensemble,active_fraction_ensemble,corr_time_theory)
end



