###########
## Fig.1 ##
###########
function plot_sample_trajectory(p,ϕ,colormap,result;only_responsive = false,ylim_active_fraction = (0,0.25),legend_active_fraction = :topright)
        palette = cgrad(colormap,6,categorical=true)
        h,C,corr_time,active_fraction,corr_time_ensemble,active_fraction_ensemble,corr_time_theory = result
        if only_responsive
                C = C[sum(C[:,1,:],dims=[2])[:,1].>0,:,:]
                Nc = size(C,1)
        else
                Nc = p.Nc
        end
        println(Nc)
        #Plot of each neuron's response over time
        #perm = sortperm(C[:,1,1])
        life_frac = sum(C[:,1,:],dims=2)[:,1]
        perm = sortperm(C[:,1,:] * (1:size(C)[3]) ./ life_frac)
        tlist = 0:p.dt:p.t_end
        p1 = heatmap(tlist,1:Nc,C[perm,1,:],
        xlabel="\$t\$",
        ylabel="Hidden neuron ID",
        colormap = colormap,
        colorbar=false)

        #Plot of decorrelation profile
        #plot(tlist,corr_time_ensemble',color=palette[1],label="")
        plot(tlist,corr_time,color=palette[5],label="sample",linewidth = 2,linestyle=:dot) 
        plot!(tlist,mean(corr_time_ensemble,dims=1)',ribbon = std(corr_time_ensemble,dims=1)',color=palette[3],label="",linewidth=3)
        plot!(tlist,mean(corr_time_ensemble,dims=1)',color=palette[4],label="mean",linewidth=3)
        plot!(tlist,corr_time_theory,color=:black,label="theory",linestyle=:dash)
        p2 = plot!(
                xlabel = "\$t\$",
                ylabel="Autocorrelation",
                ylim = (0,1))


        #Plot of active neuron fraction
        #plot(tlist,active_fraction_ensemble',color=:gray,label="")
        plot(tlist,active_fraction,color=palette[5],label="sample",linewidth=2.,linestyle=:dot) 
        plot!(tlist,mean(active_fraction_ensemble,dims=1)',ribbon = std(active_fraction_ensemble,dims=1)',color=palette[3],label="",linewidth=2)
        plot!(tlist,mean(active_fraction_ensemble,dims=1)',color=palette[4],label="mean",linewidth=3)
        plot!(tlist,p.f*ones(length(tlist)),color=:black,label="theory",linestyle=:dash)
        p3 = plot!(
                xlabel="\$t\$",
                ylim=ylim_active_fraction,
                ylabel="Active fraction",
                legend=legend_active_fraction)

        #Plot above three at once
        plot(p1,p2,p3,layout=grid(3,1,heights=[0.6,0.2,0.2]),size=(300,300))
end

function plot_decorrelation!(p,ϕ,color_ribbon,color_mean,color_theory,label,corr_time_ensemble,corr_time_theory)
    tlist = 0:p.dt:p.t_end
    plot!(tlist,mean(corr_time_ensemble,dims=1)',ribbon = std(corr_time_ensemble,dims=1)',color=color_ribbon,label="",linewidth=3)
    plot!(tlist,mean(corr_time_ensemble,dims=1)',color=color_mean,label=label,linewidth=3)
    plot!(tlist,corr_time_theory,color=:black,label="",linestyle=:dash)
    return p2 = plot!(
            xlabel = "\$t\$",
            ylabel="Autocorrelation",
            ylim = (0,1))
end

#############
## Fig.2,3 ##
#############

cbar(cdata,cticks,clabel,cmap) = begin
    cbar_fig = heatmap(
      [1],cdata,
      cdata .* ones(length(cdata), 1),
      c=cmap,
      legend=:none,
      xticks=:none,
      yticks=:none,
      formatter=:scientific,
    )
    axes_fig = twinx(cbar_fig)
    plot!(axes_fig,
      [1],cdata,
      legend=:none,
      xticks=:none,
      yticks=cticks,
      ylabel=clabel,
    )
end

function plot_tuning_similarity(C,θ,perm1,perm2,cmap,p,colorbar,put_ylabel,title,put_colorbar;xylim = (-5,5))
    θticks = ([0:π:3*π;], ["0", "\\pi", "2\\pi"])
    ylabel = put_ylabel ? "Hidden neuron ID\nselected & sorted\nat \$t=0\$" : ""
    yticks = 0:100:length(perm1)
    p_perm1 = heatmap(θ,1:length(perm1), C[perm1,:],colormap = cmap,xticks = θticks,yticks = yticks,colorbar=colorbar,ylabel=ylabel,title=title,tick_direction=:none)
    ylabel = put_ylabel ? "Hidden neuron ID\n selected & sorted \nat \$t=3\\tau\$" : ""
    yticks = 0:100:length(perm2)
    p_perm2 = heatmap(θ,1:length(perm2),C[perm2,:],colormap = cmap,xticks = θticks,yticks = yticks,colorbar=colorbar,xlabel="Phase",ylabel=ylabel,tick_direction=:none)
    
    
    
    title_null = ""
    p_PCA = plot_PCA(C,θ,p,colorbar,put_colorbar,title_null,xylim) 
    p_similarity = plot_similarity(C,θ,p,colorbar,put_colorbar,put_ylabel,title_null)
    return p_perm1,p_perm2,p_PCA,p_similarity
end

function plot_PCA(C,θ,p,colorbar,put_colorbar,title,xylim)
    pca = fit(PCA,C;maxoutdim=2)
    Y = MultivariateStats.transform(pca,C)'
    s_PCA = scatter(Y[:,1],Y[:,2],zcolor=θ,color=:twilight,xlim=xylim,ylim=xylim,legend=false,colorbar=colorbar,aspect_ratio = 1.,markerstrokewidth = 0,xlabel="PC1",ylabel="PC2",title=title,
    xticks = [-5,0,5],yticks = [-5,0,5])
    if put_colorbar
        θticks = ([0:π:3*π;], ["0", "\\pi", "2\\pi"])
        c_PCA = cbar(0:0.001:2π,θticks,"Phase",:twilight)
        p_PCA = plot(s_PCA,c_PCA,layout=@layout([grid(1, 1) a{0.03w}]))
    else
        p_PCA = s_PCA
    end
    return p_PCA
end

function plot_similarity(C,θ,p,colorbar,put_colorbar,put_ylabel,title)
    ylabel = put_ylabel ? "Phase" : ""
    θticks = ([0:π:3*π;], ["0", "\\pi", "2\\pi"])
    h_similarity = heatmap(θ,θ, cor(C),colormap = :viridis,clim=(0,1),xticks=θticks,yticks=θticks,colorbar_ticks=([0:0.5:1],["0",".5","1"]),xlabel="Phase",ylabel=ylabel,colorbar=colorbar,aspect_ratio = 1.,xlim=(0,2π),ylim=(0,2π),title=title,tick_direction=:none)
    if put_colorbar
        c_similarity = cbar(0:0.001:1,([0,0.5,1],[0,0.5,1]),"Similarity",:viridis)
        p_similarity = plot(h_similarity,c_similarity,layout=@layout([grid(1, 1) a{0.03w}]))
    else
        p_similarity = h_similarity
    end
    return p_similarity
end

function perm_tuning(X,θ)
    Xvec = X * [cos.(θ) sin.(θ)]
    Xcomp = Xvec[:,1] + Xvec[:,2]*im
    Xangle = angle_correct.(angle.(Xcomp))
    perm = sortperm(Xangle)
    #zero_index = findfirst(sum(X[perm,:],dims=2).==0)[1]
    remove_index = sum(X[perm,:],dims=2).==0 #.| (sum(X[perm,:],dims=2).==length(θ))
    return perm[.!remove_index[:,1]]
end
function perm_tuning_2(h,C,θ)
    Xvec = h * [cos.(θ) sin.(θ)]
    Xcomp = Xvec[:,1] + Xvec[:,2]*im
    Xangle = angle_correct.(angle.(Xcomp))
    perm = sortperm(Xangle)
    #zero_index = findfirst(sum(X[perm,:],dims=2).==0)[1]
    remove_index = sum(C[perm,:],dims=2).==0 #.| (sum(X[perm,:],dims=2).==length(θ))
    return perm[.!remove_index[:,1]]
end

###########
## Fig.4 ##
###########

function plot_selectivity_space(h_A,h_B,colorbar=false;color = :black)
    selectivity = h_A[:,1] - h_B[:,1]
    markersize = 1.5
    lim = (-8.,8.)
    clim = (-3,3)
    labelA = "Response to phase\$=0\$"
    labelB = "Response to phase\$=\\pi\$"
    #p1 = scatter(h_A[:,1],h_B[:,1],zcolor=selectivity,aspect_ratio = 1,markerstrokewidth=0,figsize=(400,400),color=color,ylim=lim,xlim=lim,xlabel=labelA,ylabel=labelB,markersize=markersize,clim = clim,title="\$t=0\$",permute=(:x,:y))
    #p2 = scatter(h_A[:,end],h_B[:,end],zcolor=selectivity,aspect_ratio = 1,markerstrokewidth=0,figsize=(400,400),color=color,ylim=lim,xlim=lim,xlabel=labelA,ylabel=labelB,markersize=markersize,clim=clim,title="\$t=3\\tau\$")
    p1 = scatter(h_A[:,1],h_B[:,1],aspect_ratio = 1,markerstrokewidth=0,figsize=(400,400),color=color,ylim=lim,xlim=lim,xlabel=labelA,ylabel=labelB,markersize=markersize,clim = clim,title="\$t=0\$")
    println(h_A[1,1],h_A[1,end])
    p2 = scatter(h_A[:,end],h_B[:,end],aspect_ratio = 1,markerstrokewidth=0,figsize=(400,400),color=color,ylim=lim,xlim=lim,xlabel=labelA,ylabel="",markersize=markersize,clim=clim,title="\$t=3\\tau\$")
    sparse = rand(p.Nc) .< 0.05
    #scatter(h_A[sparse,1],h_B[sparse,1],markersize=0.5)
    #quiver(h_A[sparse,1],h_B[sparse,1],quiver = (h_A[sparse,end]-h_A[sparse,1],h_B[sparse,end]-h_B[sparse,1]),line_z = repeat(selectivity[sparse],inner=4),c=color)
    quiver(h_A[sparse,1],h_B[sparse,1],quiver = (h_A[sparse,end]-h_A[sparse,1],h_B[sparse,end]-h_B[sparse,1]),c="#017100")
    p3 = plot!(aspect_ratio = 1,ylim=lim,xlim=lim,clim=clim,title= "Change from \$t=0\$ to \$3\\tau\$",xlabel=labelA,ylabel=labelB,colorbar_title = "\$h^{A}-h^{B}\\ \\mathrm{at}\\ t=0\$")
    plot(p1,p2,p3,layout=(1,3),size=(1200,400),legend=false,colorbar=colorbar,rightmargin=5Plots.mm)
end


###########
## Fig.5 ##
###########
function plot_alternating_regions!(p,X0,timepoints; fillcolor=:gray, fillrange=(NaN,NaN), alpha=0.3)
    # Input validation
    if length(timepoints) < 2
        error("Need at least 2 timepoints")
    end
    
    # Calculate number of complete pairs
    if X0 == 0
        n_pairs = floor(Int, length(timepoints)/2 - 1)
    else
        n_pairs = ceil(Int, length(timepoints)/2 - 1/2)
    end
    x_fill = []
    
    # Add filled regions for each pair
    for k in 0:n_pairs
        if X0 == 0
            idx_start = 2k + 1
            idx_end = 2k + 2
        else
            idx_start = 2k
            idx_end = 2k+1
        end
        
        # Skip if we don't have a complete pair
        if idx_end > length(timepoints)
            break
        end
        
        # Create fill coordinates
        if idx_start == 0
            x_fill = [0, timepoints[idx_end]]
        else
            x_fill = [timepoints[idx_start], timepoints[idx_end]]
        end
        y_fill = [0, 0]
        
        # Add region to plot
        plot!(x_fill, y_fill,
            fillrange = fillrange,
            fillalpha = alpha,
            fillcolor = fillcolor,
            label = ""  # No legend entry
        )
    end
    return p
end

function plot_stimuli(p)
    

    n_repeat = 3
    colors = ["#00AB8E","#FF644E"]
    shapes = [:circle,:circle]

    plots = Vector{Any}(undef,n_repeat)
    X_event = 1
    for i_repeat = 1:n_repeat
        t_event_list_poisson = simulate_poisson(p);
        t_event_list_fixed = t_fixed_step(0,1,p.t_end);
        t_event_list = merge_t_event(t_event_list_poisson,t_event_list_fixed);
        X = get_X(t_event_list)

        i_event = [[] for i in 1:p.P+2]
        t_event = [[] for i in 1:p.P+2]
        X_event = [[] for i in 1:p.P+2]
        for i in 1:length(t_event_list)
            push!(i_event[t_event_list[i][2]+1], i)
            push!(t_event[t_event_list[i][2]+1], t_event_list[i][1])
            push!(X_event[t_event_list[i][2]+1], X[i])
        end
        t_all = [t_event_list[i][1] for i in 1:length(t_event_list)];

        plot()
        plot_alternating_regions!(p,X[1],t_event[p.P+2])
        for i in 1:p.P
            scatter!(t_event[i+1],i*ones(length(t_event[i+1])),color = [colors[X_event[i+1][j] ⊻ (L[i] == +1) + 1] for j in 1:length(X_event[i+1])],markerstrokewidth=0,legend=false,markershape = shapes[i],markersize = 3)
        end
        #plot!(t_all,X)
        if i_repeat == n_repeat
            plot!(xlabel = "\$t\$")
        end
        plots[i_repeat] = plot!(
            ylim = (0,p.P+1),
            xlim = (0,10),
            yticks = (1:p.P, ["Stim$(i)" for i in 1:p.P]),
        )
    end
    plot(plots...,layout = (3,1),size=(350,250))
end


function file2PI(filename,ρLmn,ρSmn)
    jlddata = load("../data/$(filename).jld2")
    gL_λσ,p,S,L,γlist,σlist,Nclist,λlist = jlddata["gL_λσ"],jlddata["p"],jlddata["S"],jlddata["L"],jlddata["γlist"],jlddata["σlist"],jlddata["Nclist"],jlddata["λlist"]

    PI_boot_γσ = PI_boot_gL_list.(gL_λσ)
    #switched the plot order of J and h
    PI_boot_γσ = PI_boot_γσ[2:-1:1,:,:,:,:]

    PI_analytical = zeros(2,length(γlist),length(σlist),length(Nclist),length(λlist))

    for i_γ in 1:length(γlist), i_σ in 1:length(σlist), i_Nc in 1:length(Nclist), i_λ in 1:length(λlist)
        p.γh = γlist[i_γ]; p.γJ = γlist[i_γ]; p.Nc = Nclist[i_Nc]; p.λ = λlist[i_λ]
        
        p.σh = 0.; p.σJ = σlist[i_σ]
        PI_analytical[1,i_γ,i_σ,i_Nc,i_λ] = PI(p,1,ρLmn,ρSmn)

        p.σh = σlist[i_σ]; p.σJ = 0.
        PI_analytical[2,i_γ,i_σ,i_Nc,i_λ] = PI(p,1,ρLmn,ρSmn)

    end
    return PI_boot_γσ, PI_analytical,σlist,Nclist,λlist,γlist
end


function file2PI_infs(filename,ρLmn,ρSmn)
    jlddata = load("../data/$(filename).jld2")
    gL_λσ,p,S,L,γlist,σlist,Nclist,λlist = jlddata["gL_λσ"],jlddata["p"],jlddata["S"],jlddata["L"],jlddata["γlist"],jlddata["σlist"],jlddata["Nclist"],jlddata["λlist"]

    PI_boot_γσ = PI_boot_gL_list.(gL_λσ)
    #switched the plot order of J and h
    PI_boot_γσ = PI_boot_γσ[2:-1:1,:,:,:,:]

    PI_analytical = zeros(2,length(γlist),length(σlist),length(Nclist),length(λlist))

    for i_γ in 1:length(γlist), i_σ in 1:length(σlist), i_Nc in 1:length(Nclist), i_λ in 1:length(λlist)
        p.γh = γlist[i_γ]; p.γJ = γlist[i_γ]; p.Nc = Nclist[i_Nc]; p.λ = λlist[i_λ]
        
        p.σh = 0.; p.σJ = σlist[i_σ]
        PI_analytical[1,i_γ,i_σ,i_Nc,i_λ] = PI_inf(p,1,ρLmn,ρSmn)

        p.σh = σlist[i_σ]; p.σJ = 0.
        PI_analytical[2,i_γ,i_σ,i_Nc,i_λ] = PI_inf(p,1,ρLmn,ρSmn)

    end
    return PI_boot_γσ, PI_analytical,σlist,Nclist,λlist,γlist
end


function file2PI_inf(filename,ρLmn,ρSmn)
    jlddata = load("../data/$(filename).jld2")
    gL_λσ,p,S,L,γlist,σlist,Nclist,λlist = jlddata["gL_λσ"],jlddata["p"],jlddata["S"],jlddata["L"],jlddata["γlist"],jlddata["σlist"],jlddata["Nclist"],jlddata["λlist"]

    PI_analytical_inf = zeros(2,length(γlist))
    i_λ = 1; i_σ = 1;i_Nc = 1
    for i_γ in 1:length(γlist)
        p.γh = γlist[i_γ]; p.γJ = γlist[i_γ]; p.Nc = 0; p.λ = λlist[i_λ]

        p.σh = 0.; p.σJ = σlist[i_σ]
        PI_analytical_inf[1,i_γ] = PI_inf(p,1,ρLmn,ρSmn)

        p.σh = σlist[i_σ]; p.σJ = 0.
        PI_analytical_inf[2,i_γ] = PI_inf(p,1,ρLmn,ρSmn)

    end
    return PI_analytical_inf
end

function PIplot(PI_bootstrap, PI_analytical, para_list, para_name,d_palette,γlist; PI_analytical_inf = [])
    plots = Vector{Any}(undef,2)
    palette = [cgrad(:Blues,6,categorical=true),cgrad(:Oranges,6,categorical=true)]
    index_γτ = 1:length(γlist)
    for i_Jh in 1:2
        plot()
        for i_para in 1:length(para_list)
            color = palette[i_Jh][i_para+d_palette]
            scatter!(
                γlist[index_γτ],
                (1 .-mean.(PI_bootstrap[i_Jh,index_γτ,i_para]))/2,
                yerr = std.(PI_bootstrap[i_Jh,index_γτ,i_para]),
                markercolor=color,
                linecolor=color,
                markerstrokecolor=color,
                label="\$$(para_name)=$(para_list[i_para])\$"
                )
            plot!(
                γlist,
                (1 .-PI_analytical[i_Jh,index_γτ,i_para])/2,
                color = color,
                label="")
        end

        if !isempty(PI_analytical_inf)
            plot!(
                γlist,
                (1 .- PI_analytical_inf[i_Jh,index_γτ])/2,
                color = "black",
                label = "\$N_{C}=\\infty\$"
                )
        end
            
        plot!(γlist,0.5ones(length(γlist)),color="gray",linestyle=:dash,label="")
        plot!(
            xscale = :log10,
            xlabel = "\$\\gamma\$",
            ylabel = "Error",
            ylim = (0,0.55)
            )
        plots[i_Jh] = plot!()
    end
    return plot(plots...,size = (600,300))
end

function plot_normality_g(filename)
    jlddata = load("../data/$(filename).jld2")
        gL_λσ,p,S,L,γlist,σlist,Nclist,λlist = jlddata["gL_λσ"],jlddata["p"],jlddata["S"],jlddata["L"],jlddata["γlist"],jlddata["σlist"],jlddata["Nclist"],jlddata["λlist"]

        #println(γlist)
    palette = [cgrad(:Oranges,6,categorical=true),cgrad(:Blues,6,categorical=true)]
    i_γ_sublist = 1:2:length(γlist)
    γ_sublist = γlist[i_γ_sublist]
    p_array = Array{Any}(undef,2,length(σlist),length(γ_sublist))
    for i_Jh in 1:2, i_γ in 1:length(γ_sublist),i_σ in 1:length(σlist)
        xmin, xmax = extrema(gL_λσ[i_Jh,i_γ_sublist[i_γ],i_σ,1,1][1,:])
        xminmax = mean(abs.([xmin,xmax]))
        ticks = range(round(-xminmax, digits=1), round(+xminmax, digits=1), length=3)
        h = histogram(
            gL_λσ[i_Jh,i_γ_sublist[i_γ],i_σ,1,1][1,:],
            title="\$\\sigma=$(σlist[i_σ]),\\gamma = 10^{$(log10(γlist[i_γ_sublist[i_γ]]))}\$",
            titlefontsize = 10,
            normalize=:pdf,
            xticks = ticks,
            lw = 0,
            color=palette[i_Jh][4])#,xlim = (-0.6,0.6),ylim = (0,20))
        ymax = floor(Plots.ylims()[2])
        plot!(yticks = ([0,ymax],[0,ymax]))
        if i_γ == length(γ_sublist)
            plot!(xlabel = "\$g^{1}(t_{\\mathrm{end}})\$")
        end
        if i_σ == 1
            plot!(ylabel = "density")
        end
        p_array[i_Jh,i_σ,i_γ] = plot!()
    end
    return p_array,γ_sublist,σlist
end