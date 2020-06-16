export VQEtrain


"""
    VQEtrain(set::Union{Array{Int64,1}, Array{Float64,1}}; 
            depth=4, niter=100, autoTrain=true, niterMax = 5000, Optimizer=:NADAM, 
            StopTH = 2e-9, PerturbTH = 1e-7, Threshold = 0.5, PurtAmp = 0.05, ConvergeTH=1e-4)
    ->
    [Hvalues, isPerturbed, dEs, iConverge, EConverge, ECmin, ECmm]
The training function for VQE4NPP.
    \n== Arguments ==
    \n`set::Union{Array{Int64,1}, Array{Float64,1}}`: The target number set.
    \n`depth::Int64`: The depth (number of layers) of the quantum differentiable circuit.
    \n`niter::Int64`: The number of ierations.
    \n`autoTrain::Bool`: The option to enable auto training iterations.
    \n`niterMax::Float64`: The maximal number of ierations when applying `autoTain`.
    \n`Optimizer::Symbol`: The optimizer for gradient descent method. The default optimizer is NADAM.
    \n`StopTH::Float64`: The threshold of the derivative of the training curve to stop training. 
    \n`PerturbTH::Float64`: The threshold of the derivative of the training curve to apply perturbations.
    \n`Threshold::Float64`: The threshold for determining whether the <H> is small enough.  
    \n`PurtAmp::Float64`: The amplitude for perurbation.
    \n`ConvergeTH::Float64`: The threshold of the derivative of the training curve to determine whther the training curve has converged.
    \n== Output variables ==
    \n`Hvalues::Array{Float64,1}`: The <H> of each training step.
    \n`isPerturbed::Array{Float64,1}`: The mark of whether the <H> of ith step is gained under the condition where the differentiable circuit is perturbed. 1 means perturbed. 
    \n`dEs::Array{Float64,1}`: The derivative of the training curve (<H>) of each training step. 
    \n`iConverge::Int64`: The step when the curve converges.
    \n`EConverge::Float64`: The convergent energy corresponding to `iConverge`.
    \n`ECmin::Float64`: After the curve starts converging, the minimal value of <H> before the training stops.
    \n`ECmm::Float64`: A mean value of ECmin and its adjacent training values.
"""
function VQEtrain(set::Union{Array{Int64,1}, Array{Float64,1}}; 
                  depth::Int64=4, 
                  niter::Int64=100, 
                  autoTrain::Bool=true, 
                  niterMax::Int64=5000,
                  Optimizer::Symbol=:NADAM, 
                  StopTH::Float64=1e-8, 
                  PerturbTH::Float64=1e-7, 
                  Threshold::Float64=0.5, 
                  PurtAmp::Float64=0.02, 
                  ConvergeTH::Float64=1e-3,
                  showSteps::Bool=true)
    
    H = HofNPP(set)
    n = nqubits(H)
    circuit = variational_circuit(n,depth)
    dispatch!(circuit, :random)
    i = 0
    k = 0
    h = 5
    di = 2
    sign = 1
    iConverge = 0
    EConverge = 0
    Hvalues = Float64[]
    len2Grad = 2h+di 
    dEs = repeat([NaN],len2Grad)
    isPerturbed = zeros(len2Grad*2)
    if Optimizer == :ADAM
        GM = ()->ADAM()
    elseif Optimizer == :NADAM
        GM = ()->NADAM()
    elseif Optimizer == :AMSGrad
        GM = ()->AMSGrad()
    else
        GM = Optimizer
    end
   
    if autoTrain == true
        iDisplay = (i) -> i%200 == 0
    else
        iDisplay = (i) -> 70*i%niter == 0
    end
    while i < len2Grad*2
        _, grad = expect'(H, zero_state(n)=>circuit) 
        pars = parameters(circuit)
        dispatch!(circuit, Flux.Optimise.update!(GM(), pars, grad))
        append!(Hvalues, real.(expect(H, zero_state(n)=>circuit)))
        i = i + 1
    end
    while i < niter
        _, grad = expect'(H, zero_state(n)=>circuit) 
        pars = parameters(circuit)
        dispatch!(circuit, Flux.Optimise.update!(GM(), pars, grad))
        append!(Hvalues, real.(expect(H, zero_state(n)=>circuit)))
        append!(isPerturbed, floor(k/100))
        i=i+1
        cg, __ = curveGrad(i-len2Grad,Hvalues, h=5, di=2)
        append!(dEs, cg)
        iDisplay(i) && showSteps == true && println("Step $i, E = $(round(Hvalues[i], sigdigits=6)), dE = $(round(cg, sigdigits=6))")  
        if iConverge == 0 
            if abs(cg) <= ConvergeTH 
                iConverge = i-len2Grad
                EConverge = Hvalues[iConverge]
                showSteps == true && println("Started to Coverge at Step $i,E = $(round(Hvalues[i], sigdigits=6))")
            end
        else
            if abs(cg) <= PerturbTH && k == 0 && Hvalues[i] > sum(set)%2 + Threshold
                t = sum(Hvalues[i-19:i-10]) / sum(Hvalues[i-9:i])-1
                if t > 0 
                    sign = -sign
                end
                showSteps == true && println("Perturbation Activated")
                RXs = collect_blocks(RotationGate{1,Float64,XGate},circuit)
                dispatch!.(-,RXs,sign*PurtAmp*ones(length(RXs)))
                k = k + 100
            else
                k>0 && (k = k-1)
            end
        end
        if i == niter && autoTrain == true
            if Hvalues[i] > sum(set)%2 + Threshold && abs(cg) > StopTH && i < niterMax
                niter = niter + 100    
            end
        end
    end
    append!(dEs,repeat([NaN],len2Grad))
    println("Total steps: $(length(Hvalues)), Final E: $(round(Hvalues[end], sigdigits=6))")
    ECmin, imin = findmin(Hvalues)
    ECmm = 0
    if length(Hvalues)- imin < 2
        ECmm = mean(Hvalues[end-4:end])
    else
        ECmm = mean(Hvalues[imin-2:imin+2])
    end
    [Hvalues, isPerturbed, dEs, iConverge, EConverge, ECmin, ECmm]
end