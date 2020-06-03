using Yao, YaoExtensions, CuYao
using Flux
using Random
using DataFrames
using CSV
using Statistics

function curveGrad(i::Int64, f::Array{Float64,1}; h::Int64=5,di::Int64=2)
    dx = 2h+di
    iHB = i+dx
    iLB = i-dx
    f_p2 = f[i+2h-di:iHB] |> mean
    f_p1 = f[i+h-di:i+h+di] |> mean
    f_m1 = f[i-h-di:i-h+di] |> mean
    f_m2 = f[iLB:i-2h+di] |> mean
    grad = ( - f_p2 + 8*f_p1 - 8*f_m1 + f_m2 ) / 12*h  
    [grad,[iLB, iHB, dx]] 
end

function HofNPP(numSet::Array{Int64,1}, Coeff::Float64=1.0)
    nbit = length(numSet)
    sz = i->put(nbit, i=>X)
    H = map(1:nbit) do i
            numSet[i]*sz(i)
        end |> sum
    H = Coeff*H^2        
end

function VQEtrain(H::Union{ChainBlock,Add}, depth=4; niter, gradCoef = 1e-3)
    n = nqubits(H)
    circuit = variational_circuit(n,depth)
    dispatch!(circuit, :random)
    for i=1:niter
        _, grad = expect'(H, zero_state(n)=>circuit)
        dispatch!(-, circuit, gradCoef * grad)
        (25i)%niter == 0 && println("Step $i, energy = $(real.(expect(H, zero_state(n)=>circuit)))")
    end
end

function VQEtrain(set::Union{Array{Int64,1}, Array{Float64,1}}; depth=4, nMeasure = 1000, niter=100, autoTrain=true, 
                  Optimizer=:NADAM, StopTH = 2e-9, PerturbTH = 1e-7, Threshold = 0.5, PurtAmp = 0.05, ConvergeTH=1e-4, nMeasureMax = 5000)
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
        iDisplay(i) && println("Step $i, E = $(round(Hvalues[i], sigdigits=6)), dE = $(round(cg, sigdigits=6))")  
        if iConverge == 0 
            if abs(cg) <= ConvergeTH 
                iConverge = i-len2Grad
                EConverge = Hvalues[iConverge]
                println("Started to Coverge at Step $i,E = $(round(Hvalues[i], sigdigits=6))")
            end
        else
            if abs(cg) <= PerturbTH && k == 0 && Hvalues[i] > sum(set)%2 + Threshold
                t = sum(Hvalues[i-19:i-10]) / sum(Hvalues[i-9:i])-1
                if t > 0 
                    sign = -sign
                end
                println("Perturbation Activated")
                RXs = collect_blocks(RotationGate{1,Float64,XGate},circuit)
                dispatch!.(-,RXs,sign*PurtAmp*ones(length(RXs)))
                k = k + 100
            else
                k>0 && (k = k-1)
            end
        end
        if i == niter && autoTrain == true
            if Hvalues[i] > sum(set)%2 + Threshold && abs(cg) > StopTH && i < nMeasureMax
                niter = niter + 100    
            end
        end
    end
    append!(dEs,repeat([NaN],len2Grad))
    ECmin, imin = findmin(Hvalues)
    ECmm = 0
    if length(Hvalues)- imin < 2
        ECmm = mean(Hvalues[end-4:end])
    else
        ECmm = mean(Hvalues[imin-2:imin+2])
    end
    [Hvalues, isPerturbed, dEs, iConverge, EConverge, ECmin, ECmm]
end

function RandIntNumSet(size::Int64, sum::Int64)
    indexSet = [1:sum;]
    middle = Int(round(sum/2))
    deleteat!(indexSet, (middle, sum))
    selection = rand(indexSet, size-2)
    append!(selection, [0,middle,sum]) |> sort!
    set = map(2:size+1) do i
            selection[i]-selection[i-1]
        end
    @show size, sum  
    @show set
    set |> shuffle!
end