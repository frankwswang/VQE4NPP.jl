export RandIntNumSet


"""
    curveGrad(i::Int64, f::Array{Float64,1}; h::Int64=5, di::Int64=2)
    -> 
    [grad,[iLB, iHB, dx]]
Numerical differentiation (5-point method) of the input function f which is in terms of an discrete array.
    \n== Arguments ==
    \n`i::Int64`: The point (index) for derivative calcualtion.
    \n`f::Array{Float64,1}`: The array of function values.
    \n`h::Int64`: The interval h. Default value is 5. 
    \n`di::Int64`: An addtional interval for mean value at each point to compenstate the data fluctuation. Default value is 2. 
    \n== Output variables ==
    \n`grad::Float64`: Derivative of f at point (index) i.
    \n`iLB::Int64`: Lower bound requirement for the index.
    \n`iHB::Int64`: Upper bound requirement for the index.
    \n`dx::Int64`: half interval length. iHB = i+dx iLB = i-dx  
"""
function curveGrad(i::Int64, f::Array{Float64,1}; h::Int64=5, di::Int64=2)
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


"""
    HofNPP(numSet::Array{Int64,1}, Coeff::Float64=1.0)
    ->
    H
Generating the corresponding Hamiltonian given a number set.
    \n== Arguments ==
    \n`numSet::Array{Int64,1}`: The target number set.
    \n`Coeff::Float64`: An overall coefficent. Default value is 1.0.
    \n== Output variables ==
    \n`H:CompositeBlock`: The corresponding Hamiltonian.
"""
function HofNPP(numSet::Array{Int64,1}, Coeff::Float64=1.0)
    nbit = length(numSet)
    # sz = i->put(nbit, i=>X)
    sz = i->put(nbit, i=>Z)
    H = map(1:nbit) do i
            numSet[i]*sz(i)
        end |> sum
    H = Coeff*H^2        
end


"""
    RandIntNumSet(size::Int64, sum::Int64)
    ->
    set::Array{Int64,1}
Randomly generating number set with equi-partition soluttions.
    \n== Arguments ==
    \n`size::Int64`: 
    \n`sum::Int64`: 
    \n== Output variables ==
    \n`set::Array{Int64,1}`: The generated number set.
"""
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