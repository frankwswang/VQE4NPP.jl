# push!(LOAD_PATH, abspath("src"))
using Test
using VQE4NPP
using Yao

@testset "curveGrad() Test" begin
    size = 20
    func = map(1:size) do i
        7*exp(0.1i)
    end

    ders = map(1:size) do i
        0.7*exp(0.1i)
    end

    # Test function accuracy of default sewttings.
    h = 1
    di = 1
    for j=1+2h+di:size-2h-di
        @test isapprox( ders[j], VQE4NPP.curveGrad(j, func, h=h, di = di)[1]; atol=1e-1 )
    end

    # Test function accuracy without mean-value compensation of data fluctuation.
    h = 1
    di = 0
    for j=1+2h+di:size-2h-di
        @test isapprox( ders[j], VQE4NPP.curveGrad(j, func, h=h, di = di)[1]; atol=1e-4 )
    end 
end

@testset "HofNPP() Test" begin
    n = 5
    c = 0.7
    set = rand(1:20,n)    
    H1 = VQE4NPP.HofNPP(set, c)
    H2 = set[1]*put(n,1=>Z)
    for i = 2:n
        H2 = H2 + set[i]*put(n,i=>Z)
    end
    H2 = c*H2^2
    reg = rand_state(n)
    cX = chain(n, [put(n, i=>Ry(pi/2)) for i=1:n])
    cY = chain(n, [put(n, i=>Rx(pi/2)) for i=1:n])
    @test expect(H1, copy(reg) => cX) == expect(H2, copy(reg) => cX)
    @test expect(H1, copy(reg) => cY) == expect(H2, copy(reg) => cY)
    @test expect(H1, copy(reg)) == expect(H2, copy(reg))  
end

@testset "RandIntNumSet() Test" begin
    sz = 10

    # Test when the minimal diffrence of 2 subsets is 0.
    sm1 = 100
    set1 = RandIntNumSet(sz, sm1, isShuffled=false)
    s1 = Float64[]
    for i=1:sz
        s1 = push!(s1, set1[i])
        s1 |> sum == sm1/2 && break
    end 
    @test s1 |> sum == set1[length(s1)+1:end] |> sum 

    # Test when the minimal diffrence of 2 subsets is 1.
    sm2 = 101
    set2 = RandIntNumSet(sz, sm2, isShuffled=false)
    s2 = Float64[]
    for i=1:sz
        s2 = push!(s2, set2[i])
        ( s2 |> sum == sm2/2-0.5 || s2 |> sum == sm2/2+0.5 ) && break
    end 
    @test (s2 |> sum) - (set2[length(s2)+1:end] |> sum) |> abs == 1 
end

@testset "VQEtrain() Test" begin
    dx = 12 # 2*5+2, h=5, di=2    
    #Test the range of each output variables.
    n = 3
    d = 4
    s = 50
    set = RandIntNumSet(n,s)
    res = VQEtrain(set, depth = d, niter=1000, autoTrain=false, showTrain=false)
    # ic = res[4]
    # ic == 0 && ic = ic +1 
    @test res[1][end] < 0.5
    @test res[3][end-dx] < 1e-8*5
    @test res[1][res[4]] == res[5]
    @test res[3][res[4]] <= 1e-3
    @test res[6] < 0.5
    @test isapprox(res[6], res[7], atol = 1e-1)

    #Test the perturbations option and the autoTrain option.
    n = 6
    d = 6
    s = 200
    set = RandIntNumSet(n,s)
    res = VQEtrain(set, depth = d, showTrain=false)
    iPd = findall(i->i==1, res[2])
    for i = 1:length(iPd)
        @test (res[1][iPd[i]] - res[1][iPd[i]-1]) > (res[1][iPd[i]-1] - res[1][iPd[i]-2]) |> abs
        @test res[3][iPd[i]] |> abs > 1e-7
        @test res[3][iPd[i]-dx] |> abs > 1e-7
        @test res[3][iPd[i]-dx-1] |> abs < 1e-7
    end
    @test length(res[1]) <= 5000
end