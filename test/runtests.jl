using Test
using VQE4NPP

@testset "curveGrad() test" begin
    size = 20
    func = map(1:size) do i
        7*exp(0.1i)
    end

    ders = map(1:size) do i
        0.7*exp(0.1i)
    end

    # Test function accuracy of default sewttings.
    h = 5
    di = 2
    for j=1+2h+di:size-2h-di
        @test ders[j] ≈ curveGrad(j, func)
    end

    # Test function accuracy without mean-value compensation of data fluctuation.
    h = 2
    di = 0
    for j=1+2h+di:size-2h-di
        @test ders[j] ≈ curveGrad(j, func, h=2, di = 0)
    end 
end

@testset "HofNPP() test" begin
    n = 5
    c = 0.7
    set = rand(1:20,n)    
    H1 = HofNPP(set, c)
    H2 = put(n,1=>Z)
    for i = 2:n
        H2 = H2 + put(n,i=>Z)
    end
    H2 = c*H2^2
    @test H1 == H2 
end

@testset "RandIntNumSet() test" begin
    sz = 10
    sm = 100
    mid = sz/2 |> Int
    set = RandIntNumSet(sz, sm, isShuffled=false)
    # Test when the minimal diffrence of 2 subsets is 0.
    @test set[1:mid] |> sum == set[mid+1:end] |> sum 

    sm = 101
    set = RandIntNumSet(sz, sm, isShuffled=false)
    # Test when the minimal diffrence of 2 subsets is 1.
    @test ( set[1:mid] - set[mid+1:end] ) |> sum |> abs == 1
end

@testset "VQEtrain() test" begin
    #Test the range of each output variables.
    n = 3
    d = 4
    s = 50
    set = RandIntNumSet(n,s)
    res = VQEtrain(set, depth = d, niter=200, autoTrain=false)
    @test res[1][end] < 0.5
    @test res[3][end] < 1e-8*5
    @test res[1][res[4]] == res[5]
    @test res[3][res[4]] <= 1e-3
    @test res[6] < 0.5
    @test res[6] ≈ res[7]

    #Test the perturbations option and the autoTrain option.
    n = 8
    d = 6
    s = 200
    dx = 12 # 2*5+2, h=5, di=2
    set = RandIntNumSet(n,s)
    res = VQEtrain(set, depth = d)
    iPd = findall(i->i==1, res[2])
    for i = 1:length(iPd)
        (res[1][iPd] - res[1][iPd-1]) > (res[1][iPd-1] - res[1][iPd-2]) |> abs
        res[3][iPd] > 1e-7
        res[3][iPd-dx] > 1e-7
        res[3][iPd-dx-1] < 1e-7
    end
    length(res[1]) < 5000
end