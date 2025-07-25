using BatchConeKernels
using KernelAbstractions
using Test
using MathOptSetDistances
using MathOptInterface
using Random

const MOI = MathOptInterface
const MOSD = MathOptSetDistances
const BCK = BatchConeKernels


using OpenCL, pocl_jll

@testset "distance" begin
    Random.seed!(1234)
    n = 5
    batch = 100
    Xcpu = randn(n, batch)
    X = OpenCL.CLArray(Xcpu)
    Y = OpenCL.zeros(batch)

    # Zero cone
    BCK.distance_to_set(MOI.Zeros, Y, X, OpenCLBackend())
    Ycpu = Array(Y)
    ref_zeros = [distance_to_set(MOSD.DefaultDistance(), Xcpu[:,i], MOI.Zeros(n)) for i in 1:batch]
    @test maximum(abs.(Ycpu .- ref_zeros)) < 1e-6

    # Nonnegatives cone
    BCK.distance_to_set(MOI.Nonnegatives, Y, X, OpenCLBackend())
    Ycpu = Array(Y)
    ref_nonneg = [distance_to_set(MOSD.DefaultDistance(), Xcpu[:,i], MOI.Nonnegatives(n)) for i in 1:batch]
    @test maximum(abs.(Ycpu .- ref_nonneg)) < 1e-6

    # Nonpositives cone
    BCK.distance_to_set(MOI.Nonpositives, Y, X, OpenCLBackend())
    Ycpu = Array(Y)
    ref_nonpos = [distance_to_set(MOSD.DefaultDistance(), Xcpu[:,i], MOI.Nonpositives(n)) for i in 1:batch]
    @test maximum(abs.(Ycpu .- ref_nonpos)) < 1e-6

    # SOC
    n_soc = 3
    X_soccpu = randn(n_soc, batch)
    X_soc = OpenCL.CLArray(X_soccpu)
    Y_soc = OpenCL.zeros(batch)
    BCK.distance_to_set(MOI.SecondOrderCone, Y_soc, X_soc, n_soc, OpenCLBackend())
    Y_soccpu = Array(Y_soc)
    ref_soc = [distance_to_set(MOSD.DefaultDistance(), X_soccpu[:,i], MOI.SecondOrderCone(n_soc)) for i in 1:batch]
    @test maximum(abs.(Y_soccpu .- ref_soc)) < 1e-3

    # RotatedSecondOrderCone
    n_rsoc = 4
    X_rsoccpu = randn(n_rsoc, batch)
    X_rsoc = OpenCL.CLArray(X_rsoccpu)
    Y_rsoc = OpenCL.zeros(batch)
    BCK.distance_to_set(MOI.RotatedSecondOrderCone, Y_rsoc, X_rsoc, n_rsoc, OpenCLBackend())
    Y_rsoccpu = Array(Y_rsoc)
    ref_rsoc = [distance_to_set(MOSD.DefaultDistance(), X_rsoccpu[:,i], MOI.RotatedSecondOrderCone(n_rsoc)) for i in 1:batch]
    @test maximum(abs.(Y_rsoccpu .- ref_rsoc)) < 1e-3

    # PowerCone
    X_powcpu = randn(3, batch)
    X_pow = OpenCL.CLArray(X_powcpu)
    Y_pow = OpenCL.zeros(batch)
    exponent = 0.4
    BCK.distance_to_set(MOI.PowerCone, Y_pow, X_pow, exponent, OpenCLBackend())
    Y_powcpu = Array(Y_pow)
    ref_pow = [distance_to_set(MOSD.DefaultDistance(), X_powcpu[:,i], MOI.PowerCone(exponent)) for i in 1:batch]
    @test maximum(abs.(Y_powcpu .- ref_pow)) < 1e-3

    # DualPowerCone
    X_dpowcpu = rand(3, batch)
    X_dpow = OpenCL.CLArray(X_dpowcpu)
    Y_dpow = OpenCL.zeros(batch)
    BCK.distance_to_set(MOI.DualPowerCone, Y_dpow, X_dpow, exponent, OpenCLBackend())
    Y_dpowcpu = Array(Y_dpow)
    ref_dpow = [distance_to_set(MOSD.DefaultDistance(), X_dpowcpu[:,i], MOI.DualPowerCone(exponent)) for i in 1:batch]
    @test maximum(abs.(Y_dpowcpu .- ref_dpow)) < 1e-3

    # # Exponential cone
    # X_expcpu = randn(3, batch)
    # X_exp = OpenCL.CLArray(X_expcpu)
    # Y_exp = OpenCL.zeros(batch)
    # BCK.distance_to_set(MOI.ExponentialCone, Y_exp, X_exp, OpenCLBackend())
    # Y_expcpu = Array(Y_exp)
    # ref_exp = [distance_to_set(MOSD.DefaultDistance(), X_expcpu[:,i], MOI.ExponentialCone()) for i in 1:batch]
    # @test maximum(abs.(Y_expcpu .- ref_exp)) < 1e-3

    # # Dual Exponential cone
    # X_dexpcpu = randn(3, batch)
    # X_dexp = OpenCL.CLArray(X_dexpcpu)
    # Y_dexp = OpenCL.zeros(batch)
    # BCK.distance_to_set(MOI.DualExponentialCone, Y_dexp, X_dexp, OpenCLBackend())
    # Y_dexpcpu = Array(Y_dexp)
    # ref_dexp = [distance_to_set(MOSD.DefaultDistance(), X_dexpcpu[:,i], MOI.DualExponentialCone()) for i in 1:batch]
    # @test maximum(abs.(Y_dexpcpu .- ref_dexp)) < 1e-3
end
