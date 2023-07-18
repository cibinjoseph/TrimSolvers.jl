import TrimSolvers as ts
using Test

@testset "trim_newton" begin
    function system(x, u)
        y = zeros(2)
        y[1] = u[1]^2 + u[2]^2 - 4.0
        y[2] = 4*u[1]^2 - u[2]^2 - 4.0
        return y
    end

    u, res = ts.trim_newton(system, [0,0], [1.0,1.0], 2;
                         u_min=[0.0, 0.0], u_max=[5.0, 5.0]);

    @test u[end] ≈ [1.264911, 1.549193] atol=1e-6
end

@testset "pinv" begin
    A = zeros(2, 3)
    A[1, :] = [1.0, 2.0, 3.0]
    A[2, :] = [3.0, 4.0, 5.0]
    rinv_A = ts.pinv(A)
    linv_A = ts.pinv(A')

    rinv_A_true = zeros(3, 2)
    rinv_A_true[:, 1] = [-3.5, -0.5, 2.5]
    rinv_A_true[:, 2] = [2.0, 0.5, -1.0]
    rinv_A_true = rinv_A_true ./ 3

    @test rinv_A ≈ rinv_A_true atol=1e-6
    @test linv_A ≈ rinv_A' atol=1e-6
end
