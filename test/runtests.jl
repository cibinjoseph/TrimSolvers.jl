using trimSolvers
using Test

@testset "trim_newton" begin
    function system(x, u)
        y = zeros(2)
        y[1] = u[1]^2 + u[2]^2 - 4.0
        y[2] = 4*u[1]^2 - u[2]^2 - 4.0
        return y
    end

    u, res = trim_newton(system, [0,0], [1.0,1.0], 2;
                         u_min=[0.0, 0.0], u_max=[5.0, 5.0]);

    @test u[end] ≈ [1.264911, 1.549193] atol=1e-6
end
