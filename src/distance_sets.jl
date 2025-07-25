## COV_EXCL_START

@kernel function kerd_zero(Y, @Const(X))
    i = @index(Global)
    acc = zero(eltype(X))
    @inbounds for j in 1:size(X, 1)
        acc += X[j, i]^2
    end
    @inbounds Y[i] = sqrt(acc)
end

@kernel function kerd_nonneg(Y, @Const(X))
    i = @index(Global)
    acc = zero(eltype(X))
    @inbounds for j in 1:size(X, 1)
        vj = X[j, i]
        acc += (vj < 0 ? -vj : 0)^2
    end
    @inbounds Y[i] = sqrt(acc)
end

@kernel function kerd_nonpos(Y, @Const(X))
    i = @index(Global)
    acc = zero(eltype(X))
    @inbounds for j in 1:size(X, 1)
        vj = X[j, i]
        acc += (vj > 0 ? vj : 0)^2
    end
    @inbounds Y[i] = sqrt(acc)
end

@kernel function kerd_soc(Y, @Const(X), n)
    i = @index(Global)
    t = X[1, i]
    acc = zero(eltype(X))
    @inbounds for j in 2:n
        acc += X[j, i]^2
    end
    normx = sqrt(acc)
    if normx <= t
        @inbounds Y[i] = zero(normx)
    elseif normx <= -t
        @inbounds Y[i] = sqrt(t^2 + normx^2)
    else
        @inbounds Y[i] = (normx - t) / sqrt(2.0)
    end
end

@kernel function kerd_rsoc(Y, @Const(X), n)
    i = @index(Global)
    t = X[1, i]
    u = X[2, i]
    acc = zero(eltype(X))
    @inbounds for j in 3:n
        acc += X[j, i]^2
    end
    dot_xs = acc
    r1 = max(-t, 0)
    r2 = max(-u, 0)
    r3 = max(dot_xs - 2 * t * u, 0)
    @inbounds Y[i] = sqrt(r1^2 + r2^2 + r3^2)
end

@kernel function kerd_pow(Y, @Const(X), exponent)
    i = @index(Global)
    x = X[1, i]
    y = X[2, i]
    z = X[3, i]
    if x < 0 || y < 0
        r1 = max(-x, 0)
        r2 = max(-y, 0)
        @inbounds Y[i] = sqrt(r1^2 + r2^2)
    else
        result = abs(z) - x^exponent * y^(1-exponent)
        @inbounds Y[i] = max(result, 0)
    end
end

@kernel function kerd_dpow(Y, @Const(X), exponent)
    i = @index(Global)
    u = X[1, i]
    v = X[2, i]
    w = X[3, i]
    ce = 1 - exponent
    r1 = max(-u, 0)
    r2 = max(-v, 0)
    r3 = max(abs(w) - (u/exponent)^exponent * (v/ce)^ce, 0)
    @inbounds Y[i] = sqrt(r1^2 + r2^2 + r3^2)
end

@kernel function kerd_exp(Y, @Const(X))
    i = @index(Global)
    x = X[1, i]
    y = X[2, i]
    z = X[3, i]
    if x <= 0 && abs(y) < 1e-10 && z >= 0
        @inbounds Y[i] = zero(x)
    else
        r1 = max(-y, 0)
        r2 = max(y * exp(x / y) - z, 0)  #FIXME: div by zero
        @inbounds Y[i] = sqrt(r1^2 + r2^2)
    end
end

@kernel function kerd_dexp(Y, @Const(X))
    i = @index(Global)
    u = X[1, i]
    v = X[2, i]
    w = X[3, i]
    if abs(u) < 1e-10 && v >= 0 && w >= 0
        @inbounds Y[i] = zero(u)
    else
        r1 = max(u, 0)
        r2 = max(-u * exp(v / u) - ℯ * w, 0)  #FIXME: div by zero
        @inbounds Y[i] = sqrt(r1^2 + r2^2)
    end
end

## COV_EXCL_STOP

@inline function distance_to_set(::Type{C}, Y, X, backend=CPU()) where C <: MOI.Zeros
    kerd_zero(backend)(Y, X, ndrange=size(X, 2))
end
@inline function distance_to_set(::Type{C}, Y, X, backend=CPU()) where C <: MOI.Nonnegatives
    kerd_nonneg(backend)(Y, X, ndrange=size(X, 2))
end
@inline function distance_to_set(::Type{C}, Y, X, backend=CPU()) where C <: MOI.Nonpositives
    kerd_nonpos(backend)(Y, X, ndrange=size(X, 2))
end
@inline function distance_to_set(::Type{C}, Y, X, n, backend=CPU()) where C <: MOI.SecondOrderCone
    kerd_soc(backend)(Y, X, n, ndrange=size(X, 2))
end
@inline function distance_to_set(::Type{C}, Y, X, n, backend=CPU()) where C <: MOI.RotatedSecondOrderCone
    kerd_rsoc(backend)(Y, X, n, ndrange=size(X, 2))
end
@inline function distance_to_set(::Type{C}, Y, X, exponent, backend=CPU()) where C <: MOI.PowerCone
    kerd_pow(backend)(Y, X, exponent, ndrange=size(X, 2))
end
@inline function distance_to_set(::Type{C}, Y, X, exponent, backend=CPU()) where C <: MOI.DualPowerCone
    kerd_dpow(backend)(Y, X, exponent, ndrange=size(X, 2))
end
# @inline function distance_to_set(::Type{C}, Y, X, backend=CPU()) where C <: MOI.ExponentialCone
#     kerd_exp(backend)(Y, X, ndrange=size(X, 2))
# end
# @inline function distance_to_set(::Type{C}, Y, X, backend=CPU()) where C <: MOI.DualExponentialCone
#     kerd_dexp(backend)(Y, X, ndrange=size(X, 2))
# end