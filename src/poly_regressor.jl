module PolyRegression

    using Base.Iterators: product
    
    using ..LinearAlgebra

    using ..DocStringExtensions

    export polynomial_exponents, regression_weights

    """
    $TYPEDSIGNATURES

    Obtain exponent tuples for a local polynomial regression of 
    a given degree.
    """
    polynomial_exponents(
        ndim::Int, degree::Int
    ) = let exps = product(
        fill(0:degree, ndim)...
    ) |> collect |> vec
        filter(
            e -> sum(e) <= degree, exps
        )
    end

    safe_exponent(x::Float64, e::Int) = (
        e == 0 ?
        1.0 :
        x ^ e
    )

    """
    $TYPEDSIGNATURES

    Obtain regression weights for local polynomial regression.
    Returns dictionary pointing exponent tuples to vectors of weights.

    `exponents` is a vector of tuples of polynomial exponents for a given degree
        (i. e. the output of `polynomial_exponents`).
    """
    function regression_weights(
        X::AbstractMatrix{Float64}, x::AbstractVector{Float64},
        exponents::AbstractVector
    )
        dX = X .- x'

        Φ = Vector{Float64}[]
        for e in exponents
            ve = collect(e)
            v = prod(
                (@. safe_exponent(dX, ve'));
                dims = 2
            ) |> vec |> x -> x ./ prod(factorial, ve)

            push!(
                Φ, v
            )
        end

        Φ = reduce(hcat, Φ)

        pinv(Φ) |> permutedims
    end

end