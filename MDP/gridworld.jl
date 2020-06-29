function evaluate_policy(T::Int; γ = 1, π = fill(0.25, 4))
    v = zeros(16)
    coor = reshape(vcat(15, 1:14, 16), 4, :)'
    for t = 1:T
        for xs = 1:4
            for ys = 1:4
                s = coor[ys, xs]
                if s == 16
                    continue
                end
                tmp = 0.0
                # summation over a
                # left
                if xs == 1
                    s′ = s
                else
                    s′ = coor[ys, xs - 1]
                end
                tmp += π[1] * (-1 + γ * v[s′])
                # right (can reach the target)
                if xs == 4
                    s′ = s
                else
                    s′ = coor[ys, xs + 1]
                end
                if s′ == 16
                    tmp += π[2] * (0 + γ * v[s′])
                else
                    tmp += π[2] * (-1 + γ * v[s′])
                end
                # upward
                if ys == 1
                    s′ = s
                else
                    s′ = coor[ys - 1, xs]
                end
                tmp += π[3] * (-1 + γ * v[s′])
                # downward (can reach the target)
                if ys == 4
                    s′ = s
                else
                    s′ = coor[ys + 1, xs]
                end
                if s′ == 16
                    tmp += π[4] * (0 + γ * v[s′])
                else
                    tmp += π[4] * (-1 + γ * v[s′])
                end
                v[s] = tmp
            end
        end
        println("t = $t: ", v)
    end
end

function evaluate_policy!(v::Vector{Float64}, π::Matrix, coor::Matrix{Int}; γ = 1, maxit = 100, verbose = false)
    lastv = ones(16)*100
    t = 1
    while true
        for xs = 1:4
            for ys = 1:4
                s = coor[ys, xs]
                if s == 16
                    continue
                end
                tmp = 0.0
                # summation over a
                # left
                if xs == 1
                    s′ = s
                else
                    s′ = coor[ys, xs - 1]
                end
                tmp += π[1, s] * (-1 + γ * v[s′])
                # right (can reach the target)
                if xs == 4
                    s′ = s
                else
                    s′ = coor[ys, xs + 1]
                end
                if s′ == 16
                    tmp += π[2, s] * (0 + γ * v[s′])
                else
                    tmp += π[2, s] * (-1 + γ * v[s′])
                end
                # upward
                if ys == 1
                    s′ = s
                else
                    s′ = coor[ys - 1, xs]
                end
                tmp += π[3, s] * (-1 + γ * v[s′])
                # downward (can reach the target)
                if ys == 4
                    s′ = s
                else
                    s′ = coor[ys + 1, xs]
                end
                if s′ == 16
                    tmp += π[4, s] * (0 + γ * v[s′])
                else
                    tmp += π[4, s] * (-1 + γ * v[s′])
                end

                v[s] = tmp
            end
        end
        ϵ = sum((lastv .- v) .^2)
        if verbose
            println("it = $t: $ϵ, $v")
        end
        t += 1
        if (ϵ < 1e-3) || (t > maxit)
            break
        end
        lastv .= v
    end
end

function update_Q!(Q::Matrix{Float64}, v::Vector{Float64}, coor::Matrix{Int}, γ = 1)
    for xs = 1:4
        for ys = 1:4
            s = coor[ys, xs]
            if s == 16
                continue
            end
            # left
            # s′ = ifelse(xs == 1, s, coor[ys, xs - 1])
            s′ = xs == 1 ? s : coor[ys, xs - 1]
            Q[1, s] = -1 + γ * v[s′]
            # right
            s′ = xs == 4 ? s : coor[ys, xs + 1]
            Q[2, s] = s′ == 16 ? 0 + γ * v[s′] : -1 + γ * v[s′]
            # up
            s′ = ys == 1 ? s : coor[ys - 1, xs]
            Q[3, s] = -1 + γ * v[s′]
            # down
            s′ = ys == 4 ? s : coor[ys + 1, xs]
            Q[4, s] = s′ == 16 ? 0 + γ * v[s′] : -1 + γ * v[s′]
        end
    end
end

function policy_iteration(T)
    coor = Matrix{Int}(reshape(vcat(15, 1:14, 16), 4, :)')
    # init policy
    π = zeros(Int, 4, 16)
    π[1, :] .= 1 # all to left
    v = zeros(16)
    Q = zeros(4, 16)
    println("t = 0: ")
    print_policy(π)
    println("")
    for t = 1:T
        # evaluate the policy
        # no need to run many times until converge, so set maxit = 0
        evaluate_policy!(v, π, coor, maxit = 0)
        # calculate the state-action value
        update_Q!(Q, v, coor)
        # compute new policy
        π .= 0
        π[argmax(Q, dims = 1)] .= 1
        # println("t = $t: ", v)
        # display(π)
        # display(Q)
        println("t = $t: ")
        print_policy(π)
        println("")
    end
    return π
end

function print_policy(π)
    for (i, s) in enumerate(vcat(15, 1:14, 16))
        if i == 1
            print("o ")
            continue
        elseif i == 16
            print("x")
            continue
        end
        if π[1, s] == 1
            print("← ")
        elseif π[2, s] == 1
            print("→ ")
        elseif π[3, s] == 1
            print("↑ ")
        elseif π[4, s] == 1
            print("↓ ")
        end
        if i % 4 == 0
            print("\n")
        end
    end
end
