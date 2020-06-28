function evaluate_policy(T; γ = 1, π = fill(0.25, 4))
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
