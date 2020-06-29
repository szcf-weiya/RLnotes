using StatsBase

function ϵgreedy(q::Vector{Float64}, ϵ = 0.1)
    if rand() < 1 - ϵ
        return argmax(q)
    else
        return sample(1:length(q), 1)[1]
    end
end

iscliff(s::Vector{Int}) = (s[1] == 4) && (2 <= s[2] <= 11)

function take_action(s::Vector{Int}, a::Int)
    if a == 1 # left
        s′ = copy(s[2] == 1 ? s : s .- [0, 1])
        if iscliff(s′)
            r = -100
            s′ = [4, 1]
        else
            r = -1
        end
    elseif a == 2 #right
        s′ = copy(s[2] == 12 ? s : s .+ [0, 1])
        if iscliff(s′)
            r = -100
            s′ = [4, 1]
        else
            r = -1
        end
    elseif a == 3 #up
        s′ = copy(s[1] == 1 ? s : s .- [1, 0])
        r = -1
    else
        s′ = copy(s[1] == 4 ? s : s .+ [1, 0])
        if iscliff(s′)
            r = -100
            s′ = [4, 1]
        else
            r = -1
        end
    end
    return s′, r
end

function update_status!(status::Array{Char, 2}, s::Vector{Int}, a::Int; actions = ['←', '→', '↑', '↓'])
    status[s...] = actions[a]
end

function init_status!(status::Array{Char, 2})
    status .= ' '
    status[4, 2:11] .= '*'
end

function print_status(status::Array{Char, 2})
    for i = 1:size(status, 1)+2
        print('|', ' ')
        if (i == 1) || (i == size(status, 1) + 2)
            for j = 1:size(status, 2)
                print('—', ' ') #\emdash
            end
        else
            for j = 1:size(status, 2)
                print(status[i-1, j], ' ')
            end
        end
        print('|', '\n')
    end
    println("")
end

function sarsa(EPISODE; α = 0.01, γ = 1, ϵ = 0.1, verbose = true)
    # initialization
    Q = rand(4, 12, 4)
    Q[4, 12, :] .= 0
    status = Array{Char, 2}(undef, 4, 12)
    rewards = zeros(EPISODE)
    for episode = 1:EPISODE
        if verbose
            println("episode $episode: ")
        end
        init_status!(status)
        # start from S
        s = [4, 1]
        # choose action
        a = ϵgreedy(Q[s..., :], ϵ)
        update_status!(status, s, a)
        if verbose
            # update_status!(status, s, a)
            print_status(status)
        end
        while true
            # take action
            s′, r = take_action(s, a)
            rewards[episode] += r
            # choose A'
            a′ = ϵgreedy(Q[s′..., :], ϵ)
            Q[s..., a] += α * (r + γ * Q[s′..., a′] - Q[s..., a])
            s .= s′
            a = a′
            update_status!(status, s, a)
            if verbose
                # update_status!(status, s, a)
                print_status(status)
            end
            if s == [4, 12]
                break
            end
            # sleep(0.01)
            # if read(stdin, Char) == ' '
            #     continue
            # end
        end
        if episode == EPISODE
            print_status(status)
        end
    end
    return rewards
end

function qlearning(EPISODE; α = 0.01, γ = 1, ϵ = 0.1, verbose = true)
    # initialization
    Q = rand(4, 12, 4)
    Q[4, 12, :] .= 0
    status = Array{Char, 2}(undef, 4, 12)
    rewards = zeros(EPISODE)
    for episode = 1:EPISODE
        # start from
        s = [4, 1]
        init_status!(status)
        while true
            # choose action
            a = ϵgreedy(Q[s..., :], ϵ)
            update_status!(status, s, a)
            # take action
            s′, r = take_action(s, a)
            rewards[episode] += r
            # qlearning
            Q[s..., a] += α * (r + γ * maximum(Q[s′..., :]) -  Q[s..., a])
            s .= s′
            if s == [4, 12]
                break
            end
        end
        if episode == EPISODE
            print_status(status)
        end
    end
    return rewards
end

function plot_res(; kw...)
    println("Sarsa:")
    rewards1 = sarsa(500, verbose=false, ϵ=0.1, α=0.5)
    println("Q-learning:")
    rewards2 = qlearning(500, verbose=false, ϵ=0.1, α=0.5)
    plot(cumsum(rewards1) ./ collect(1:length(rewards1)); label = "Sarsa", kw...)
    plot!(cumsum(rewards2) ./ collect(1:length(rewards2)), label="Q-learning")
end
