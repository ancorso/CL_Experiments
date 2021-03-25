# imitation-guided continual learning
using Parameters

@with_kw mutable struct IGCL <: Solver
    π # Policy
    D # Discriminator
    S::AbstractSpace # State space
    A::AbstractSpace = action_space(π) # Action space
    N::Int = 1000 # Number of environment interactions
    ΔN::Int = 4 # Number of interactions between updates
    max_steps::Int = 100 # Maximum number of steps per episode
    log::Union{Nothing, LoggerParams} = Crux.LoggerParams(;dir="log/prior_er") # The logging parameters
    i::Int = 0 # The current number of environment interactions
    a_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the actor
    c_opt::TrainingParams = TrainingParams(;loss=Crux.td_loss, name="critic_", epochs=ΔN)# Training parameters for the critic
    d_opt::TrainingParams = TrainingParams(;loss=(D, 𝒟, 𝒟_past; info = Dict()) -> begin
            xtilde = vcat(𝒟[:s], Flux.onehotbatch(π, 𝒟[:a]))
            x = vcat(𝒟_past[:s], Flux.onehotbatch(π, 𝒟_past[:a]))
            Lᴰ(Crux.GAN_BCELoss(), D, x, xtilde)
        end, name="discriminator_", epochs=1) # Training parameters for the discriminator
        
    # Off-policy-specific parameters
    π⁻ = deepcopy(π)
    π_explore::Policy = ϵGreedyPolicy(LinearDecaySchedule(1., 0.1, floor(Int, N/2)), π.outputs) # exploration noise
    target_update = (π⁻, π; kwargs...) -> polyak_average!(π⁻, π, 0.005f0) # Function for updating the target network
    target_fn = Crux.DQN_target # Target for critic regression with input signature (π⁻, 𝒟, γ; i)
    buffer_size = 1000 # Size of the buffer
    buffer::ExperienceBuffer = ExperienceBuffer(S, A, buffer_size) # The replay buffer
    buffer_init::Int = max(c_opt.batch_size, 200) # Number of observations to initialize the buffer with
    required_columns = prioritized(buffer) ? [:weight] : Symbol[] # Extra data columns to store
    past_experience::ExperienceBuffer # extra buffers (i.e. for experience replay in continual learning)
end

function POMDPs.solve(𝒮::IGCL, mdp)
    # Construct the training buffer, constants, and sampler
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.c_opt.batch_size, 𝒮.required_columns, device=device(𝒮.π))
    𝒟_past = buffer_like(𝒮.past_experience, capacity=𝒮.c_opt.batch_size, device=device(𝒮.π))
    
    γ = Float32(discount(mdp))
    s = Sampler(mdp, 𝒮.π, 𝒮.S, 𝒮.A, max_steps=𝒮.max_steps, π_explore=𝒮.π_explore)

    # Log the pre-train performance
    log(𝒮.log, 𝒮.i, s=s)

    # Fill the buffer with initial observations before training
    𝒮.i += fillto!(𝒮.buffer, s, 𝒮.buffer_init, i=𝒮.i, explore=true)
    
    # Loop over the desired number of environment interactions
    for 𝒮.i in range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
        # Sample transitions into the replay buffer
        data = steps!(s, Nsteps=𝒮.ΔN, explore=true, i=𝒮.i)
        data[:r] .+= 0.1f0*sigmoid.(value(𝒮.D, vcat(data[:s], Flux.onehotbatch(𝒮.π, data[:a]))))
        push!(𝒮.buffer, data )
        infos = []
        # Loop over the desired number of training steps
        for epoch in 1:𝒮.c_opt.epochs
            rand!(𝒟, 𝒮.buffer, i=𝒮.i)
            rand!(𝒟_past, 𝒮.past_experience, i=𝒮.i)
            
            # Train the discriminator
            Crux.train!(𝒮.D, (;kwargs...)->𝒮.d_opt.loss(𝒮.D, 𝒟, 𝒟_past), 𝒮.d_opt)
            
            # Sample a new random minibatch
            rand!(𝒟, 𝒮.buffer, i=𝒮.i)
            rand!(𝒟_past, 𝒮.past_experience, i=𝒮.i)
            
            # Compute target
            y = 𝒮.target_fn(𝒮.π⁻, 𝒟, γ, i=𝒮.i)
            
            # Update priorities (for prioritized replay)
            (isprioritized = prioritized(𝒮.buffer)) && update_priorities!(𝒮.buffer, 𝒟.indices, cpu(td_error(𝒮.π, 𝒟, y)))
            
            # Train the critic
            info = Crux.train!(𝒮.π, (;kwargs...) -> 𝒮.c_opt.loss(𝒮.π, 𝒟, y; weighted=isprioritized, kwargs...), 𝒮.c_opt)
            
            # Train the actor 
            if !isnothing(𝒮.a_opt) && ((epoch-1) % 𝒮.a_opt.update_every) == 0
                info_a = train!(𝒮.π.A, (;kwargs...) -> 𝒮.a_opt.loss(𝒮.π, 𝒟; kwargs...), 𝒮.a_opt)
                info = merge(info, info_a)
            
                # Update the target network
                𝒮.target_update(𝒮.π⁻, 𝒮.π)
            end
            
            # Store the training information
            push!(infos, info)
            
        end
        # If not using a separate actor, update target networks after critic training
        isnothing(𝒮.a_opt) && 𝒮.target_update(𝒮.π⁻, 𝒮.π, i=𝒮.i + 1:𝒮.i + 𝒮.ΔN)
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, aggregate_info(infos), s=s)
    end
    𝒮.i += 𝒮.ΔN
    𝒮.π
end


function igcl(;default_policy, D, experience_per_task=1000, eval_eps=10, solver_args...)
  (;i, solvers=[], tasks=[], history) -> begin
      # Construct samplers for previous tasks (for recording the new policy performance on previous tasks)
      pol = isempty(solvers) ? default_policy() : deepcopy(solvers[end].π)
      samplers = [Sampler(t, pol, state_space(t)) for t in tasks]
      if i>1
          last_task = tasks[end-1]
          S = state_space(last_task)
          samp = Sampler(last_task, solvers[end].π, S, π_explore=solvers[end].π)  # Sampler for the previous task (swap out with different samplers here)
          new_buffer = ExperienceBuffer(steps!(samp, Nsteps=experience_per_task)) # sample trajectories from that task
          push!(history, new_buffer)
          𝒟expert = vcat(history...)
          return IGCL(;π=pol, solver_args..., D=D(), past_experience=𝒟expert, log=LoggerParams(dir=string(out,"log/pr_replay/task$i"), fns=[log_undiscounted_return(samplers, eval_eps)]))
      else
          return DQN(;π=pol, solver_args..., log=(dir=string(out,"log/pr_replay/task$i"), fns=[log_undiscounted_return(samplers, eval_eps)]))
      end
  end 
end

