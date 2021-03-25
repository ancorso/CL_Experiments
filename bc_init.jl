function bc_init(;solver, default_policy, experience_per_task=1000, eval_eps=10, λe::Float32=1f-3, solver_args...)
    (;i, solvers=[], tasks=[], history) -> begin
        # Add new data and behavioral clone pretrainthe policy
        pol = isempty(solvers) ? default_policy() : deepcopy(solvers[end].π)
        if i>1
            last_task = tasks[end-1]
            S = state_space(last_task)
            samp = Sampler(last_task, solvers[end].π, S, π_explore=solvers[end].π)  # Sampler for the previous task (swap out with different samplers here)
            new_buffer = ExperienceBuffer(steps!(samp, Nsteps=experience_per_task, explore=true)) # sample trajectories from that task
            new_buffer.data[:value] = value(solvers[end].π, new_buffer[:s]) # compute their values (for behavioral cloning regularization)
            push!(history, new_buffer)
            𝒟expert = vcat(history...)
            solve(BC(π=pol, 𝒟_expert=𝒟expert, S=S, loss=Crux.mse_value_loss(λe), log=(period=200,), opt=(epochs=100000, batch_size=128)), tasks[end])
        end
        
        # Construct samplers for previous tasks (for recording the new policy performance on previous tasks)
        samplers = [Sampler(t, pol, state_space(t)) for t in tasks]
        
        # Construct the solver
        𝒮 = solver(;π=pol, solver_args..., log=(dir=string(out,"log/bc_init/task$i"), fns=[log_undiscounted_return(samplers, eval_eps)]))
        𝒮
    end 
end

