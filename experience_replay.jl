function experience_replay(;solver, default_policy,
                            experience_per_task=1000, # Number of samples to store for each task
                            experience_frac=0.5, # Fraction of the data that will come from past experience
                            bc_batch_size=64, # Batch size of the behavioral cloning regularization
                            λ_bc=1f0, # Behaviroal cloning regularization coefficient
                            eval_eps=10,
                            solver_args...
                          )
    (;i, solvers=[], tasks=[], kwargs...) -> begin
        # Copy over the previous policy 
        pol = isempty(solvers) ? default_policy() : deepcopy(solvers[end].π)
        
        # Construct samplers for previous tasks (for recording the new policy performance on previous tasks)
        samplers = [Sampler(t, pol, state_space(t)) for t in tasks]
        
        # Experience replay
        experience, fracs = isempty(solvers) ? ([], [1.0]) : begin 
            s_last = samplers[end-1] # Sampler for the previous task (swap out with different samplers here)
            new_buffer = ExperienceBuffer(steps!(s_last, Nsteps=experience_per_task)) # sample trajectories from that task
            new_buffer.data[:value] = value(pol, new_buffer[:s]) # compute their values (for behavioral cloning regularization)
            [solvers[end].extra_buffers..., new_buffer], [1-experience_frac, experience_frac * ones(i-1) ./ (i-1) ...]
        end
        
        bcreg = i>1 ? BatchRegularizer(buffers=experience, batch_size=bc_batch_size, λ=λ_bc, loss=value_regularization) : (x)->0
        
        # Construct the solver
        solver(;π=pol, solver_args..., log=(dir=string(out,"log/er/task$i"), fns=[log_undiscounted_return(samplers, eval_eps)]), 
            extra_buffers=experience,
            buffer_fractions=fracs,
            c_opt=(regularizer=bcreg,))
    end 
end

