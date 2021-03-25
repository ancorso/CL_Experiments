function from_scratch(;solver, default_policy, eval_eps=10, solver_args...)
    (;i, kwargs...) -> begin
        solver(;π=default_policy(), solver_args..., log=(dir=string(out,"log/scratch/task$i"), fns=[log_undiscounted_return(eval_eps, name="undiscounted_return/T$i")]))
    end
end 

function warm_start(;solver, default_policy, eval_eps=10, solver_args...)
    (;i, solvers=[], tasks=[], kwargs...) -> begin
        # Copy over the previous policy 
        pol = isempty(solvers) ? default_policy() : deepcopy(solvers[end].π)
        
        # Construct samplers for previous tasks (for recording the new policy performance on previous tasks)
        samplers = [Sampler(t, pol, state_space(t)) for t in tasks]
        
        # Construct the solver
        solver(;π=pol, solver_args..., log=(dir=string(out,"log/warmstart/task$i"), fns=[log_undiscounted_return(samplers, eval_eps)]))
    end
end

