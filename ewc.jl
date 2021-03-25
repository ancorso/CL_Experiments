function ewc(;solver, default_policy, λ_fisher, fisher_batch_size, eval_eps=10, solver_args...)
    (;i, solvers=[], tasks=[], kwargs...) -> begin
        # Copy over the previous policy 
        pol = isempty(solvers) ? default_policy() : deepcopy(solvers[end].π)
        
        # Construct samplers for previous tasks (for recording the new policy performance on previous tasks)
        samplers = [Sampler(t, pol, state_space(t)) for t in tasks]
        
        # Setup the regularizer
        reg = (x) -> 0
        i==2 && (reg = DiagonalFisherRegularizer(Flux.params(pol), λ_fisher)) # construct a brand new on
        i > 2 && (reg = deepcopy(solvers[end].c_opt.regularizer))
        if i>1
            loss = (𝒟) -> -mean(exp.(value(pol, 𝒟[:s])))
            update_fisher!(reg, solvers[end].buffer, loss, Flux.params(pol), fisher_batch_size) # update it with new data
        end
    
        # Construct the solver
        solver(;π=pol, solver_args..., log=(dir=string(out,"log/ewc/task$i"), fns=[log_undiscounted_return(samplers, eval_eps)]), 
            c_opt=(regularizer=reg,))
    end
end

