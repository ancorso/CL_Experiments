using Crux, Flux, POMDPs, POMDPGym, Random, Plots
using BSON, TensorBoardLogger, StaticArrays, POMDPModels, Zygote

include("baselines.jl")
include("experience_replay.jl")
include("ewc.jl")
include("bc_init.jl")
include("discriminative_experience_replay.jl")
include("igcl.jl")

## Build the tasks
Random.seed!(3) # Set the random seed
Ntasks = 3 # Number of tasks to solver
sz = (10,10) # Dimension of the gridworld
N = Float32(maximum(sz)) / 2
LavaWorldMDP
tasks = [LavaWorldMDP(size = sz, tprob = 0.99, goal=:random, randomize_lava=false, num_lava_tiles=20, observation_type=:twoD, discount=0.8) for _=1:Ntasks]
discount(tasks)
tasks = repeat(tasks,2)
convert_s(AbstractArray, rand(initialstate(tasks[1])), tasks[1])
S = state_space(tasks[1]) # The state space of the tasks
input_dim = prod(dim(S)) # three channels represent player position, lava, and goal
as = [actions(tasks[1])...] # The actions 
# render(tasks[1], GWPos(5,5)) # Plots the task
# render(tasks[2], GWPos(5,5)) # Plots the task
# render(tasks[3], GWPos(5,5)) # Plots the task

## Training hyperparameters
out = "output/"
try mkdir(out) catch end
eval_eps = 10 # Number os episodes used for evaluation of the policy
solver_args=(N=150000, S=S, )

# Define the network
Q() = DiscreteNetwork(Chain(x->(reshape(x, input_dim, :) .- N) ./ N, Dense(input_dim, 64, relu), Dense(64,64, relu), Dense(64, 4), ), as)
DSN_SASR() = ContinuousNetwork(Chain(x-> (x .- [N, N, 0f0, 0f0, 0f0, 0f0, N, N, 0f0]) ./ [N, N, 1f0, 1f0, 1f0, 1f0, N, N, 1f0], DenseSN(9, 64, relu), DenseSN(64,64, relu), DenseSN(64, 1)))
DSN_SA() = ContinuousNetwork(Chain(x-> (x .- [N, N, 0f0, 0f0, 0f0, 0f0]) ./ [N, N, 1f0, 1f0, 1f0, 1f0], DenseSN(6, 64, relu), DenseSN(64,64, relu), DenseSN(64, 1)))

## from scratch
scratch_solvers = continual_learning(tasks, from_scratch(;solver=DQN, default_policy=Q, solver_args...))
BSON.@save string(out, "scratch_solvers.bson") scratch_solvers

## warm start
warmstart_solvers = continual_learning(tasks, warm_start(;solver=DQN, default_policy=Q, solver_args...))
BSON.@save string(out,"warmstart_solvers.bson") warmstart_solvers

## BC Initialization
bc_init_solvers = continual_learning(tasks, bc_init(;solver=DQN, default_policy=Q, experience_per_task=1000, λe=0.1f0, solver_args...))
BSON.@save string(out,"bc_init_solvers.bson") bc_init_solvers

## discriminative experience replay
der_solvers = continual_learning(tasks, discriminative_experience_replay(;default_policy=Q, experience_per_task=1000, D=DSN_SASR, solver_args...))

for 𝒮 in der_solvers
    if 𝒮 isa DiscriminitiveExperienceReplay
        𝒮.past_experience = 𝒮.buffer
    end
end
BSON.@save string(out,"der_solvers.bson") der_solvers

## imitation-guided continual learning
igcl_solvers = continual_learning(tasks, igcl(;default_policy=Q, experience_per_task=1000, D=DSN_SA, solver_args...))
BSON.@save string(out,"igcl_solvers.bson") igcl_solvers


## Experience replay
er_solvers = continual_learning(tasks, experience_replay(;solver=DQN, default_policy=Q,
                                                        experience_per_task=1000, # Number of samples to store for each task
                                                        experience_frac=0.5, # Fraction of the data that will come from past experience
                                                        bc_batch_size=64, # Batch size of the behavioral cloning regularization
                                                        λ_bc=1f0, # Behaviroal cloning regularization coefficient
                                                        solver_args...
                                                        ))
BSON.@save string(out,"er_solvers.bson") er_solvers


## Elastic Weight consolidation
ewc_solvers = continual_learning(tasks, ewc(;solver=DQN, default_policy=Q, λ_fisher=1e12, fisher_batch_size=128, eval_eps=10, solver_args...))
BSON.@save string(out,"ewc_solvers.bson") ewc_solvers


## Plot the results

# load the results (optional)
scratch_solvers = BSON.load(string(out,"scratch_solvers.bson"))[:scratch_solvers]
warmstart_solvers = BSON.load(string(out,"warmstart_solvers.bson"))[:warmstart_solvers]
bc_init_solvers = BSON.load(string(out,"bc_init_solvers.bson"))[:bc_init_solvers]
der_solvers = BSON.load(string(out,"der_solvers.bson"))[:der_solvers]
igcl_solvers = BSON.load(string(out,"igcl_solvers.bson"))[:igcl_solvers]
er_solvers = BSON.load(string(out,"er_solvers.bson"))[:er_solvers]
ewc_solvers = BSON.load(string(out,"ewc_solvers.bson"))[:ewc_solvers]



## Cumulative_rewards
p_rew = plot_cumulative_rewards(scratch_solvers, label="scratch", legend=:topleft, show_lines=true)
plot_cumulative_rewards(warmstart_solvers, p=p_rew, label="warm start")
plot_cumulative_rewards(bc_init_solvers, p=p_rew, label="BC Init")
plot_cumulative_rewards(der_solvers, p=p_rew, label="discriminative experience replay")
plot_cumulative_rewards(igcl_solvers, p=p_rew, label="IGCL")
plot_cumulative_rewards(er_solvers, p=p_rew, label="experience replay")
plot_cumulative_rewards(ewc_solvers, p=p_rew, label="ewc")
savefig(string(out,"cumulative_reward.pdf"))

## Jumpstart Performance
p_jump = plot_jumpstart(scratch_solvers, label="scratch", legend=:right)
plot_jumpstart(warmstart_solvers, p=p_jump, label="warm start")
plot_jumpstart(bc_init_solvers, p=p_jump, label="BC Init")
plot_jumpstart(der_solvers, p=p_jump, label="discriminative experience replay")
plot_jumpstart(igcl_solvers, p=p_jump, label="IGCL")
plot_jumpstart(er_solvers, p=p_jump, label="experience replay")
plot_jumpstart(ewc_solvers, p=p_jump, label="ewc")
savefig(string(out,"jumpstart.pdf"))

## Peak performance
p_perf = plot_peak_performance(scratch_solvers, label="scratch", legend=:bottomleft)
plot_peak_performance(warmstart_solvers, p=p_perf, label="warm start")
plot_peak_performance(bc_init_solvers, p=p_perf, label="BC Init")
plot_peak_performance(der_solvers, p=p_perf, label="discriminative experience replay")
plot_peak_performance(igcl_solvers, p=p_perf, label="IGCL")
plot_peak_performance(er_solvers, p=p_perf, label="experience replay")
plot_peak_performance(ewc_solvers, p=p_perf, label="ewc")
savefig(string(out,"peak_performance.pdf"))

## Steps to threshold
p_thresh = plot_steps_to_threshold(scratch_solvers, .99, label="scratch")
plot_steps_to_threshold(warmstart_solvers, .99, p=p_thresh, label="warm start")
plot_steps_to_threshold(bc_init_solvers, .99, p=p_thresh, label="BC Init")
plot_steps_to_threshold(der_solvers, .99, p=p_thresh, label="discriminative experience replay")
plot_steps_to_threshold(igcl_solvers, .99, p=p_thresh, label="IGCL")
plot_steps_to_threshold(er_solvers, .99, p=p_thresh, label="experience replay")
plot_steps_to_threshold(ewc_solvers, .99, p=p_thresh, label="ewc")
savefig(string(out,"steps_to_threshold.pdf"))

## Catastrophic forgetting
p_forget = Crux.plot_forgetting(scratch_solvers, label="scratch", legend=:bottomleft, size=(900,1200))
Crux.plot_forgetting(warmstart_solvers, p=p_forget, label="warm start")
Crux.plot_forgetting(bc_init_solvers, p=p_forget, label="BC Init")
Crux.plot_forgetting(der_solvers, p=p_forget, label="discriminative experience replay")
Crux.plot_forgetting(igcl_solvers, p=p_forget, label="IGCL")
Crux.plot_forgetting(er_solvers, p=p_forget, label="experience replay")
Crux.plot_forgetting(ewc_solvers, p=p_forget, label="ewc")
savefig(string(out,"catastrophic_forgetting.pdf"))

