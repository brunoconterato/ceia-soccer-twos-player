from src.experiments.ppo_multiagent.fixPickle import fix_pickle
from src.experiments.ppo_multiagent.export import export
from src.experiments.ppo_multiagent.experiment import run_experiment

if __name__ == "__main__":
    # run_experiment()
    # export()
    fix_pickle("/home/bruno/Workspace/soccer-tows-player/src/ray_results/Testing_env/PPO_Soccer_18d23_00000_0_2021-11-24_20-34-41/params.pkl")