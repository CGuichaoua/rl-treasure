#!/bin/bash

# List of supported agents
agents=("tabular_q" "near_sighted" "oblivious" "DQN" "DQN-smaller" "DQN-larger" "PPO" "PPO-smaller" "PPO-larger")

# List of supported environments
environments=("fixed" "static" "base")

epochs=2000

# Loop through each agent
for agent in "${agents[@]}"; do
    # Loop through each environment
    for environment in "${environments[@]}"; do
        start_time=$(date +%s)
        echo "Running $agent on $environment"
        python -m treasure_hunt.main --agent $agent --env $environment --epochs $epochs --no-show
        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        echo "Runtime for $agent on $environment: $runtime seconds"
    done
done