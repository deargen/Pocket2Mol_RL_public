# Download file from S3
wget https://pocket2mol-rl-public.s3.amazonaws.com/test_cache.tar
wget https://pocket2mol-rl-public.s3.amazonaws.com/test_outputs.tar

# Download checkpoint
mkdir -p checkpoints
wget https://pocket2mol-rl-public.s3.amazonaws.com/Pocket2Mol_RL.pt -O checkpoints/Pocket2Mol_RL.pt

# Extract the tar file
tar -xvf test_cache.tar
tar -xvf test_outputs.tar