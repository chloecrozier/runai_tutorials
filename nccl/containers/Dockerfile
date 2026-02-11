# Multi-node PyTorch + OpenMPI + NCCL tests + SSH for RunAI distributed workloads
# Image name: runai-nccl-pytorch-26.01
FROM nvcr.io/nvidia/pytorch:26.01-py3

# Install SSH server/client and build tools
# NOTE: OpenMPI is already bundled in the NGC PyTorch image (under /opt/hpcx/ompi)
#       so we do NOT install libopenmpi-dev / openmpi-bin via apt
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        openssh-server \
        openssh-client \
        git \
        && \
    rm -rf /var/lib/apt/lists/*

# Configure sshd for MPI (passwordless root login when keys are mounted by the job)
RUN mkdir -p /run/sshd /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keygen -A && \
    sed -i 's/#PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
    echo "UserKnownHostsFile=/dev/null" >> /etc/ssh/ssh_config

# Build nccl-tests with MPI support
# Dynamically find MPI_HOME from the NGC container's bundled OpenMPI (avoids hardcoded paths)
WORKDIR /workspace
RUN MPI_HOME=$(dirname $(dirname $(which mpirun))) && \
    echo "Detected MPI_HOME=$MPI_HOME" && \
    git clone https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
    make MPI=1 CUDA_HOME=/usr/local/cuda MPI_HOME=$MPI_HOME && \
    cd /workspace

# NCCL test binaries (all_reduce_perf, etc.) are in build/
ENV PATH="/workspace/nccl-tests/build:${PATH}"

WORKDIR /workspace

# Start sshd in background then exec the user command (for RunAI MPI jobs)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
