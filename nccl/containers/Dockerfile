# Multi-node PyTorch + OpenMPI + NCCL tests + SSH for RunAI distributed workloads
FROM nvcr.io/nvidia/pytorch:26.01-py3

# Install SSH server/client and build tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        openssh-server \
        openssh-client \
        git \
        && \
    rm -rf /var/lib/apt/lists/*

# Configure sshd for MPI
# StrictModes off: the MPI operator mounts /root/.ssh as a k8s secret with 777 perms.
# With StrictModes on (default), sshd silently ignores authorized_keys.
RUN mkdir -p /run/sshd && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
    echo "UserKnownHostsFile=/dev/null" >> /etc/ssh/ssh_config

# Build nccl-tests with MPI support
WORKDIR /workspace
RUN MPI_HOME=$(dirname $(dirname $(which mpirun))) && \
    echo "Detected MPI_HOME=$MPI_HOME" && \
    git clone https://github.com/NVIDIA/nccl-tests.git && \
    cd nccl-tests && \
    make MPI=1 CUDA_HOME=/usr/local/cuda MPI_HOME=$MPI_HOME && \
    cd /workspace

ENV PATH="/workspace/nccl-tests/build:${PATH}"

COPY verify_image.sh /workspace/verify_image.sh

WORKDIR /workspace
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
