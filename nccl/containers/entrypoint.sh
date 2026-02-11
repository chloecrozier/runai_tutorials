#!/bin/bash
set -e

# Start sshd so MPI can SSH between workers
if command -v sshd >/dev/null 2>&1; then
    /usr/sbin/sshd
fi

exec "$@"
