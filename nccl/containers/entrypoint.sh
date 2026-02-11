#!/bin/bash
set -e
# Start sshd so MPI can SSH between workers (RunAI may inject authorized_keys at runtime)
if command -v sshd >/dev/null 2>&1; then
    /usr/sbin/sshd
fi
exec "$@"
