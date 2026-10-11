#!/bin/sh
set -eu
umask 077
mkdir -p "$HOME" "$DSH_HOME" "$DSH_AGENTS_HOME"
# DSH accepts one concrete interface, not a wildcard bind.
bind="$(node --input-type=module -e 'import os from "node:os"; const address=Object.values(os.networkInterfaces()).flat().find(item=>item?.family==="IPv4"&&!item.internal)?.address; if(!address)process.exit(1); process.stdout.write(address)')"
# Launcher flags must precede the first app flag (Commander passes the rest through).
exec node /opt/dsh/node_modules/@deepseek-ai/dsh/lib/bin.js "$@" --host "$bind"
