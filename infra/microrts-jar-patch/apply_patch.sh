#!/usr/bin/env bash
# Patch gym-microrts 0.3.2's vendored microrts.jar so JNIGridnetVecClient
# exposes the pre-reset terminal observation (see JNIGridnetVecClient.java).
#
# Run INSIDE the research container (needs the JDK + the installed package):
#   bash /workspace/infra/microrts-jar-patch/apply_patch.sh
#
# Idempotent: keeps a one-time microrts.jar.orig backup and recompiles from
# the checked-in source every run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAR="$(python -c 'import gym_microrts, os; print(os.path.join(gym_microrts.__path__[0], "microrts", "microrts.jar"))')"
echo "[patch] target jar: $JAR"

if [ ! -f "$JAR.orig" ]; then
    cp "$JAR" "$JAR.orig"
    echo "[patch] backed up original to $JAR.orig"
fi

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

# --release 8 keeps the class-file version compatible with the rest of the
# jar (built 2021, Java 8 era) regardless of the installed JDK.
javac --release 8 -encoding UTF-8 -cp "$JAR.orig" -d "$BUILD" "$HERE/JNIGridnetVecClient.java"
ls "$BUILD/tests/"

( cd "$BUILD" && jar uf "$JAR" tests/*.class )
echo "[patch] updated $(basename "$JAR") with:"
unzip -l "$JAR" | grep -E "tests/JNIGridnetVecClient" || true

# Smoke check: every patched field/method must be visible on the class.
for sym in terminalObservation opponentAction playerIds setPlayerIds fullState fullGlobals terminalFullState terminalFullGlobals counterfactualFullState counterfactualFullGlobals computeCounterfactual; do
    javap -cp "$JAR" tests.JNIGridnetVecClient | grep -q "$sym" \
        && echo "[patch] OK: $sym present" \
        || { echo "[patch] FAIL: $sym missing"; exit 1; }
done
