#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ] && [ -z "${ZSH_VERSION:-}" ]; then
    echo "env.sh must be sourced from bash or zsh" >&2
    return 1 2>/dev/null || exit 1
fi

REPO_ROOT="${PWD}"
VENV_DIR="${REPO_ROOT}/.venv"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo ".venv virtualenv was not found at ${VENV_DIR}" >&2
    return 1 2>/dev/null || exit 1
fi

if [ "${VIRTUAL_ENV:-}" != "${VENV_DIR}" ]; then
    # Source the repo-local virtualenv if it is not already active.
    . "${VENV_DIR}/bin/activate"
fi

export PIP_CACHE_DIR="${VIRTUAL_ENV}/.pip-cache"
export TILELANG_CACHE_DIR="${VIRTUAL_ENV}/.tilelang-cache"
export TILELANG_TMP_DIR="${TILELANG_CACHE_DIR}/tmp"

mkdir -p "${PIP_CACHE_DIR}" "${TILELANG_CACHE_DIR}" "${TILELANG_TMP_DIR}"

echo "VIRTUAL_ENV=${VIRTUAL_ENV}"
echo "VIRTUAL_ENV_PROMPT=${VIRTUAL_ENV_PROMPT:-}"
echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR}"
echo "TILELANG_TMP_DIR=${TILELANG_TMP_DIR}"
