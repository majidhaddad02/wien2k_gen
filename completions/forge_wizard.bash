# forge_wizard completion for bash

_forge_wizard() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="--help --version"

    COMPREPLY=($(compgen -W "${opts}" -- ${cur}))
    return 0
}
complete -F _forge_wizard forge_wizard