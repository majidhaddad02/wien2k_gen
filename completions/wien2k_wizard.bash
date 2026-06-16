# wien2k_wizard completion for bash

_wien2k_wizard() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="--help --version"

    COMPREPLY=($(compgen -W "${opts}" -- ${cur}))
    return 0
}
complete -F _wien2k_wizard wien2k_wizard