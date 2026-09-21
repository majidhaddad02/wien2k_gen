# forge_wizard completion for bash

_forge_wizard() {
    local cur
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    COMPREPLY=($(compgen -W "--help --version" -- "$cur"))
}

complete -o bashdefault -o default -F _forge_wizard forge_wizard
