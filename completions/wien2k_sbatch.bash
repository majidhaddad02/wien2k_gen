# wien2k_sbatch completion for bash

_wien2k_sbatch() {
    local cur prev words cword
    _init_completion -n || return
    local cmd="${words[1]}"
    local subcmds="generate validate preview submit"
    local global_opts="--verbose -v --quiet -q --json --backend --help"

    if [[ $cword -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "$subcmds" -- "$cur") )
        return
    fi

    if [[ $cword -eq 2 && ! " $subcmds " == *" $prev "* ]]; then
        COMPREPLY=( $(compgen -W "$subcmds" -- "$cur") )
        return
    fi

    case "$cmd" in
        generate)
            local opts="-J --job-name -p --partition -N --nodes -n --ntasks -c --cpus-per-task --mem -t --time --dependency --qos --gres --output -o --dry-run --backup --preview"
            if [[ "$prev" == *(-o|--output) ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            elif [[ "$prev" == "--mem" ]]; then COMPREPLY=( "4G" "8G" "16G" "32G" )
            elif [[ "$prev" == "-t" || "$prev" == "--time" ]]; then COMPREPLY=( "01:00:00" "24:00:00" "48:00:00" "1-00:00:00" )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        validate|preview|submit)
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=( $(compgen -f -- "$cur") )
            else
                local sub_opts="--dry-run --watch --strict --highlight"
                COMPREPLY=( $(compgen -W "$sub_opts $global_opts" -- "$cur") )
            fi
            ;;
        *)
            COMPREPLY=( $(compgen -W "$global_opts" -- "$cur") )
            ;;
    esac
}

complete -F _wien2k_sbatch wien2k_sbatch