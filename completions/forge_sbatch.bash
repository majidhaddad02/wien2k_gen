# forge_sbatch completion for bash

_forge_sbatch() {
    local cur prev words cword
    _init_completion -n || return
    local cmd="${words[1]}"
    local subcmds="generate validate preview submit"
    local global_opts="--verbose -v --quiet -q --json --backend --config --log-file --help"

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
            if [[ "$prev" == "-o" || "$prev" == "--output" ]]; then
                COMPREPLY=( $(compgen -f -- "$cur") )
            elif [[ "$prev" == "--mem" ]]; then
                COMPREPLY=( $(compgen -W "4G 8G 16G 32G 64G 128G" -- "$cur") )
            elif [[ "$prev" == "-t" || "$prev" == "--time" ]]; then
                COMPREPLY=( $(compgen -W "01:00:00 24:00:00 48:00:00 1-00:00:00" -- "$cur") )
            elif [[ "$prev" == "--gres" ]]; then
                COMPREPLY=( $(compgen -W "gpu:1 gpu:2 gpu:4 gpu:a100:1 gpu:a100:2" -- "$cur") )
            elif [[ "$prev" == "--qos" ]]; then
                COMPREPLY=( $(compgen -W "normal debug long high" -- "$cur") )
            else
                COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        validate)
            local sub_opts="--strict --highlight"
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=( $(compgen -f -- "$cur") )
            else
                COMPREPLY=( $(compgen -W "$sub_opts $global_opts" -- "$cur") )
            fi
            ;;
        preview)
            local sub_opts="--highlight"
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=( $(compgen -f -- "$cur") )
            else
                COMPREPLY=( $(compgen -W "$sub_opts $global_opts" -- "$cur") )
            fi
            ;;
        submit)
            local sub_opts="--dry-run --watch"
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=( $(compgen -f -- "$cur") )
            else
                COMPREPLY=( $(compgen -W "$sub_opts $global_opts" -- "$cur") )
            fi
            ;;
        *)
            COMPREPLY=( $(compgen -W "$global_opts" -- "$cur") )
            ;;
    esac
}

complete -F _forge_sbatch forge_sbatch
