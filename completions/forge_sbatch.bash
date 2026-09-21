# forge_sbatch completion for bash
# Works with or without the bash-completion package.

_forge_sbatch() {
    local cur prev words cword cmd skip i w
    COMPREPLY=()

    if declare -F _init_completion >/dev/null 2>&1; then
        _init_completion || return
    else
        words=("${COMP_WORDS[@]}")
        cword=${COMP_CWORD}
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
    fi

    local subcmds="generate validate preview submit"
    local global_opts="--verbose -v --quiet -q --json --backend --config --log-file --help"

    skip=0
    cmd=""
    for ((i=1; i<cword; i++)); do
        w="${words[i]}"
        if (( skip )); then
            skip=0
            continue
        fi
        case "$w" in
            --config|--backend|--log-file)
                skip=1
                ;;
            --config=*|--backend=*|--log-file=*|-v|-vv|-vvv|-q|--verbose|--quiet|--json|--help)
                ;;
            -*)
                ;;
            *)
                cmd="$w"
                break
                ;;
        esac
    done

    if [[ -z "$cmd" ]]; then
        case "$prev" in
            --config|--log-file)
                COMPREPLY=( $(compgen -f -- "$cur") )
                return
                ;;
            --backend)
                COMPREPLY=( $(compgen -W "wien2k qe vasp cp2k" -- "$cur") )
                return
                ;;
        esac
        COMPREPLY=( $(compgen -W "$subcmds $global_opts" -- "$cur") )
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
            if [[ "$prev" == "$cmd" ]]; then
                COMPREPLY=( $(compgen -f -- "$cur") )
            else
                COMPREPLY=( $(compgen -W "$sub_opts $global_opts" -- "$cur") )
            fi
            ;;
        preview)
            local sub_opts="--highlight"
            if [[ "$prev" == "$cmd" ]]; then
                COMPREPLY=( $(compgen -f -- "$cur") )
            else
                COMPREPLY=( $(compgen -W "$sub_opts $global_opts" -- "$cur") )
            fi
            ;;
        submit)
            local sub_opts="--dry-run --watch"
            if [[ "$prev" == "$cmd" ]]; then
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

complete -o bashdefault -o default -F _forge_sbatch forge_sbatch
