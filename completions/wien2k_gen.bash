# wien2k_gen completion for bash
# Place in /usr/share/bash-completion/completions/wien2k_gen

_wien2k_gen() {
    local cur prev words cword
    _init_completion -n || return
    local cmd="${words[1]}"
    local subcmds="generate submit benchmark diagnostics analyze tui"
    local global_opts="--verbose -v --quiet -q --json --config --backend --log-file --version --help"

    # سوییچ بین ساب‌کامندها
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
            local opts="--nodes --cores --omp --mode --dry-run --export --overwrite"
            local mode_vals="mpi hybrid kpoint"
            if [[ "$prev" == "--mode" ]]; then COMPREPLY=( $(compgen -W "$mode_vals" -- "$cur") )
            elif [[ "$prev" == "--export" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        submit)
            local opts="--partition --nodes --ntasks --time --mem --job-name --dependency --dry-run --export"
            if [[ "$prev" == "--export" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        benchmark)
            local opts="--type --max-cores --walltime --output --skip-cleanup"
            local type_vals="real synthetic"
            if [[ "$prev" == "--type" ]]; then COMPREPLY=( $(compgen -W "$type_vals" -- "$cur") )
            elif [[ "$prev" == "--output" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        diagnostics)
            local opts="--export --full"
            if [[ "$prev" == "--export" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        analyze)
            local opts="--log --code --export"
            local code_vals="wien2k vasp qe"
            if [[ "$prev" == "--log" || "$prev" == "--export" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            elif [[ "$prev" == "--code" ]]; then COMPREPLY=( $(compgen -W "$code_vals" -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        tui)
            COMPREPLY=( $(compgen -W "--compact $global_opts" -- "$cur") )
            ;;
        *)
            COMPREPLY=( $(compgen -W "$global_opts" -- "$cur") )
            ;;
    esac
}

complete -F _wien2k_gen wien2k_gen