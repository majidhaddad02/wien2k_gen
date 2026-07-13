# forge completion for bash
# Place in /usr/share/bash-completion/completions/forge

_forge() {
    local cur prev words cword
    _init_completion -n || return
    local cmd="${words[1]}"
    local subcmds="generate submit benchmark diagnostics hardware analyze tui monitor run workflow diagnose optimize screen predict converge advise history analyze-bands"
    local global_opts="--verbose -v --quiet -q --json --config --backend --log-file --version --plain --no-color --help"

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
            local opts="--nodes --cores --omp --mode --target --max-cores --reserve-os-cores --memory-limit --dry-run --export --overwrite --scheduler -S --gpu --gpu-mixed-precision --manual"
            local mode_vals="mpi hybrid kpoint"
            local target_vals="time memory balanced cost"
            local sched_vals="slurm pbs lsf sge auto"
            if [[ "$prev" == "--mode" ]]; then COMPREPLY=( $(compgen -W "$mode_vals" -- "$cur") )
            elif [[ "$prev" == "--target" ]]; then COMPREPLY=( $(compgen -W "$target_vals" -- "$cur") )
            elif [[ "$prev" == "--scheduler" || "$prev" == "-S" ]]; then COMPREPLY=( $(compgen -W "$sched_vals" -- "$cur") )
            elif [[ "$prev" == "--export" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        submit)
            local opts="--scheduler -S --partition --nodes --ntasks --time --mem --job-name --dependency --dry-run --export"
            local sched_vals="slurm pbs lsf sge auto"
            if [[ "$prev" == "--scheduler" || "$prev" == "-S" ]]; then COMPREPLY=( $(compgen -W "$sched_vals" -- "$cur") )
            elif [[ "$prev" == "--export" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        benchmark)
            local opts="--type --max-cores --walltime --output --skip-cleanup --scheduler -S"
            local type_vals="real synthetic"
            local sched_vals="slurm pbs lsf auto"
            if [[ "$prev" == "--type" ]]; then COMPREPLY=( $(compgen -W "$type_vals" -- "$cur") )
            elif [[ "$prev" == "--scheduler" || "$prev" == "-S" ]]; then COMPREPLY=( $(compgen -W "$sched_vals" -- "$cur") )
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
        hardware)
            local opts="--recommend -r --case"
            if [[ "$prev" == "--case" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
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
        monitor)
            local opts="--interval --output"
            if [[ "$prev" == "--output" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        run)
            local opts="--auto-retry --no-retry --max-retries --poll"
            COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            ;;
        workflow)
            local opts="--case --steps --output"
            local actions="create list visualize"
            if [[ "$prev" == "--output" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            elif [[ $cword -eq 2 ]]; then COMPREPLY=( $(compgen -W "$actions" -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        diagnose)
            local opts="--log"
            if [[ "$prev" == "--log" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        optimize)
            local opts="--case --budget --target --simulated --verbose -v"
            COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            ;;
        screen)
            local opts="--formula --elements --mp-id --max --api-key --output"
            if [[ "$prev" == "--output" ]]; then COMPREPLY=( $(compgen -d -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        predict)
            local opts="--case --struct --no-history"
            if [[ "$prev" == "--struct" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        converge)
            local opts="--case --mode --tolerance --kpoints --rkmax"
            local mode_vals="kpoints rkmax both"
            if [[ "$prev" == "--mode" ]]; then COMPREPLY=( $(compgen -W "$mode_vals" -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        advise)
            local opts="--case --nmat --kpoints --cores --target --plain --json"
            local tgt_vals="time energy cost balanced"
            if [[ "$prev" == "--target" ]]; then COMPREPLY=( $(compgen -W "$tgt_vals" -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        history)
            local opts="--list --show --similar-to --limit"
            COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            ;;
        analyze-bands)
            local opts="--case --output --dos"
            if [[ "$prev" == "--output" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        *)
            COMPREPLY=( $(compgen -W "$global_opts" -- "$cur") )
            ;;
    esac
}

complete -F _forge forge
