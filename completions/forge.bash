# forge completion for bash
# Works with or without the bash-completion package.

_forge() {
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

    local subcmds="generate submit benchmark diagnostics hardware analyze tui monitor run workflow diagnose optimize screen predict converge advise history analyze-bands"
    local global_opts="--verbose -v --quiet -q --json --config --backend --log-file --version --plain --no-color --help"

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
            --config=*|--backend=*|--log-file=*|-v|-vv|-vvv|-q|--verbose|--quiet|--json|--plain|--no-color|--help|--version)
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
            local opts="--nodes --cores --omp --mode --target --max-cores --reserve-os-cores --memory-limit --dry-run --export --overwrite --scheduler -S --gpu --gpu-mixed-precision --manual"
            if [[ "$prev" == "--mode" ]]; then COMPREPLY=( $(compgen -W "mpi hybrid kpoint" -- "$cur") )
            elif [[ "$prev" == "--target" ]]; then COMPREPLY=( $(compgen -W "time memory balanced cost" -- "$cur") )
            elif [[ "$prev" == "--scheduler" || "$prev" == "-S" ]]; then COMPREPLY=( $(compgen -W "slurm pbs lsf sge auto" -- "$cur") )
            elif [[ "$prev" == "--export" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        submit)
            local opts="--scheduler -S --partition --nodes --ntasks --time --mem --job-name --dependency --dry-run --export"
            if [[ "$prev" == "--scheduler" || "$prev" == "-S" ]]; then COMPREPLY=( $(compgen -W "slurm pbs lsf sge auto" -- "$cur") )
            elif [[ "$prev" == "--export" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        benchmark)
            local opts="--type --max-cores --walltime --output --skip-cleanup --scheduler -S"
            if [[ "$prev" == "--type" ]]; then COMPREPLY=( $(compgen -W "real synthetic" -- "$cur") )
            elif [[ "$prev" == "--scheduler" || "$prev" == "-S" ]]; then COMPREPLY=( $(compgen -W "slurm pbs lsf auto" -- "$cur") )
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
            if [[ "$prev" == "--log" || "$prev" == "--export" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            elif [[ "$prev" == "--code" ]]; then COMPREPLY=( $(compgen -W "wien2k vasp qe" -- "$cur") )
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
            COMPREPLY+=( $(compgen -f -- "$cur") )
            ;;
        workflow)
            local opts="--case --steps --output"
            local actions="create list visualize"
            if [[ "$prev" == "--output" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            elif [[ "$prev" == "$cmd" ]]; then COMPREPLY=( $(compgen -W "$actions" -- "$cur") )
            else COMPREPLY=( $(compgen -W "$actions $opts $global_opts" -- "$cur") )
            fi
            ;;
        diagnose)
            local opts="--log"
            if [[ "$prev" == "--log" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        optimize)
            local opts="--case --budget --target --strategy --simulated --verbose -v"
            if [[ "$prev" == "--strategy" ]]; then COMPREPLY=( $(compgen -W "gp_ei bohb" -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
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
            if [[ "$prev" == "--mode" ]]; then COMPREPLY=( $(compgen -W "kpoints rkmax both" -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        advise)
            local opts="--case --nmat --kpoints --cores --target --plain --json"
            if [[ "$prev" == "--target" ]]; then COMPREPLY=( $(compgen -W "time energy cost balanced" -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
            ;;
        history)
            local opts="--list --show --similar-to --limit --export --format --backend"
            if [[ "$prev" == "--format" ]]; then COMPREPLY=( $(compgen -W "csv json" -- "$cur") )
            elif [[ "$prev" == "--export" ]]; then COMPREPLY=( $(compgen -f -- "$cur") )
            elif [[ "$prev" == "--backend" ]]; then COMPREPLY=( $(compgen -W "wien2k qe vasp cp2k" -- "$cur") )
            else COMPREPLY=( $(compgen -W "$opts $global_opts" -- "$cur") )
            fi
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

complete -o bashdefault -o default -F _forge forge
