#compdef forge_sbatch

_forge_sbatch() {
    local -a actions
    local cmd
    integer skip=0 i
    local w

    actions=(
        'generate:Generate SBATCH script from topology and config'
        'validate:Check script syntax and resource constraints'
        'preview:Render formatted script to stdout'
        'submit:Submit script to SLURM controller'
    )

    cmd=""
    for ((i=2; i<=$#words; i++)); do
        w="${words[i]}"
        if (( skip )); then
            skip=0
            continue
        fi
        if (( i == CURRENT )); then
            break
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
        _arguments -C \
            '(-v --verbose)'{-v,--verbose}'[Increase verbosity]' \
            '(-q --quiet)'{-q,--quiet}'[Suppress non-essential output]' \
            '--json[Return results as JSON]' \
            '--config=[Path to custom config file]:file:_files' \
            '--log-file=[Redirect logs to file]:file:_files' \
            '--backend=[DFT backend context]:backend:(wien2k qe vasp cp2k)' \
            '1:action:->action'
        if [[ "$state" == action ]]; then
            _describe -t actions 'action' actions
        fi
        return
    fi

    case "$cmd" in
        generate)
            _arguments \
                '(-o --output)'{-o,--output}'=[Output script path]:file:_files' \
                '(-J --job-name)'{-J,--job-name}'=[SLURM job name]' \
                '(-p --partition)'{-p,--partition}'=[Target partition/queue]' \
                '(-N --nodes)'{-N,--nodes}'=[Number of nodes]' \
                '(-n --ntasks)'{-n,--ntasks}'=[Total tasks (0 = auto)]' \
                '(-c --cpus-per-task)'{-c,--cpus-per-task}'=[CPUs per task]' \
                '--mem=[Memory per node]:mem:(4G 8G 16G 32G 64G 128G)' \
                '(-t --time)'{-t,--time}'=[Walltime limit]:time:(01:00:00 24:00:00 48:00:00 1-00:00:00)' \
                '--dependency=[Job dependency (e.g., afterok:12345)]' \
                '--qos=[Quality of service]:qos:(normal debug long high)' \
                '--gres=[Generic resources (e.g., gpu:a100:2)]:gres:(gpu:1 gpu:2 gpu:4 gpu:a100:1 gpu:a100:2)' \
                '--dry-run[Print to stdout without writing]' \
                '--backup[Rotate existing script]' \
                '--preview[Preview script in terminal after generation]'
            ;;
        validate)
            _arguments \
                '1:script:_files' \
                '--strict[Fail on warnings, not just errors]'
            ;;
        preview)
            _arguments \
                '1:script:_files' \
                '--highlight[Apply syntax highlighting (if Rich available)]'
            ;;
        submit)
            _arguments \
                '1:script:_files' \
                '--dry-run[Validate and print submission response only]' \
                '--watch[Poll job status after submission]'
            ;;
    esac
}
