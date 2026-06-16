#compdef wien2k_sbatch

_wien2k_sbatch() {
    local curcontext="$curcontext" state line
    typeset -A opt_args

    _arguments -C \
        '(- :)'{-v,--verbose}'[Increase verbosity]' \
        '(- :)'{-q,--quiet}'[Suppress output]' \
        '(- :)--json[JSON output]' \
        '(- :)--backend=[Backend]:(wien2k vasp qe)' \
        '1:action:->action' \
        '*::args:->args'

    case $state in
        (action)
            local -a actions
            actions=(
                'generate:Generate SBATCH script'
                'validate:Validate script syntax'
                'preview:Preview script'
                'submit:Submit script to SLURM'
            )
            _describe -t actions 'action' actions
            ;;
        (args)
            case $line[1] in
                generate)
                    _arguments \
                        '(-o --output)'{-o,--output}'[Output path]:_files' \
                        '(-J --job-name)'{-J,--job-name}'[Job name]' \
                        '(-p --partition)'{-p,--partition}'[Partition]:_normal' \
                        '(-N --nodes)'{-N,--nodes}'[Nodes]' \
                        '(-n --ntasks)'{-n,--ntasks}'[Tasks]' \
                        '(-c --cpus-per-task)'{-c,--cpus-per-task}'[CPUs/task]' \
                        '--mem=[Memory]:(4G 8G 16G 32G)' \
                        '(-t --time)'{-t,--time}'[Time]:(01:00:00 24:00:00 48:00:00)' \
                        '--dependency=[Dep]' \
                        '--qos=[QoS]' \
                        '--gres=[Res]' \
                        '--dry-run' \
                        '--preview'
                    ;;
                validate)
                    _arguments \
                        '1:script:_files' \
                        '--strict'
                    ;;
                preview)
                    _arguments \
                        '1:script:_files' \
                        '--highlight'
                    ;;
                submit)
                    _arguments \
                        '1:script:_files' \
                        '--dry-run' \
                        '--watch'
                    ;;
            esac
            ;;
    esac
}

_wien2k_sbatch