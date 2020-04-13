import argparse
import pickle

import cupy_prof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-c', '--csv', action='store_true')
    parser.add_argument('-d', '--dump-pickle', type=str, default=None)
    parser.add_argument(
        '-r', '--repo', nargs='+', type=str, default=None, required=False)
    parser.add_argument(
        '-cm', '--commits', nargs='+', type=str, default=None, required=False)
    args, paths = parser.parse_known_args()
    if args.repo is None and args.commits is None:
        collector = cupy_prof.Collector()
        collector.collect(paths)
        dfs = {}
        for bench_class in collector.benchmarks:
            bench = bench_class()
            df = cupy_prof.Measure(bench).measure(csv=args.csv, plot=args.plot)
            if args.dump_pickle is not None:
                dfs[bench_class.__name__] = df
        if args.dump_pickle is not None:
            with open(args.dump_pickle, 'wb') as handle:
                pickle.dump(dfs, handle)
    elif ((args.repo != args.commits)
            and (args.repo is None or args.commits is None)):
        raise ValueError('--repo and --commits need to be specified together')
    else:
        if len(args.repo) != 1 and len(args.repo) != len(args.commits):
            raise ValueError(
                '--repo must be a single repository or one per commit')
        comparer = cupy_prof.Comparer(args.commits, args.repo, paths)
        comparer.compare(csv=args.csv, plot=args.plot)


if __name__ == '__main__':
    main()
