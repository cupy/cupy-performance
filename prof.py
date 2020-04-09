import argparse

import cupy_prof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--keep', type=str, default=None)
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-c', '--csv', action='store_true')
    args, paths = parser.parse_known_args()
    collector = cupy_prof.Collector()
    collector.collect(paths)
    for bench_class in collector.benchmarks:
        bench = bench_class()
        cupy_prof.Measure(bench).measure(plot=args.plot, csv=args.csv)


if __name__ == '__main__':
    main()
