import argparse

import cupy_prof


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-k', 'select', type=str, default=None)
    args, paths = parser.parse_known_args()
    collector = cupy_prof.Collector()
    collector.collect(paths)
    for bench_class in collector.benchmarks:
        bench = bench_class()
        cupy_prof.Measure(bench).measure()


if __name__ == '__main__':
    main()
