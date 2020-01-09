import importlib.util
import inspect
import os
import re
import sys


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo


class Collector(object):
    def __init__(self):
        self.benchmarks = []

    def process_file(self, path):
        module = load_module('', path)
        self.filter_benchmarks(module)
        sys.stdout.write("\n")

    def collect(self, paths):
        # Access the paths, import the bench_*py files and
        # Access the Classes inside
        file_regex = re.compile(r'^bench_.+py$')
        for path in paths:
            if path[-3:] == '.py':
                print(path)
                self.process_file(path)
            else:
                for root, _, files in os.walk(path):
                    for file_n in files:
                        if file_regex.match(file_n) is None:
                            continue
                        b_path = os.path.join(root, file_n)
                        print(b_path)
                        self.process_file(b_path)

    def filter_benchmarks(self, module):
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                if 'Benchmark' in obj.__name__:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    self.benchmarks.append(obj)


if __name__ == '__main__':
    Collector().collect(sys.argv[1])
