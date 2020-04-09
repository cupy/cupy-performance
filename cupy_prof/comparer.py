import multiprocessing
import os
import subprocess

import cupy_prof


class _GitCommandError(Exception):
    pass


def _run_benchmarks(benchmarks, queue):
    collector = cupy_prof.Collector()
    collector.collect(benchmarks)
    dfs = {}
    for bench_class in collector.benchmarks:
        bench = bench_class()
        df = cupy_prof.Measure(bench).measure(plot=False, csv=False)
        dfs[bench_class.__name__] = df
    queue.put(dfs)


def _git(repository, command, stdout=None, stderr=None):
    # Taken mostly from chainer-test
    cmd = ['git']
    if repository is not None:
        assert os.path.isdir(repository)
        cmd += ['-C', repository]
    cmd += list(command)

    print('**GIT** {}'.format(' '.join(cmd)))

    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise _GitCommandError(
            "Git command failed with code {}".format(proc.returncode),
            cmd)
    if stdout is not None:
        stdout = stdout.decode('utf8')


def _build(repository, command):
    subprocess.check_call(command)


class Comparer(object):
    """ Compares different commits of a GH repo for the same benchmarks

    Args:
        commits (List str) commit hashes or branches to compare
        repository (str) path to the github repository to compare
        run_args (list str) list with the arguments that prof.py was called
            it is used to call prof.py recursively to evaluate the actual
            benchmarks
    """
    def __init__(self, commits, repository, benchmarks):
        self.commits = commits
        self.repository = repository
        self.benchmarks = benchmarks

    def compare(self):
        # We use spawn to reload the newly installed environment
        multiprocessing.set_start_method('spawn')
        dfs = []
        for commit in self.commits:
            # _git(self.repository, ['clean', '-f', '-x'])
            # _git(self.repository, ['checkout', commit])
            # # Install the module calling pip
            # # TODO(ecastill) abstract this by adding config files
            # # with build commands
            # _build(self.repository, ['pip', 'install', '-e',
            #        self.repository])

            try:
                _build(self.repository, ['pip', 'uninstall', commit,
                       self.repository])
            except Exception:
                pass
            _build(self.repository, ['pip', 'install', commit])

            # Now we need to spawn a new process that gets the
            # newly installed module and returns the benchmark results
            queue = multiprocessing.Queue()
            args = (self.benchmarks, queue)
            bench = multiprocessing.Process(target=_run_benchmarks, args=args)
            bench.start()
            dfs.append(queue.get())
            bench.join()

        dfs = self._compare_results(dfs)
        # Now we generate the plots or csvs if needed
        # We will need to retrieve the actual benchmarks
        collector = cupy_prof.Collector()
        collector.collect(self.benchmarks)
        bench_classes = {}
        for bench_class in collector.benchmarks:
            bench_classes[bench_class.__name__] = bench_class()
        for bench in dfs:
            bench_classes[bench].plot(dfs[bench])

    def _compare_results(self, dfs):
        # We create a new df that joins all the experiments
        # but with a backend per each commit
        res = {}
        for bench in dfs[0]:
            base_df = dfs[0][bench]
            commit = self.commits[0]
            base_df['backend'] = commit + '-' + base_df['backend'].astype(str)
            for i, df in enumerate(dfs[1:]):
                commit = self.commits[1+i]
                df = df[bench]
                df['backend'] = commit + '-' + df['backend'].astype(str)
                joined_df = base_df.append(df)
            res[bench] = joined_df
            # Plot the benchmarks
        return res


if __name__ == '__main__':
    # run_args = ['python', 'prof.py', 'benchmarks']
    Comparer(['cupy-cuda102==8.0.0b1', 'cupy-cuda102==8.0.0a1'],
             '/home/ecastill/em-cupy/', ['benchmarks/bench_raw.py']).compare()
