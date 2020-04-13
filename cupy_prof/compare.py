import os
import subprocess
import pickle

import cupy_prof


class _GitCommandError(Exception):
    pass


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


script = """virtualenv venv-{commit};
source venv-{commit}/bin/activate;
pip install pandas
pip install seaborn
pip install Cython
pip install -e {repository} -vvv;
python -c 'import cupy; print(cupy.__version__)';
# Run the benchmarks
python prof.py --dump-pickle {commit}.pkl -- {benchmarks}
"""


def _create_environment(repository, commit, benchmarks):
    # Creates a virtual env that is run with the above command
    # command = ['virtualenv', '/tmp/commit']
    # subprocess.check_output(command)
    with open('test.sh', 'w') as f:
        f.write(script.format(commit=commit, repository=repository,
                              benchmarks=' '.join(benchmarks)))
    proc = subprocess.Popen(['bash', 'test.sh'], stdout=subprocess.PIPE)
    rc = None
    while rc is None:
        output = proc.stdout.readline()
        if output:
            print(output.strip().decode('utf8'))
        rc = proc.poll()


class Comparer(object):
    """ Compares different commits of a GH repo for the same benchmarks

    Args:
        commits (List str) commit hashes or branches to compare
        repository (str) path to the github repository to compare
        run_args (list str) list with the arguments that prof.py was called
            it is used to call prof.py recursively to evaluate the actual
            benchmarks
    """
    def __init__(self, commits, repos, benchmarks):
        self.commits = commits
        if len(repos) == 1:
            repos = repos * len(commits)
        self.repos = repos
        self.benchmarks = benchmarks

    def compare(self, csv=True, plot=True, force_clean=True):
        # We use spawn to reload the newly installed environment
        dfs = []
        for commit, repo in zip(self.commits, self.repos):
            if force_clean:
                _git(repo, ['clean', '-f', '-x'])
            _git(repo, ['checkout', commit])
            # Install the module calling pip
            # TODO(ecastill) abstract this by adding config files
            # with build commands
            # If the module is installed in editable mode
            # It won't be loadable from this interpreter instance
            #
            _create_environment(repo, commit, self.benchmarks)
            with open('{}.pkl'.format(commit), 'rb') as picklefile:
                dfs.append(pickle.load(picklefile))

        dfs = self._compare_results(dfs)
        # Now we generate the plots or csvs if needed
        # We will need to retrieve the actual benchmarks
        collector = cupy_prof.Collector()
        collector.collect(self.benchmarks)
        bench_classes = {}
        for bench_class in collector.benchmarks:
            bench_classes[bench_class.__name__] = bench_class()
        for bench in dfs:
            if csv:
                dfs[bench].to_csv('{}.csv'.format(bench))
            if plot:
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
        return res
