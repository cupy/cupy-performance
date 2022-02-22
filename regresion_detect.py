import sys

import pandas


def sanitize_df(df):
    df = df.iloc[:, 1:].drop("run", axis=1)
    g_df = df.groupby(["name", "key", "dev"])
    mean = g_df.mean().reset_index()
    return mean


def join_dfs(df_a, df_b):
    df = pandas.merge(df_a, df_b, on=["name", "key", "dev"])
    return df


def calc_relative_perf(df):
    df["perf"] = df["time_y"] / df["time_x"]
    return df


def main():
    df_1 = pandas.read_csv(sys.argv[1])
    df_2 = pandas.read_csv(sys.argv[2])
    mean_1 = sanitize_df(df_1)
    mean_2 = sanitize_df(df_2)
    comp = join_dfs(mean_1, mean_2)
    perf = calc_relative_perf(comp)
    print(perf.to_string())
    # Find columns where performance has degraded at least 5%
    degrade = perf[perf["perf"] >= 1.05]
    if degrade.shape[0] == 0:
        print("No perf degradation")
    else:
        print("Performance degradation detected for")
        print(degrade.to_string())
        raise RuntimeError("Performance regresion")


if __name__ == '__main__':
    main()
