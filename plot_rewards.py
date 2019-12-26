import matplotlib.pyplot as plt
import pandas as pd


def save_reward_plot_to_file():
    df = pd.read_json("data/videos/openaigym.episode_batch.0.84915.stats.json")

    ax = plt.gca()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")

    df.plot(kind="line", y="episode_rewards", use_index=True, ax=ax, legend=False, title="Reward vs. Episode")

    fig = ax.get_figure()
    fig.savefig("images/rewards.png")


if __name__ == '__main__':
    save_reward_plot_to_file()
