import matplotlib.pyplot as plt

# ref: 
#   https://qiita.com/FukuharaYohei/items/c87f61aee2a24466d5d4
#   https://qiita.com/FukuharaYohei/items/391d4418d8afe1ae2767

DICT_SURVIVED = {0: '0: Dead', 1: '1: Survived'}
DICT_PCLASS = {1: '1: 1st(Upper)', 2: '2: 2nd(Middle)', 3: '3: 3rd(Lower)'}
DICT_EMBARK = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}

def arrange_stack_bar(ax):
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=30, horizontalalignment="center")
    ax.grid(axis='y', linestyle='dotted')

def output_bars(df, column, index={}):    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)    

    # Key-Valueラベルなしの場合
    if len(index) == 0:
        df_vc = df.groupby([column])["Survived"].value_counts(
            sort=False).unstack().rename(columns=DICT_SURVIVED)
        df[column].value_counts().plot.pie(ax=axes[0, 0], autopct="%1.1f%%")
        df.groupby([column])["Survived"].value_counts(
            sort=False, normalize=True).unstack().rename(columns=DICT_SURVIVED).plot.bar(ax=axes[1, 1], stacked=True)

    # Key-Valueラベルありの場合
    else:
        df_vc = df.groupby([column])["Survived"].value_counts(
            sort=False).unstack().rename(index=index, columns=DICT_SURVIVED)
        df[column].value_counts().rename(index).plot.pie(ax=axes[0, 0], autopct="%1.1f%%")
        df.groupby([column])["Survived"].value_counts(
            sort=False, normalize=True).unstack().rename(index=index, columns=DICT_SURVIVED).plot.bar(ax=axes[1, 1], stacked=True)   

    df_vc.plot.bar(ax=axes[1, 0])

    for rect in axes[1, 0].patches:
        height = rect.get_height()

        # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
        axes[1, 0].annotate('{:.0f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    df_vc.plot.bar(ax=axes[0, 1], stacked=True)

    arrange_stack_bar(axes[0, 1])
    arrange_stack_bar(axes[1, 0])
    arrange_stack_bar(axes[1, 1])

    # データラベル追加
    [axes[0, 1].text(i, item.sum(), item.sum(), horizontalalignment='center') 
     for i, (_, item) in enumerate(df_vc.iterrows())]

    plt.show()
