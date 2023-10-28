import matplotlib.pyplot as plt
import seaborn as sns


def barplot(df, x_column, y_column, x_label=None, y_label=None, title=None, x_labels=None):
    """Groups the dataand lots a basic plot using matplotlib"""
    grouped_df = df.groupby(x_column)[y_column].mean()
    plt.bar(grouped_df.index, grouped_df.values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    if x_labels is None:
        plt.xticks(grouped_df.index)
    else:
        ticks = range(len(grouped_df.index)) if grouped_df.index[0] == 0 else range((grouped_df.index)[0], len(grouped_df.index)+1)
        plt.xticks(ticks, x_labels)
    
    plt.show()

def correlations(df):
    correlations = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix Heatmap")
    plt.show()