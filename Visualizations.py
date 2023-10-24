import matplotlib.pyplot as plt
import plotly.express as px

# Check correlation between variables
correlation_df = merged_df.corr()
correlation_df

# function to plot a basic bar plot

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


barplot(
    df = merged_df, 
    x_column = "Klokkeslett", 
    y_column = "Trafikkmengde", 
    x_label = "Klokkeslett", 
    y_label = "Gjennomsnittelig trafikkmengde", 
    title = 'Gjennomsnittelig trafikkmengde per time av døgnet'
    )