import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def pca(X, D):
    """
    PCA
    Input:
        X - An (1024, 1024) numpy array
        D - An int less than 1024. The target dimension
    Output:
        X - A compressed (1024, 1024) numpy array
    """
    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    u, e, v = np.linalg.svd(np.dot(X.T, X))
    eigen = v.T[:, 0:D]
    final = np.dot(X, eigen)
    variances = (e / np.sum(e))[:D]
    return final, variances


def sklearn_pca(X, D):
    """
    Your PCA implementation should be equivalent to this function.
    Do not use this function in your implementation!
    """
    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    from sklearn.decomposition import PCA
    p = PCA(n_components=D, svd_solver='full')
    trans_pca = p.fit_transform(X)
    return trans_pca


if __name__ == '__main__':
    df = pd.read_csv("Data/BAL/normdataBAL0715.txt", sep = '\t', usecols=range(2,154))
    # X = pd.read_csv("Data/BAL/normdataBAL0715.txt", sep = '\t', nrows= 1000, usecols = range(2, 155))
    anova_results = pd.DataFrame(index=df.index, columns=['F-value', 'p-value'])

    #anova testing across sub-types
    for gene in df.index:
        group1 = df.loc[gene, df.columns.str.contains('NC')]
        group2 = df.loc[gene, df.columns.str.contains('SA')]
        group3 = df.loc[gene, df.columns.str.contains('notSA')]
        group4 = df.loc[gene, df.columns.str.contains('VSA')]
        f_value, p_value = stats.f_oneway(group1, group2, group3, group4)
        anova_results.loc[gene, 'F-value'] = f_value
        anova_results.loc[gene, 'p-value'] = p_value         

    #BH correction for alpha
    alpha = 0.05
    bh_corrected_pvals = multipletests(anova_results['p-value'], method='fdr_bh')[1]
    filtered_genes = anova_results.index[bh_corrected_pvals < alpha] 

    filtered_df = df.loc[filtered_genes]
    filtered_df.to_csv("filtered_data_BAL.csv")

    D = 2
    pca_final, variances=pca(filtered_df.T, D)
    pca_sklearn=sklearn_pca(filtered_df, D)

    total_variance = np.sum(variances)
    pct_first_two_pcs = np.sum(variances[:2]) / total_variance * 100

    print(variances)

    plt.scatter(pca_final[:, 0], pca_final[:, 1])
    plt.title('PCA')
    plt.show()




