import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Analyzer:
    def __init__(self, minimum_samples: int = 20):
        self.min_n = minimum_samples
    
    def _kruskal_wallis_test(self,
                             data: pd.DataFrame,
                             variables: list[str] = None) -> dict:
        """
        Apply Kruskal Wallis H-test (non-parametric ANOVA) to all columns in the dataframe other than label

        Args:
            data (pd.DataFrame): The physical properties to test and the clustering assignment
            min_group_size (int): Minimum number of samples required in a group to include it in the test.


        Returns:
            dict: {column: (statistic, p-value)} for each tested column
        """
    
        from scipy.stats import kruskal
    
        results = {}
        grouped = data.groupby('Label')
        
        if variables is None:
            variables = data.columns
        
        # Filter groups by size
        valid_groups = {name: group for name, group in grouped if len(group) >= self.min_n}
        for col in variables:
            # Exclude label
            if col == 'Label':
                continue
            
            # Gather values for each group
            groups = [group[col].dropna().values for group in valid_groups.values()]
            
            # Store result
            if all(len(g) > 0 for g in groups):
                stat, p = kruskal(*groups)
                results[col] = (stat, p)
            else:
                results[col] = (np.nan, np.nan)
                
        return results

    def _post_hoc_comparison(self,
                             data: pd.DataFrame,
                             variables: list[str] = None) -> dict[int, pd.DataFrame]:
        """
        Perform a pairwise analysis on the non-normal vairables

        Args:
            data (pd.DataFrame): Physical properties of the cores
            variables (list[str], optional): List of variables to test. If None, all variables will be tested.
        """
        
        if variables is None:
            variables = data.columns
        
        # Filter groups by size        
        labels, sizes = np.unique(data['Label'], return_counts=True)
        valid_groups = [labels[i] for i in range(len(labels)) if sizes[i] >= self.min_n]
        df = data[data['Label'].isin(valid_groups)]
        
        # Perform test
        import scikit_posthocs as sp
        
        results = {}
        for var in variables:
            results[var] = sp.posthoc_dunn(df, val_col=var, group_col='Label', p_adjust='bonferroni')
            
            plt.imshow(results[var], cmap='bwr')
            plt.colorbar(label='Test')
            plt.show()
            
        return results

    def _anova_test(self, 
                    data: pd.DataFrame, 
                    variables: list[str] = None) -> dict:
        """
        Perform one-way ANOVA (F-test) for normal distributions on the specified columns, grouped by 'Label'.

        Args:
            data (pd.DataFrame): DataFrame containing the data and 'Label' column.
            variables (list[str], optional): List of columns to test. If None, all columns except 'Label' are tested.

        Returns:
            dict: {column: (statistic, p-value)} for each tested column
        """
        from scipy.stats import f_oneway
        
        results = {}
        grouped = data.groupby('Label')
        if variables is None:
            variables = [col for col in data.columns if col != 'Label']

        # Filter groups by size
        valid_groups = {name: group for name, group in grouped if len(group) >= self.min_n}

        for col in variables:
            # Gather values for each group
            groups = [group[col].dropna().values for group in valid_groups.values()]
            # Only test if all groups have data
            if all(len(g) > 0 for g in groups) and len(groups) > 1:
                stat, p = f_oneway(*groups)
                results[col] = (stat, p)
            else:
                results[col] = (np.nan, np.nan)
        return results

    def _t_test(self,
                data: pd.DataFrame,
                variables: list[str] = None) -> dict[str, np.ndarray]:
        """
        Perform pairwise t-tests across the variables

        Args:
            data (pd.DataFrame): Physical properties of the variables
            variables (list[str], optional): Variables to test. If None, all variables will be tested.

        Returns:
            dict[str, np.ndarray]: dictinary with arrays representing the p-values of the t-tests.
        """
        from scipy.stats import ttest_ind

        if variables is None:
            variables = [col for col in data.columns if col != 'Label']
        
        labels, sizes = np.unique(data['Label'], return_counts=True)
        valid_labels = [labels[i] for i in range(len(labels)) if sizes[i] >= self.min_n]
        
        n = len(valid_labels) - 1
        results = {}
        for var in variables:
            values = np.ones((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    stat, p_value = ttest_ind(data[var][data['Label'] == valid_labels[i]], data[var][data['Label'] == valid_labels[j]], nan_policy='propagate')
                    values[i, j] = p_value
                    values[j, i] = p_value
            
            results[var] = values
        return results

    def test_normality(self,
                    data: pd.DataFrame,
                    variables: list[str] = None,
                    plot_results: bool = False) -> dict[str: dict[int: float]]:
        """
        Test the normality of the variables in the data, grouped by label

        Args:
            data (pd.DataFrame): Physical properties of the clusters
            variables (list[str], optional): Variables to test for normality. If None, all variables will be tested
            plot_results (bool, optional): Flag for plotting the results of the normality test as boxplots

        Returns:
            dict[str: dict[int: float]]: For each variable, test whether each cluster's distribution is normal
        """
        from scipy.stats import shapiro
        
        if variables is None:
            variables = data.columns
        
        results = {}
        unique_labels, size = np.unique(data['Label'], return_counts=True)
        
        # Iterate through variables
        valid_labels = []
        for var in variables:
            # Exclude label
            if var == 'Label':
                continue
            
            normality = {}
            # Iterate through clusters
            for i, label in enumerate(unique_labels):
                if size[i] < self.min_n:
                    continue
                
                valid_labels.append(label)
                
                # Filter based on cluster label
                df = data[data['Label'] == label]
                
                # Perform test
                score, p_value = shapiro(df[var])
                
                # Append result
                normality[label] = p_value
            
            # Store result
            results[var] = normality
        
        # Plot results
        if plot_results:
            fig, axes = plt.subplots(len(variables) - 1, 1, figsize=(8, 5 * (len(variables) - 1)))
            
            import seaborn as sns
            for i, var in enumerate(variables):
                if i == len(variables) - 1:
                    continue
                
                # Select axis
                ax = axes[i]
                
                # Determine box color
                color_map = ['g' if results[var][j] < 0.05 else 'r' for j in results[var].keys()]
                
                # Draw boxplot
                box = sns.boxplot(data[data['Label'].isin(valid_labels)], x='Label', y=var, ax=ax, palette=color_map)
                
                # Set title
                ax.set_title(var)
            
            plt.tight_layout()
            plt.show()
        
        return results
                
    def test_median_difference(self,
                               data: pd.DataFrame,
                               variables: list[str] = None) -> tuple[dict[str: tuple[float, float, float]], dict[str: list[float]]]:
        """
        Test whether the difference in the median of variables is statistically significant
        
        Args:
            data (pd.DataFrame): Physical properties of the clusters
            variables (list[str]): Variables to test for normality. If None, all variables will be tested

        Returns:
            tuple[dict, dict]: Results of the ANOVA and posthoc tests
        """
        from scipy.stats import ttest_ind
        from scipy.stats import mannwhitneyu

        
        if variables is None:
            variables = [col for col in data.columns if col != 'Label']
        
        normality = self.test_normality(data=data, variables=variables, plot_results=False)
        
        unique_labels, sizes = np.unique(data['Label'], return_counts=True)
        labels = [unique_labels[i] for i in range(len(unique_labels)) if sizes[i] >= self.min_n]
        
        # Initialize variables
        results = {}
        posthoc_results = {}

        # Compare the distributions
        print('-' * 25)
        print('Test results:')
        print('-' * 25)
        for var in variables:
            # Perform ANOVA tests
            vals = np.array(list(normality[var].values()))
            if np.all(vals < 0.05):  
                # All distributions of the variable are normal
                test_result = self._anova_test(data=data, variables=[var])
                test = 'anova'
            else:
                # Not all distributions are normal
                test_result = self._kruskal_wallis_test(data=data, variables=[var])
                test = 'kruskal'
            
            score, p_value = test_result[var]
            results[var] = (test, score, p_value)
            
            # Report results
            print(f'{var}:\t{round(p_value, 5)}')
            
            # Perform post-hoc tests
            n = 1
            independence = {}
            if p_value > 0.05:
                # All distributions come from the same underlying distribution
                independence[0] = []
                for label in labels:
                    df_0 = data[data['Label'] == label]
                    df_1 = data[data['Label'] != label]
                    
                    if test == 'anova':  # Distribution is normal
                        stat, p_value = ttest_ind(df_0[var], df_1[var], nan_policy='propagate')
                    else:
                        stat, p_value = mannwhitneyu(df_0[var], df_1[var], alternative='two-sided')
                    
                    if p_value <= 0.05:
                        independence[n] = [label]
                        n += 1
                    else:
                        independence[0].append(label)
                    
                    # TODO: test if the independent groups are equal
            else:
                # Distributions do not come from the same distribution
                for i, label in enumerate(labels):
                    df_0 = data[data['Label'] == label]
                    if i == 0:  # First cluster and group
                        independence[0] = [label]
                    else:
                        # Check whether the signal belongs to any existing groups
                        new_group = True
                        for j in range(n):
                            df_1 = data[data['Label'].isin(independence[j])]
                            if test == 'anova':
                                stat, p_value = ttest_ind(df_0[var], df_1[var])
                            else:
                                stat, p_value = mannwhitneyu(df_0[var], df_1[var], alternative='two-sided')
                            
                            if p_value <= 0.05:
                                # The signal is compatible with this group
                                independence[j].append(label)
                                new_group = False
                                break
                        
                        if new_group:
                            independence[n] = [label]
                            n += 1
                    
            posthoc_results[var] = independence
                
        print('-' * 25)
        
        return results, posthoc_results



def kruskal_wallis_test(data: pd.DataFrame,
                        min_group_size: int = 20) -> dict:
    """
    Apply Kruskal Wallis H-test (non-parametric ANOVA) to all columns in the dataframe other than label

    Args:
        data (pd.DataFrame): The physical properties to test and the clustering assignment
        min_group_size (int): Minimum number of samples required in a group to include it in the test.


    Returns:
        dict: {column: (statistic, p-value)} for each tested column
    """
    from scipy.stats import kruskal
    
    results = {}
    grouped = data.groupby('Label')
    # Filter groups by size
    valid_groups = {name: group for name, group in grouped if len(group) >= min_group_size}
    
    for col in data.columns:
        # Exclude 'Label'
        if col == 'Label':
            continue
        
        # Gather values for each group
        groups = [group[col].dropna().values for group in valid_groups.values()]
        if all(len(g) > 0 for g in groups):
            stat, p = kruskal(*groups)
            results[col] = (stat, p)
        else:
            results[col] = (np.nan, np.nan)
    return results
    
    


def plot_cluster_distributions(core_data: pd.DataFrame,
                               core_info: pd.DataFrame,
                               variable: str,
                               min_cluster_size: int = 10,
                               cluster_labels: list = None,
                               ):
    """
    Plot boxplots for a given property and the relevant clusters
    Args:
        core_data (pd.DataFrame): physical properties of the cores
        core_info (pd.DataFrame): label assignment and core info (source + coreID)
        variable (str): variable from core_data to test
        min_cluster_size (int, optional): Minimum cluster size to consider. Defaults to 10.
        cluster_labels (list, optional): Clusters to consider. Defaults to None.

    Raises:
        ValueError: {variable} not in core_data
        ValueError: Either min_cluster_size or cluster_labels must be specified.
    """
    
    labels = core_info['Labels'].values
    
    # Check inputs    
    if cluster_labels is None:  # Clusters to plot
        cluster_labels = []
        if min_cluster_size is not None:
            unique_labels, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                if count >= min_cluster_size:
                    cluster_labels.append(label)
        else:
            raise ValueError("Either min_cluster_size or cluster_labels must be specified.")
        
    
    if variable not in core_data.columns:
        raise ValueError(f"'{variable}' not found in the DataFrame.")
    
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")
    
    # Find mapping: core_info -> core_data
    mapping = []
    for i in range(len(core_info)):
        idx = core_info[(core_info['Source'].iloc[i] == core_data['CLUMP']) & (core_info['Core'].iloc[i] == core_data['ID'])].index
        if len(idx) > 0:
            mapping.append([i, idx[0]])
    
    # Find mask
    data_mask = []
    info_mask = []
    for i in range(len(mapping)):
        info_idx, data_idx = mapping[i]
        
        if core_info['Labels'].iloc[info_idx] in cluster_labels:
            data_mask.append(data_idx)
            info_mask.append(info_idx)
    
    # Select data
    core_data = core_data.iloc[data_mask]
    core_info = core_info.iloc[info_mask]
    
    # Plot properties
    df = pd.DataFrame()
    df[variable] = core_data[variable]
    df['Labels'] = core_info['Labels']
    df.boxplot(column=variable, by='Labels')
    plt.ylabel(variable)
    plt.show()


def find_mapping(signal_data: np.ndarray, 
                 core_data: pd.DataFrame,
                 core_mapping: dict) -> list:
    """
    Find the mapping of signal indices to data values.
    :param data: Array with signal data
    :param core_mapping: Mapping from signal indices to source and core tags
    :return: List of mapped cluster labels
    """
    
    mapping = []
    for i in range(len(signal_data)):
        # Extract region and core from the mapping
        region, core = core_mapping[i]
        
        # Locate datapoint in core_data
        core_idx = core_data[(core_data['CLUMP'] == region) & (core_data['ID'] == core[4:])].index
        if len(core_idx) == 0:
            mapping.append(np.nan)
        else:
            mapping.append(core_idx[0])
    
    return mapping



if __name__ == '__main__':
    # Load core data
    data_info = pd.read_csv('Data/Raw/7MTM2TM1_Core_catalogue_out_official_v3_run6Apr23+16Jan24_sn-5_may24_clump_cat.txt', sep='\s+', comment='\\')
    data_info = data_info.drop(data_info.index[0]).reset_index(drop=True)
    data_info.to_csv('Data/data_info.csv', index=False)