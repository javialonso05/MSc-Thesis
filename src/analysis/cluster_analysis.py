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
                               variables: list[str] = None,
                               show: bool = False, save: bool = False) -> dict:
        """
        Test whether the difference in the median of variables is statistically significant
        
        Args:
            data (pd.DataFrame): Physical properties of the clusters
            variables (list[str], optional): Variables to test for normality. If None, all variables will be tested
            show(bool, optional): Flag for plotting the distributions of the data.
            save(bool, optional): Flag for saving the plot.

        Returns:
            tuple[dict, dict]: Results of the ANOVA and posthoc tests
        """
        
        if variables is None:
            variables = [col for col in data.columns if col != 'Label']
        
        normality = self.test_normality(data=data, variables=variables, plot_results=False)
        
        # Initialize variables
        results = {}

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
            
            # Plot results
            if show:
                # Compute mean per label and sort
                means = data.groupby('Label')[var].median().sort_values()
                sorted_labels = means.index[::-1]

                # Reorder 'Label' as a categorical with sorted order
                subset = data.copy()
                subset['Label'] = pd.Categorical(subset['Label'], categories=sorted_labels, ordered=True)

                # Plot boxplot
                subset.boxplot(column=var, by='Label', grid=False, figsize=(12, 5))
                
                plt.title(f'{var} - {test}: {p_value:.3f}')
                plt.suptitle('')
                plt.xlabel('Label')
                plt.ylabel(var)
                plt.tight_layout()
                if save:
                    if var == 'Lclump/Mclump':
                        plt.savefig(f'results/Plots/Lclump_Mclump_boxplot.pdf')
                    else:
                        plt.savefig(f'results/Plots/{var}_boxplot.pdf')
                plt.show()
                
        print('-' * 25)
        
        return results

    def posthoc_test(self,
                     data: pd.DataFrame,
                     results: dict) -> dict:
        """_summary_

        Args:
            data (pd.Dataframe): physical properties of the cores and labeling assignment
            results (dict[variable: (test, score, p_value)]): output from self.test_median_difference

        Returns:
            dict: results of the posthoc test
        """
        from scipy.stats import ttest_ind
        from scipy.stats import mannwhitneyu
        
        # Retrieve labels of interest
        unique_labels, sizes = np.unique(data['Label'], return_counts=True)
        labels = [unique_labels[i] for i in range(len(unique_labels)) if sizes[i] >= self.min_n]
        
        # Perform post-hoc tests
        posthoc_results = {}
        for var in results.keys():
            n = 1
            test, _, p_value = results[var]
            independence = {}
            
            if p_value > 0.05:
                # All distributions come from the same underlying distribution
                independence[0] = []
                for label in labels:
                    df_0 = data[data['Label'] == label]
                    df_1 = data[data['Label'] != label]
                    
                    if test == 'anova':  # Distribution is normal
                        _, p_value = ttest_ind(df_0[var], df_1[var], nan_policy='propagate')
                    else:
                        _, p_value = mannwhitneyu(df_0[var], df_1[var], alternative='two-sided')
                    
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
        
        return posthoc_results

    def visualize_results(self,
                          data: pd.DataFrame,
                          posthoc_results: dict,
                          spectra: np.ndarray,
                          frequency: np.ndarray,
                          variables: list[str] = None) -> None:
        """
        Visualize the distributions identified by the posthoc test for the given variables and their corresponding spectra

        Args:
            data (pd.DataFrame): physical properties of the cores and their label assignment.
            posthoc_results (dict): output from self.posthoc_test.
            spectra (np.ndarray): intensity spectra for each of the signals.
            frequency (np.ndarray): frequency channels corresponding to spectra.
            variables (list[str], optional): list of variables to plot. If None, all variables will be plotted.
        """
        import seaborn as sns
        
        # Define variables
        if variables is None:
            variables = posthoc_results.keys()
        
        # Iterate through variables
        for var in variables:
            # Retrieve groups
            groups = posthoc_results[var].keys()
            
            # Initialize density figure parameters
            fig0, ax0 = plt.subplots(figsize=(8, 5))  # Figure for the density distributions
            cmap = plt.get_cmap('tab10')
            
            # Initialize second figure
            fig, ax = plt.subplots(len(groups), 1, sharex=True, figsize=(10, 4 * len(groups)))
                
            for i, group in enumerate(groups):
                # Retrieve clusters belonging to that group
                clusters = posthoc_results[var][group]
                
                # Define color and linestyle
                if i < 10:
                    clr = cmap.colors[i]
                    style = '-'
                elif i < 20:
                    clr = cmap.colors[i - 10]
                    style = '--'
                else:
                    clr = cmap.colors[i - 20]
                    style = ':'
                
                # Plot density distribution
                sns.kdeplot(data[data['Label'].isin(clusters)], x=var, color=clr, ax=ax0, linestyle=style)
                
                mean_spectrum = spectra[data['Label'].isin(clusters)].mean(axis=0)
                mean_spectrum /= np.max(mean_spectrum)
                
                ax[i].plot(frequency, mean_spectrum, color=clr, linestyle=style, label=i)
                
            # Define legend and show figures
            fig0.legend(list(groups), title='Group:')
            
            fig.legend(title='Group:')
            fig.supxlabel('Frequency [MHz]')
            fig.supylabel('Normalized Intensity [-]')
            
            plt.show()
