import pandas as pd
import numpy as np


class GroupEstimate:
    """
    A simple estimator that predicts values based on group statistics (mean or median).
    
    Parameters
    ----------
    estimate : str
        The type of estimate to use. Either "mean" or "median".
    """
    
    def __init__(self, estimate='mean'):
        """
        Initialize the GroupEstimate estimator.
        
        Parameters
        ----------
        estimate : str, default='mean'
            The type of estimate to use. Either "mean" or "median".
        """
        if estimate not in ['mean', 'median']:
            raise ValueError("estimate must be either 'mean' or 'median'")
        
        self.estimate = estimate
        self.group_estimates_ = None
        self.column_names_ = None
        self.default_category_ = None
        self.default_estimates_ = None
    
    def fit(self, X, y, default_category=None):
        """
        Fit the estimator by calculating group statistics.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Categorical features to group by.
        y : array-like
            Target values to estimate.
        default_category : str, optional
            Column name to use as fallback when combination is missing.
            
        Returns
        -------
        self : GroupEstimate
            Returns self for method chaining.
        """
        # Convert X to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Convert y to Series if needed
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Store column names for later use
        self.column_names_ = X.columns.tolist()
        self.default_category_ = default_category
        
        # Combine X and y into a single DataFrame
        df = X.copy()
        df['_target_'] = y.values
        
        # Group by all columns in X and calculate the estimate
        if self.estimate == 'mean':
            self.group_estimates_ = df.groupby(self.column_names_)['_target_'].mean()
        else:  # median
            self.group_estimates_ = df.groupby(self.column_names_)['_target_'].median()
        
        # If default_category is specified, calculate fallback estimates
        if default_category is not None:
            if default_category not in self.column_names_:
                raise ValueError(f"default_category '{default_category}' not found in X columns")
            
            if self.estimate == 'mean':
                self.default_estimates_ = df.groupby(default_category)['_target_'].mean()
            else:  # median
                self.default_estimates_ = df.groupby(default_category)['_target_'].median()
        
        return self
    
    def predict(self, X_):
        """
        Predict estimates for new observations.
        
        Parameters
        ----------
        X_ : array-like or pandas.DataFrame
            New observations to predict.
            
        Returns
        -------
        predictions : numpy.ndarray
            Predicted values for each observation.
        """
        if self.group_estimates_ is None:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")
        
        # Convert X_ to DataFrame if needed
        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_, columns=self.column_names_)
        
        # Ensure columns match
        if X_.columns.tolist() != self.column_names_:
            X_.columns = self.column_names_
        
        predictions = []
        missing_count = 0
        
        for idx, row in X_.iterrows():
            # Create a tuple key for the group
            key = tuple(row[col] for col in self.column_names_)
            
            # Try to get the estimate for this group
            try:
                if len(self.column_names_) == 1:
                    # For single column, key is not a tuple in the index
                    prediction = self.group_estimates_.loc[key[0]]
                else:
                    prediction = self.group_estimates_.loc[key]
                predictions.append(prediction)
            except KeyError:
                # Group not found - use default category if available
                if self.default_category_ is not None and self.default_estimates_ is not None:
                    default_key = row[self.default_category_]
                    try:
                        prediction = self.default_estimates_.loc[default_key]
                        predictions.append(prediction)
                    except KeyError:
                        # Even default category not found
                        predictions.append(np.nan)
                        missing_count += 1
                else:
                    # No default category - return NaN
                    predictions.append(np.nan)
                    missing_count += 1
        
        # Print message if there are missing groups
        if missing_count > 0:
            print(f"Warning: {missing_count} observation(s) had missing groups and returned NaN")
        
        return np.array(predictions)