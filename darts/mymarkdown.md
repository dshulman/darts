        """
        Use a regression model for ensembling individual models' predictions using the stacking technique [1]_.

        The provided regression model must implement ``fit()`` and ``predict()`` methods
        (e.g. scikit-learn regression models). Note that here the regression model is used to learn how to
        best ensemble the individual forecasting models' forecasts. It is not the same usage of regression
        as in :class:`RegressionModel`, where the regression model is used to produce forecasts based on the
        lagged series.

        If `future_covariates` or `past_covariates` are provided at training or inference time,
        they will be passed only to the forecasting models supporting them.

        If `forecasting_models` contains exclusively GlobalForecastingModels, they can be pre-trained. Otherwise,
        the `forecasting_models` must be untrained.

        The regression model does not leverage the covariates passed to ``fit()`` and ``predict()``.

        Parameters
        ----------
        forecasting_models
            List of forecasting models whose predictions to ensemble
        regression_train_n_points
            The number of points per series to use to train the regression model. Can be set to `-1` to use the
            entire series to train the regressor if `forecasting_models` are already fitted and
            `train_forecasting_models=False`.
        regression_model
            Any regression model with ``predict()`` and ``fit()`` methods (e.g. from scikit-learn)
            Default: ``darts.models.LinearRegressionModel(fit_intercept=False)``

            .. note::
                if `regression_model` is probabilistic, the `RegressionEnsembleModel` will also be probabilistic.
            ..
        regression_train_num_samples
            Number of prediction samples from each forecasting model to train the regression model (samples are
            averaged). Should be set to 1 for deterministic models. Default: 1.

            .. note::
                if `forecasting_models` contains a mix of probabilistic and deterministic models,
                `regression_train_num_samples will be passed only to the probabilistic ones.
            ..
        regression_train_samples_reduction
            If `forecasting_models` are probabilistic and `regression_train_num_samples` > 1, method used to
            reduce the samples before passing them to the regression model. Possible values: "mean", "median"
            or float value corresponding to the desired quantile. Default: "median"
        train_forecasting_models
            If set to `False`, the `forecasting_models` are not retrained when calling `fit()` (only supported
            if all the `forecasting_models` are pretrained `GlobalForecastingModels`). Default: ``True``.
        train_using_historical_forecasts
            If set to `True`, use `historical_forecasts()` to generate the forecasting models' predictions used to
            train the regression model in `fit()`. Available when `forecasting_models` contains only
            `GlobalForecastingModels`. Recommended when `regression_train_n_points` is greater than
            `output_chunk_length` of the underlying `forecasting_models`.
            Default: ``False``.
        show_warnings
            Whether to show warnings related to forecasting_models covariates support.
        References
        ----------
        .. [1] D. H. Wolpert, “Stacked generalization”, Neural Networks, vol. 5, no. 2, pp. 241–259, Jan. 1992

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import RegressionEnsembleModel, NaiveSeasonal, LinearRegressionModel
        >>> series = AirPassengersDataset().load()
        >>> model = RegressionEnsembleModel(
        >>>     forecasting_models = [
        >>>         NaiveSeasonal(K=12),
        >>>         LinearRegressionModel(lags=4)
        >>>     ],
        >>>     regression_train_n_points=20
        >>> )
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[494.24050364],
               [464.3869697 ],
               [496.53180506],
               [544.82269341],
               [557.35256055],
               [630.24334385]])
        """
