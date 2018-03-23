from sklearn import cross_validation, metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

def gbmPredict(params, X_train, X_test, y_train, y_test, accuracy_report=True):
    gbmclf = GradientBoostingClassifier(**params)
    gbmfit = gbmclf.fit(X_train, y_train)

    #create accuracy
    def printReport(X,y):
        predictions = gbmfit.predict(X)
        predprob = gbmfit.predict_proba(X)[:,1]

        #print "\nModel Report"
        print("Accuracy : %.4g" % metrics.accuracy_score(y, predictions))
        print("AUC Score : %f" % metrics.roc_auc_score(y, predprob))
    if accuracy_report:
        print("\nModel Report on Training Data")
        printReport(X_train, y_train)
        print("\nModel Report on Test Data")
        printReport(X_test, y_test)

    # Plot feature importance
    feature_importance = gbmfit.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)

    feature_importance_df = pd.DataFrame({'features': np.array(X_train.columns)[sorted_idx],
                                         'importance': feature_importance[sorted_idx]})


    return({'fit': gbmfit, 'feature_importance': feature_importance_df})

# plot feature_importance
def plotFeatureImportance(feature_importance, top_n=10):
    # Plot feature importance
    #feature_importance = fit.feature_importances_
    # make importances relative to max importance
    #feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance.importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    #plt.subplot(1, 2, 2)
    fig = plt.figure(figsize=(6,10))
    plt.barh(pos[-top_n:], np.array(feature_importance.importance)[sorted_idx][-top_n:], align='center')
    plt.yticks(pos[-top_n:], np.array(feature_importance.features)[sorted_idx][-top_n:])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import gamma
from skopt import BayesSearchCV
import matplotlib.backends.backend_pdf
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from jinja2 import Environment, FileSystemLoader
import capytal.models.revenue as revenue
import capytal.models.revenue.processing as processing
import capytal.models.revenue.validation as validation
# import processing
# import validation

class SegmentQuantileClassifier():
    def __init__(self, quantiles=[0.05, 0.25, 0.5]):
        self.quantiles = quantiles

    def fit(self, X, y, dimensions=[]):
        self.lookup_table = (pd.concat([X[dimensions],y], axis=1)
              .groupby(dimensions)
              .apply(lambda g: g.quantile(self.quantiles)
                     .reset_index().rename(columns={'index':'percentile'})[['percentile',y.name]]
                    )
              .pivot_table(index=dimensions, columns='percentile', values=y.name)
              .reset_index()
             )

    def predict(self, X_pred):
        return(pd.merge(X_pred[dimensions], self.lookup_table, how='left', on=dimensions)[lookup_tbl.columns.difference(dimensions)])

    def score(self, X_pred, y, type='rmse'):
        # type: any one in ['rmse', 'rmse_ic']
        if type=='rmse':
            y_pred = pd.merge(X_pred[dimensions], self.lookup_table[dimensions.append('y')], how='left', on=dimensions)[lookup_tbl.columns.difference(dimensions)]
            rmse = y_pred.apply(lambda c: np.sqrt(sum((c-y)**2)/len(y_pred)), axis=0)
        else:
            print("This type is not supported. Please choose from 'rmse', 'rmse_ic'")


class RpEstimator(BaseEstimator, RegressorMixin):
    # some words
    def __init__(self, estimator, params=None, method=None, features=None):
        # method: 'percentile', 'straight_value'
        self.method = method
        self.groupby = None
        self.features = features
        self.estimator = estimator
        self.params = params
        self.model = processing.defineEstimator(estimator)
        if params is not None:
            self.model.set_params(params)

    def create_percentile_label(self, X, y, groupby=None, weighted=True):
        # groupby: list of variables. e.g. ['dim_user_market']
        y.name = 'y_true'
        if groupby is None:
            y_percentile = processing.percentileScore(y).sort_index()
        else:
            self.groupby = groupby
            df = pd.concat([X[groupby],y], axis=1)
            y_percentile = (df.copy().groupby(groupby)
                             .apply(lambda g: processing.percentileScore(g['y_true'], weighted=weighted))
                             .reset_index(level=groupby, drop=True)
                            )
        return(y_percentile)

    def get_gamma_params(self, X, y, groupby=None):
        if groupby is None:
            dct = {}
            for k, v in zip(['fit_alpha', 'fit_loc', 'fit_beta'], gamma.fit(y)):
                dct[k] = y
            self.gamma_params_table = dct
        else:
            self.groupby = groupby
            y.name = 'y_true'
            df = pd.concat([X,y], axis=1)
            self.gamma_params_table = (df.copy()
                                          .groupby(groupby)
                                          .apply(lambda g: stats.gamma.fit(g['y_true']))
                                          .apply(pd.Series)
                                          .rename(columns={0:'fit_alpha', 1:'fit_loc', 2:'fit_beta'})
                                          #.reset_index()
                                       )
        return(self)

    def fit(self, X, y, **kwargs):
        if self.features is None:
            self.model.fit(X, y)
        else:
            self.model.fit(X[self.features], y)
        self.fit_ = True
        return(self)

    def search_cv(self, X, y, params_space, **kwargs):
        opt = BayesSearchCV(estimator = self.model, search_spaces = params_space, n_iter=30, random_state=1234, cv=3, verbose=0)
        if self.features is None:
            opt.fit(X, y)
        else:
            opt.fit(X[self.features], y)
        self.params = opt.best_params_
        if self.features is None:
            self.model.fit(X, y)
        else:
            self.model.fit(X[self.features], y)
        self.fit_ = True
        return(self)



    def predict(self, X_new, groupby=None, method='percentile'):
        try:
            getattr(self, 'fit_')
        except AttributeError:
            raise RuntimeError("You must train the model before making prediction!")

        if method == 'percentile':
            y_percentile_pred = pd.Series(self.model.predict(X_new[self.features]))
            y_percentile_pred.name = 'y_percentile_pred'
            if isinstance(self.gamma_params_table, dict):
                y_pred = y_percentile_pred.apply(lambda y: gamma.ppf(y, self.gamma_params_table['fit_alpha'], loc = self.gamma_params_table['fit_loc'], scale = 1/self.gamma_params_table['fit_beta']))
                y_pred.name = 'y_pred'
                return(pd.concat([y_percentile_pred, y_pred], axis=1))
            elif isinstance(self.gamma_params_table, pd.core.frame.DataFrame):
                df_pred = pd.concat([X_new[self.groupby], y_percentile_pred], axis = 1).merge(self.gamma_params_table, how='left', on=self.groupby, right_index=True)
                df_pred['y_pred'] = df_pred.apply(lambda r: gamma.ppf(r['y_percentile_pred'], r['fit_alpha'], loc = r['fit_loct'], scale = 1/r['fit_beta']), axis=0)
                return(df_pred[['y_percentile_pred', 'y_pred']])
            else:
                return(y_percentile_pred)
                raise("No distribution look up table avaible to inverse percentile to value")
        elif method == 'straight_value':
            y_pred = [max(y, 0) for y in self.model.predict(X_new)]
            return(y_pred)
        else:
            raise("Predict method can't be empty! Choose 'percentile' or 'straight_value'.")

    def get_params(self):
        return(self.model.get_params())

    def score(self, y, y_pred):
        print(validation.rmsle(y, y_pred))

    def score_p_plot(self, y_percentile, y_percentile_pred):
        pass

    def score_plot(sel, y, y_pred):
        pass



class ModelEngine():
    def __init__(self, output_path, model_name):
        self.output_path = output_path
        self.ds = None
        self.current_ds_path = None
        self.current_path = None
        self.model_name = model_name

    def make_new_folder(self, ds, name=None):
        if name is None:
            if not os.path.isdir(os.path.join(self.output_path, ds)):
                os.makedirs(os.path.join(self.output_path, ds))
        else:
            if os.path.isdir(os.path.join(self.current_ds_path, name)):
                raise ValueError("The directory already exists. Please pick another model name")
            os.makedirs(os.path.join(self.current_ds_path, name))
        self.current_ds_path = os.path.join(self.output_path, ds)
        self.ds = ds
        if name is not None:
            self.current_path = os.path.join(self.current_ds_path, name)



    def create_eda(self, df, features, weight=None):
        if not os.path.isdir(os.path.join(self.current_ds_path, name)):
            os.makedirs(os.path.join(self.current_ds_path, self.model_name))

        pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(self.current_path, "feature_eta.pdf"))
        for col in features: ## will open an empty extra figure :(
            fig = plt.figure(figsize=(15,6))
            if df[col].dtype == 'O' or len(df[col].unique()) <= 10:
                if weight is None:
                    df.groupby(col).count().plot(kind='bar')
                else:
                    df.groupby(col)[weight].sum().plot(kind='bar')
            else:
                if weight is None:
                    plt.hist(df[col].fillna(0))
                else:
                    plt.hist(df[col].fillna(0), weights=df[weight])
            pdf.savefig(fig)
            plt.close()
        pdf.close()

    def generate_data(self, ds, as_of_date, term=180, filters_hosts=[], filters_features=[]):
        if self.current_ds_path is None:
            self.make_new_folder(ds)
            #revenue.render_get_eligible_hosts(as_of_date=as_of_date, term=180, filters=filters_hosts)
            revenue.get_eligible_hosts(as_of_date=as_of_date, term=180, filters=filters_hosts)
            mydata = revenue.get_features(as_of_date=as_of_date, filters=filters_features)
            mydata_file_name = os.path.join(self.current_ds_path, 'rp_model_data_' + as_of_date + '.pkl')
            mydata.to_pickle(mydata_file_name)
        else:
            if os.path.exists(os.path.join(self.current_ds_path, 'rp_model_data_' + as_of_date + '.pkl')):
                mydata = pd.read_pickle(os.path.join(self.current_ds_path, 'rp_model_data_' + as_of_date + '.pkl'))
            else:
                revenue.get_eligible_hosts(as_of_date=as_of_date, term=180, filters=filters_hosts)
                mydata = revenue.get_features(as_of_date=as_of_date, filters=filters_features)
                mydata_file_name = os.path.join(self.current_ds_path, 'rp_model_data_' + as_of_date + '.pkl')
                mydata.to_pickle(mydata_file_name)
        return(mydata)

class QuantileXGBRegressor(BaseEstimator, RegressorMixin):
    # Based on:
    # http://www.bigdatarepublic.nl/regression-prediction-intervals-with-xgboost/

    def __init__(self, quant_alpha, quant_delta, quant_thres, quant_var,
                 n_estimators=100, max_depth=3, reg_alpha=5., reg_lambda=1.0,
                 gamma=0.5, n_jobs=4):
        """
        quant_alpha: target percentile
        quant_delta: 1 over the slope of gradient about alpha (theoretically,
                     should be zero, but in practice smoother values work well).
        quant_thres: threshold after which random values should be provided to
                     allow tree model to do random splits.
        quant_var: The amplitude of the noise to add after threshold away from
                   nominal quantile.
        """
        self.quant_alpha = quant_alpha
        self.quant_delta = quant_delta
        self.quant_thres = quant_thres
        self.quant_var = quant_var
        # xgboost parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.n_jobs = n_jobs
        # keep xgboost estimator in memory
        self.clf = None

    def fit(self, X, y):

        def quantile_loss(y_true, y_pred, _alpha, _delta, _threshold, _var):
            x = y_true - y_pred
            grad = (x < (_alpha - 1.0) * _delta) * (1.0 - _alpha) - ((x >= (_alpha - 1.0) * _delta) &
                                                                     (x < _alpha * _delta)) * x / _delta - _alpha * (x > _alpha * _delta)
            hess = ((x >= (_alpha - 1.0) * _delta) & (x < _alpha * _delta)) / _delta
            _len = np.array([y_true]).size
            var = (2 * np.random.randint(2, size=_len) - 1.0) * _var
            grad = (np.abs(x) < _threshold) * grad - (np.abs(x) >= _threshold) * var
            hess = (np.abs(x) < _threshold) * hess + (np.abs(x) >= _threshold)
            return grad, hess

        self.clf = XGBRegressor(
            objective=partial(
                quantile_loss,
                _alpha=self.quant_alpha,
                _delta=self.quant_delta,
                _threshold=self.quant_thres,
                _var=self.quant_var
            ),
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            n_jobs=self.n_jobs
        )
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, X, y):
        y_pred = self.clf.predict(X)
        score = (
            (self.quant_alpha - 1.0) * (y - y_pred) * (y < y_pred) +
            self.quant_alpha * (y - y_pred) * (y >= y_pred)
        )
        score = 1. / np.sum(score)
        return score
