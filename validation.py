
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def liftChart(data,actual, pred):
    a = data.ix[:,[actual, pred]].sort_values(pred)
    a.ix[:,'pred_g'] = pd.cut(np.arange(len(a)), 10, labels = np.arange(1,11,1))
    ax = a.groupby('pred_g').mean()[actual].plot(title='Lift Chart')
    ax.set_xlabel('predicted group')
    ax.set_ylabel('actual '+ actual+' rat')
    print("The lift is {}".format(a.groupby('pred_g').mean()[actual][-1]/a.groupby('pred_g').mean()[actual].mean()))

def gbmValidation(params, X_train, X_test, y_train, y_test, accuracy_report=True):
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
