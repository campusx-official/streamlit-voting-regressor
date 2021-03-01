import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error

def train_voting_regressor(algos):

    vr = VotingRegressor(algos)
    vr.fit(X_train,y_train)

    y_pred = vr.predict(X_test1)

    r2 = r2_score(y_test1,y_pred)
    mae = mean_absolute_error(y_test1,y_pred)

    return vr,r2,mae

plt.style.use('seaborn-bright')

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Random state - 8
X_train,X_test1,y_train,y_test1 = train_test_split(X,y,test_size=0.1,random_state=8)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]


st.sidebar.markdown("# Voting Regressor")

# Estimator Multi-select
estimators = st.sidebar.multiselect(
    'Estimators',
    [
        'Linear Regression',
        'SVR',
        'Decision Tree Regressor'
    ]
)

# Build estimators
algos = []

if 'Linear Regression' in estimators:
    lr_reg = LinearRegression()
    algos.append(('lr', lr_reg))
if 'SVR' in estimators:
    svr_reg = SVR()
    algos.append(('svr', svr_reg))
if 'Decision Tree Regressor' in estimators:
    dt_reg = DecisionTreeRegressor(max_depth=5)
    algos.append(('dt', dt_reg))

fig, ax = plt.subplots()
ax.scatter(X, y, s=100,color="yellow", edgecolor="black")
orig = st.pyplot(fig)


if st.sidebar.button("Run Algorithm"):
    vr,r2,mae = train_voting_regressor(algos)
    y_2 = vr.predict(X_test)
    ax.plot(X_test, y_2, linewidth=3,label="Voting Regressor")
    ax.legend()
    orig.pyplot(fig)
    figs = []
    r2_scores = []
    maes = []
    for i in algos:
        i[1].fit(X_train,y_train)
        y_pred = i[1].predict(X_test)
        y_pred1 = i[1].predict(X_test1)
        r2_scores.append(r2_score(y_test1,y_pred1))
        maes.append(mean_absolute_error(y_test1,y_pred1))
        ax.plot(X_test, y_pred, linewidth=1,label=i[0],linestyle='dashdot')
        ax.legend()

    counter = 0
    for i in st.beta_columns(len(algos)):
        with i:
            orig.pyplot(fig)
        counter += 1

    st.sidebar.subheader("Regression Metrics")
    st.sidebar.text("R2 score Voting Regressor " + str(round(r2,2)))
    st.sidebar.text("MAE Voting Regressor " + str(round(mae,2)))

    for i in range(len(algos)):
        st.sidebar.text("*"*35)
        st.sidebar.text("R2 score for " + algos[i][0] + " " + str(round(r2_scores[i],2)))
        st.sidebar.text("MAE score for " + algos[i][0] + " " + str(round(maes[i], 2)))

