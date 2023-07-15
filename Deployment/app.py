import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask('__name__')

df_1=pd.read_csv("load_data.csv")
df_1.drop('Unnamed: 0',axis=1,inplace=True)
print(df_1.shape)


data = [[0,29.85, 29.85, "Female", "Yes", "No", "No","No phone service","DSL","No","Yes","No","No","No","No",
         "Month-to-month","Yes","Electronic check",1]]


@app.route('/')
def load_app():
    return render_template('home1.html')

@app.route('/',methods=['POST'])
def predict():
    
    SeniorCitizen =  int(request.form.get('SeniorCitizen'))
    MonthlyCharges = float(request.form.get('MonthlyCharges'))
    TotalCharges =  float(request.form.get('TotalCharges'))
    Gender = request.form.get('Gender')
    Partner = request.form.get('Partner')
    Dependents = request.form.get('Dependents')
    PhoneService = request.form.get('PhoneService')
    MultipleLines = request.form.get('MultipleLines')
    InternetService = request.form.get('InternetService')
    OnlineSecurity = request.form.get('OnlineSecurity')
    OnlineBackup = request.form.get('OnlineBackup')
    DeviceProtection = request.form.get('DeviceProtection')
    TechSupport = request.form.get('TechSupport')
    StreamingTV = request.form.get('StreamingTV')
    StreamingMovies = request.form.get('StreamingMovies')
    Contract = request.form.get('Contract')
    PaperlessBilling = request.form.get('PaperlessBilling')
    PaymentMethod = request.form.get('PaymentMethod')
    Tenure = int(request.form.get('Tenure'))

    data = [[SeniorCitizen,MonthlyCharges,TotalCharges,Gender,Partner,Dependents,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,
                      DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,Tenure]]
    columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                        'PaymentMethod', 'tenure']
   
    new_df = pd.DataFrame(data,columns=columns)

    df_2 = pd.concat([df_1, new_df], ignore_index = True) 
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    #drop column customerID and tenure
    df_2.drop(columns= ['tenure'], axis=1, inplace=True)

    print(df_2.columns.values,df_2.shape)

    dummy_df = pd.get_dummies(df_2)
    print(dummy_df.shape)
    print(new_df.dtypes)
    print(dummy_df.columns.values)

    model = pickle.load(open("model.sav", "rb"))
    single = model.predict(dummy_df.tail(1))

    result = [single[0],model.predict_proba(dummy_df.tail(1))[:,1][0],model.predict_proba(dummy_df.tail(1))[:,0][0]]
    # if single == 1:
    #     probablity = model.predict_proba(dummy_df.tail(1))[:,1]
    #     # result = "Customer is likely to churn! with a confidence of {0}".format(100*probablity[0])
    #     print(result)
    # else:
    #     probablity = model.predict_proba(dummy_df.tail(1))[:,0]
    #     # result = "Customer not likely to churn! with a confidence of {0}".format(100*probablity[0])
    #     print(result)
    return render_template('home.html',result=result)    

app.debug = True 
app.run()