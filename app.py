from flask import Flask,render_template,request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score



app = Flask(__name__)   # app is the object name.
global file
file=""

@app.route('/')
def index():  # put application's code here
    return render_template('index1.html')


# FOR UPLOAD
@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/uploadfile', methods = ['POST','GET'])
def uploadfile():
    global file
    if request.method == 'POST':
        if request.files['file'].filename=="":
            return render_template("upload.html", msg="unsuccess")
        else:
            file = request.files['file']
            file.save("dataset/" + file.filename)
            return render_template("upload.html", msg="success")


# FOR VIEW
@app.route('/view')
def viewdata():
    global file
    if file== "":
        return render_template('view.html', msg="unsuccess")
    else:
        df=pd.read_csv("dataset/"+str(file.filename))
        df.drop(['Unnamed: 0'],inplace=True,axis=1)

        return render_template('view.html',data=df.to_html())


#FOR SPLIT
@app.route('/split', methods = ['POST','GET'])
def split():
    global file
    if file== "":
        return render_template('split.html',msg="unsuccess")
    else:
        global x_train, x_test, y_train, y_test
        if request.method=="POST":
            testsize=request.form['testsize']
            if testsize== "":
                return render_template("split.html")
            else:
             df=pd.read_csv("dataset/"+str(file.filename))
             x=df.iloc[:,1:-1]      # it wont take the deleted column("Outcome")
             y=df.iloc[:,-1]
             x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(testsize))
             lentr=len(x_train)
             lentes =len(x_test)
             return render_template('split.html', msg1="done",tr1=lentr,te1=lentes)

    return render_template('split.html')



#FOR SELECT.
@app.route('/select' , methods=["POST" , "GET"])
def select():
    global file,model
    if file == "":
        return render_template('select.html', msg="unsuccess")
    else:
        global x_train, x_test, y_train, y_test
        if request.method == "POST":
            value =(request.form['select'])
            if value== "":
                return render_template('select.html')
            else:

                if value == "1":
                    model = LogisticRegression()
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    acc1 = accuracy_score(y_test, y_pred)
                    return render_template('select.html',msg="accuracy",acc=acc1,alg="LogisticRegression")

                if value == "2":
                    model = GaussianNB()
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    acc2 = accuracy_score(y_test, y_pred)
                    return render_template('select.html', msg="accuracy", acc=acc2, alg="Navie Bayes")

                if value == "3":
                    model = SVC()
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    acc3 = accuracy_score(y_test, y_pred)
                    return render_template('select.html', msg="accuracy", acc=acc3, alg="SVM")

                if value == "4":
                    model = DecisionTreeClassifier()
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    acc4 = accuracy_score(y_test, y_pred)
                    return render_template('select.html', msg="accuracy", acc=acc4, alg="DecisionTree")

                if value == "5":
                    model = KNeighborsClassifier()
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    acc5 = accuracy_score(y_test, y_pred)
                    return render_template('select.html', msg="accuracy", acc=acc5, alg="KNN")

                if value == "6":
                    model = RandomForestClassifier()
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    acc6 = accuracy_score(y_test, y_pred)
                    return render_template('select.html', msg="accuracy", acc=acc6, alg="RandomForest")

                if value == "7":
                    model = AdaBoostClassifier()
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    acc7 = accuracy_score(y_test, y_pred)
                    return render_template('select.html', msg="accuracy", acc=acc7, alg="AdaBoost")

                if value == "8":
                    model = GradientBoostingClassifier()
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    acc8 = accuracy_score(y_test, y_pred)
                    return render_template('select.html', msg="accuracy", acc=acc8, alg="Gradient Boosting")

    return render_template('select.html')


#FOR PREDICT.
@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global file
    if file=="":
        return render_template('prediction.html',msg="unsuccess")

    else:

        b=[]
        if request.method=="POST":
            Gender=request.form['Gender']
            Married=request.form['Married']
            Education=request.form['Education']
            Self_Employed=request.form['Self_Employed']
            ApplicantIncome=request.form['Applicant_Income']
            CoapplicantIncome=request.form['Coapplicant_Income']
            LoanAmount=request.form['LoanAmount']
            Loan_Amount_Term=request.form['Loan_Amount_Term']
            Credit_History=request.form['Credit_History']
            Property_Area=request.form['Property_Area']
            b.extend([Gender,Married,Education,Self_Employed,ApplicantIncome,
                      CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area])
            global model
            try:
                y_pred=model.predict([b])
            except:
                return render_template('prediction.html')
            return render_template('prediction.html',msg="success",op=y_pred)
    return render_template('prediction.html')


if __name__ == '__main__':

    app.run(debug=True)
