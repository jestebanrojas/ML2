import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from urllib import parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os
import io
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from unsupervised.dim_red import svd
import cgi

# Load Iris Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names=['setosa', 'versicolor', 'virginica']

n_componentes=2

# create an object from svd
iris_svd = svd.SVD(n_components=n_componentes)
# Apply the svd transformation
X_transformed = iris_svd.fit_transform(X)
print("X",X_transformed )
# split dataset into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
print(y_train)




# create and tarin a logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# predict with the test dataset
y_pred = model.predict(X_test)

# cald model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# HTTP Server definition Class, itis intended just to recive POSTs requests
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        print("Peticion recibida")
        content_type, pdict = cgi.parse_header(self.headers['content-type'])
        
        if content_type == 'multipart/form-data':
            # Parse the form data
            pdict['boundary'] = pdict['boundary'].encode('ascii')  # Ensure it's bytes
            form_data = cgi.parse_multipart(self.rfile, pdict)

            # Extract the field values (without double quotes)
            sepal_length = float(form_data['sepal_length'][0])
            sepal_width = float(form_data['sepal_width'][0])
            petal_length = float(form_data['petal_length'][0])
            petal_width = float(form_data['petal_width'][0])

        # create a dataset to allocate the iris params
        parametros_iris = [sepal_length, sepal_width, petal_length, petal_width]
        parametros_iris_array = np.array(parametros_iris).reshape(1, -1)
        print("Parámetros de Iris:", parametros_iris_array)

        # Apply the svd transformation
        param_transformed = iris_svd.transform(parametros_iris_array)
        print("Parámetros de Iris trasnformados:", param_transformed)

        prediccion = model.predict(param_transformed)
        print(prediccion[0])
        print("Predicción:", target_names[prediccion[0]])

        # Construct HTML response
        html_response = f"""
        <html>
        <head><title>Resultado de la predicción</title></head>
        <body>
            <h2>Resultado de la predicción</h2>
            <p>Los datos corresponden a una flor de tipo: {target_names[prediccion[0]]}</p>
        </body>
        </html>
        """

        # HTTP request response
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(html_response.encode('utf-8'))
        """
        # HTTP request response
        self.send_response(200)
        self.end_headers()

        self.wfile.write(b'Parametros recibidos')    
        self.wfile.write(b'\nLos datos corresponden a una flor de tipo: '+target_names[prediccion[0]].encode('utf-8') )  
        """

#Server init over the 8000 port
print("Iniciando el servidor...")
server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
server.serve_forever()