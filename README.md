This is my project that uses a linear regression ML model to predict future values. It is trained/validated on 366 prior data points to predict a future 366 data points. It is user friendly allowing various inputs and visualization of the data.

Frontend: NextJs

Backend: Flask - (Regression model in server -> regression_model.py)

Here is how to run it:

Open up your terminal and clone the github repository. (1)

Navigate to VSCode and open the file that was cloned. You should also have docker open and running.

Compile the docker-compose.yml to install required files and run the code through the use of your terminal. (2)

After finished compiling, you should see a message indicating the front-end has opened up at localhost:3000. Navigate to localhost:3000 in your browser. (3)

Code for installation:

(1): git clone https://github.com/TaranP1/linear_regression

(2): docker-compose up --build

(3): http://localhost:3000/

