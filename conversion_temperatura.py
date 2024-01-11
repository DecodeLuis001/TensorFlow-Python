import numpy as np
from sklearn.linear_model import LinearRegression

def convert_temperature():
    choice = input("Enter '1' to convert Celsius to Fahrenheit or '2' to convert Fahrenheit to Celsius: ")
    
    if choice == '1':
        celsius = float(input("Enter the temperature in Celsius: "))
        X_train = np.array([[0], [100]])  # Training data: 0°C and 100°C
        y_train = np.array([[32], [212]])  # Corresponding Fahrenheit values
        
        # Create and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict the Fahrenheit value for the given Celsius temperature
        fahrenheit = model.predict([[celsius]])
        print(f"{celsius}°C is equal to {fahrenheit[0][0]}°F")
        
    elif choice == '2':
        fahrenheit = float(input("Enter the temperature in Fahrenheit: "))
        X_train = np.array([[32], [212]])  # Training data: 32°F and 212°F
        y_train = np.array([[0], [100]])  # Corresponding Celsius values
        
        # Create and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict the Celsius value for the given Fahrenheit temperature
        celsius = model.predict([[fahrenheit]])
        print(f"{fahrenheit}°F is equal to {celsius[0][0]}°C")
        
    else:
        print("Invalid choice. Please enter either '1' or '2'.")

convert_temperature()
