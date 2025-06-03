import torch
import torch.nn as nn

from model import CoffeeModel


def load_and_predict(model_path, input_data):
    model = CoffeeModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output = model(input_tensor)
        return 0 if output < 0.5 else 1


def create_input_data():
    try:
        drank_today = int(input('This is the first cup of coffee (1/0): '))
        morning = int(input("It's morning (1/0): "))
        want_sleep = int(input('You want sleep (1/0): '))
        cookies = int(input('You have cookies (1/0): '))
        return [drank_today, morning, want_sleep, cookies]
    except:
        print('Please enter a number (1/0).')



if __name__ == '__main__':

    input_data = create_input_data()
    result = load_and_predict('coffee_model.pth', input_data)

    print("Don't drink coffee" if result > 0.5 else 'Drink coffee')