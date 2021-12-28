from flask import Flask,render_template,request
import pickle
import numpy as np
import lightgbm

model = pickle.load(open('LGBM_regression.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict_placement():
    building_id = int(request.form.get('building_id'))
    square_feet = float(request.form.get('square_feet'))
    air_temperature = float(request.form.get('air_temperature'))
    relative_humidity = float(request.form.get('relative_humidity'))

    meter = 0
    primary_use = 0
    cloud_coverage =0.0
    precip_depth_1_hr =0.0
    sea_level_pressure = 1000.0
    wind_direction = 140.0
    wind_speed =3.1
    hour= 2
    dayofweek = 6
    month = 1
    day = 1
    isHoliday = 1
    season = 3
    IsDayTime = 0



    # prediction
    log_energy_usage = model.predict(np.array([building_id,meter,primary_use,square_feet,air_temperature,cloud_coverage,precip_depth_1_hr,sea_level_pressure,wind_direction,wind_speed,hour,dayofweek,month,day,isHoliday,season,IsDayTime,relative_humidity]).reshape(1,3))
    energy_consumed = np.expm1(log_energy_usage)

    return render_template('index.html',result=energy_consumed)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)